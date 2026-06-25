import os
import sys
import re
from datetime import datetime

# ==========================================
# FIX LỖI: ÉP HỆ THỐNG DÙNG PYSPARK 3.5.1
# ==========================================
modules_to_remove = [mod for mod in sys.modules if mod.startswith('pyspark') or mod.startswith('py4j')]
for mod in modules_to_remove: 
    del sys.modules[mod]

sys.path = [p for p in sys.path if "/usr/local/spark" not in p]
if "PYTHONPATH" in os.environ: 
    del os.environ["PYTHONPATH"]
    
airflow_site_packages = "/home/airflow/.local/lib/python3.10/site-packages"
if airflow_site_packages not in sys.path: 
    sys.path.insert(0, airflow_site_packages)
    
os.environ["SPARK_HOME"] = os.path.join(airflow_site_packages, "pyspark")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ==========================================
# IMPORT THƯ VIỆN SAU KHI ÉP VERSION
# ==========================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, udf, when
from pyspark.sql.types import FloatType
import nltk

nltk.download('vader_lexicon', quiet=True)

# ==========================================
# CẤU HÌNH THỜI GIAN & MINIO
# ==========================================
if len(sys.argv) > 1:
    RUN_DATE = sys.argv[1] 
else:
    RUN_DATE = datetime.now().strftime("%Y/%m/%d")

print(f"🗓️ ĐANG CHẤM ĐIỂM CẢM XÚC CHO DỮ LIỆU NGÀY: {RUN_DATE}")

MINIO_ENDPOINT = os.getenv("MINIO_URL", "http://minio:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "dataNLPmining-lab")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "dataNLPmining-lab")
BUCKET_NAME = "raw-financial-data"

spark = SparkSession.builder \
    .appName(f"Sentiment_Scoring_{RUN_DATE.replace('/', '_')}") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.memory.offHeap.enabled", "false") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0") \
    .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
    .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.my_catalog.type", "hadoop") \
    .config("spark.sql.catalog.my_catalog.warehouse", f"s3a://{BUCKET_NAME}/iceberg_warehouse_daily") \
    .getOrCreate()

hadoop_conf = spark._jsc.hadoopConfiguration()
iterator = hadoop_conf.iterator()
while iterator.hasNext():
    entry = iterator.next()
    val = str(entry.getValue()).strip().lower()
    match = re.fullmatch(r"(\d+)([smhd])", val)
    if match:
        num, unit = int(match.group(1)), match.group(2)
        ms_val = num * 1000 if unit == 's' else num * 60000 if unit == 'm' else num * 3600000 if unit == 'h' else num * 86400000
        hadoop_conf.set(entry.getKey(), str(ms_val))

print("✅ Khởi tạo Spark và cấu hình Hadoop hoàn tất!")

# ==========================================
# ĐỊNH NGHĨA UDF (LAZY LOAD)
# ==========================================
print("🧠 Đang cấu hình các hàm Sentiment UDF (Lazy Load)...")

_vader_analyzer = None

def get_tb_polarity(text):
    if not text or not str(text).strip(): return 0.0
    from textblob import TextBlob
    return float(TextBlob(text).sentiment.polarity)

def get_tb_subjectivity(text):
    if not text or not str(text).strip(): return 0.0
    from textblob import TextBlob
    return float(TextBlob(text).sentiment.subjectivity)

def get_vader_compound(text):
    if not text or not str(text).strip(): return 0.0
    global _vader_analyzer
    if _vader_analyzer is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return float(_vader_analyzer.polarity_scores(text)['compound'])

udf_tb_pol = udf(get_tb_polarity, FloatType())
udf_tb_sub = udf(get_tb_subjectivity, FloatType())
udf_vader = udf(get_vader_compound, FloatType())

# ==========================================
# KHỐI XỬ LÝ CHÍNH
# ==========================================
def process_sentiment():
    source_table = "my_catalog.processed_zone.daily_news_nlp"
    target_table = "my_catalog.processed_zone.comprehensive_sentiment_scores"
    
    spark.sql("CREATE NAMESPACE IF NOT EXISTS my_catalog.processed_zone")
    
    try:
        print(f"📖 Đang đọc dữ liệu từ bảng nguồn: {source_table}")
        df = spark.read.table(source_table)
        
        print("🔄 Đang chuyển Array thành String để đưa vào mô hình...")
        df_text = df \
            .withColumn("title_lemmas_str", concat_ws(" ", col("title_lemmas"))) \
            .withColumn("summary_lemmas_str", concat_ws(" ", col("summary_lemmas"))) \
            .withColumn("title_tokens_str", concat_ws(" ", col("title_tokens"))) \
            .withColumn("summary_tokens_str", concat_ws(" ", col("summary_tokens")))

        print("⚙️ Đang tính điểm TextBlob và VADER song song...")
        df_scores = df_text \
            .withColumn("tb_pol_title_lemma", udf_tb_pol(col("title_lemmas_str"))) \
            .withColumn("tb_sub_title_lemma", udf_tb_sub(col("title_lemmas_str"))) \
            .withColumn("vd_comp_title_lemma", udf_vader(col("title_lemmas_str"))) \
            .withColumn("tb_pol_title_token", udf_tb_pol(col("title_tokens_str"))) \
            .withColumn("tb_sub_title_token", udf_tb_sub(col("title_tokens_str"))) \
            .withColumn("vd_comp_title_token", udf_vader(col("title_tokens_str"))) \
            .withColumn("tb_pol_summary_lemma", udf_tb_pol(col("summary_lemmas_str"))) \
            .withColumn("tb_sub_summary_lemma", udf_tb_sub(col("summary_lemmas_str"))) \
            .withColumn("vd_comp_summary_lemma", udf_vader(col("summary_lemmas_str"))) \
            .withColumn("tb_pol_summary_token", udf_tb_pol(col("summary_tokens_str"))) \
            .withColumn("tb_sub_summary_token", udf_tb_sub(col("summary_tokens_str"))) \
            .withColumn("vd_comp_summary_token", udf_vader(col("summary_tokens_str")))

        print("🏷️ Đang gắn nhãn Tích cực/Tiêu cực/Trung tính...")
        TB_POS, TB_NEG = 0.1, -0.1
        VD_POS, VD_NEG = 0.05, -0.05
        TB_SUB = 0.5 

        df_labeled = df_scores \
            .withColumn("tb_label_title_lemma", 
                when(col("tb_pol_title_lemma") >= TB_POS, "Tích cực 🟢")
                .when(col("tb_pol_title_lemma") <= TB_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪")) \
            .withColumn("tb_sub_label_title_lemma", 
                when(col("tb_sub_title_lemma") >= TB_SUB, "Chủ quan 🧠")
                .otherwise("Khách quan 📊")) \
            .withColumn("vd_label_title_lemma", 
                when(col("vd_comp_title_lemma") >= VD_POS, "Tích cực 🟢")
                .when(col("vd_comp_title_lemma") <= VD_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪")) \
            .withColumn("tb_label_title_token", 
                when(col("tb_pol_title_token") >= TB_POS, "Tích cực 🟢")
                .when(col("tb_pol_title_token") <= TB_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪")) \
            .withColumn("tb_sub_label_title_token", 
                when(col("tb_sub_title_token") >= TB_SUB, "Chủ quan 🧠")
                .otherwise("Khách quan 📊")) \
            .withColumn("vd_label_title_token", 
                when(col("vd_comp_title_token") >= VD_POS, "Tích cực 🟢")
                .when(col("vd_comp_title_token") <= VD_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪")) \
            .withColumn("tb_label_summary_lemma", 
                when(col("tb_pol_summary_lemma") >= TB_POS, "Tích cực 🟢")
                .when(col("tb_pol_summary_lemma") <= TB_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪")) \
            .withColumn("tb_sub_label_summary_lemma", 
                when(col("tb_sub_summary_lemma") >= TB_SUB, "Chủ quan 🧠")
                .otherwise("Khách quan 📊")) \
            .withColumn("vd_label_summary_lemma", 
                when(col("vd_comp_summary_lemma") >= VD_POS, "Tích cực 🟢")
                .when(col("vd_comp_summary_lemma") <= VD_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪")) \
            .withColumn("tb_label_summary_token", 
                when(col("tb_pol_summary_token") >= TB_POS, "Tích cực 🟢")
                .when(col("tb_pol_summary_token") <= TB_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪")) \
            .withColumn("tb_sub_label_summary_token", 
                when(col("tb_sub_summary_token") >= TB_SUB, "Chủ quan 🧠")
                .otherwise("Khách quan 📊")) \
            .withColumn("vd_label_summary_token", 
                when(col("vd_comp_summary_token") >= VD_POS, "Tích cực 🟢")
                .when(col("vd_comp_summary_token") <= VD_NEG, "Tiêu cực 🔴")
                .otherwise("Trung tính ⚪"))

        # --- ĐÃ THÊM CỘT "ticker" VÀO DANH SÁCH LƯU ---
        final_cols = [
            "ticker", "id", "published_at", "title", "summary",
            "tb_pol_title_lemma", "tb_label_title_lemma",
            "tb_sub_title_lemma", "tb_sub_label_title_lemma",
            "vd_comp_title_lemma", "vd_label_title_lemma",
            "tb_pol_title_token", "tb_label_title_token",
            "tb_sub_title_token", "tb_sub_label_title_token",
            "vd_comp_title_token", "vd_label_title_token",
            "tb_pol_summary_lemma", "tb_label_summary_lemma",
            "tb_sub_summary_lemma", "tb_sub_label_summary_lemma",
            "vd_comp_summary_lemma", "vd_label_summary_lemma",
            "tb_pol_summary_token", "tb_label_summary_token",
            "tb_sub_summary_token", "tb_sub_label_summary_token",
            "vd_comp_summary_token", "vd_label_summary_token"
        ]
        
        df_final = df_labeled.select(*final_cols)

        print(f"💾 Đang ghi bổ sung dữ liệu vào Iceberg: {target_table}...")
        
        # --- ĐÃ THÊM LỆNH PARTITION THEO TICKER ---
        df_final.write \
            .format("iceberg") \
            .partitionBy("ticker") \
            .mode("append") \
            .saveAsTable(target_table)
            
        print("🎉 Xử lý chấm điểm cảm xúc hoàn tất và lưu thành công (Đã phân tách theo mã)!")
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy Sentiment Pipeline: {e}")

if __name__ == "__main__":

    process_sentiment()
    spark.stop()