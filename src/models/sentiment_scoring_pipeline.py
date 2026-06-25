import os
import sys
import re
from datetime import datetime

# ==========================================
# FIX LỖI: ÉP HỆ THỐNG DÙNG PYSPARK 3.5.1 (TRÁNH BẢN 4.1.1)
# ==========================================
conda_site_packages = "/opt/conda/lib/python3.13/site-packages"
if conda_site_packages not in sys.path:
    sys.path.insert(0, conda_site_packages)
sys.path = [p for p in sys.path if "/usr/local/spark" not in p]

os.environ["SPARK_HOME"] = os.path.join(conda_site_packages, "pyspark")
os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

# ==========================================
# IMPORT THƯ VIỆN SAU KHI ÉP VERSION
# ==========================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, udf, when
from pyspark.sql.types import FloatType
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Tải từ điển VADER (Chạy 1 lần, nếu có rồi nó sẽ tự bỏ qua)
nltk.download('vader_lexicon')

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
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0") \
    .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
    .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.my_catalog.type", "hadoop") \
    .config("spark.sql.catalog.my_catalog.warehouse", f"s3a://{BUCKET_NAME}/iceberg_warehouse_daily") \
    .getOrCreate()

# ==========================================
# ĐỊNH NGHĨA UDF (USER DEFINED FUNCTIONS)
# ==========================================
print("🧠 Đang khởi tạo mô hình TextBlob và VADER...")
sid = SentimentIntensityAnalyzer()

def get_tb_polarity(text):
    return TextBlob(text).sentiment.polarity if text else 0.0

def get_tb_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity if text else 0.0

def get_vader_compound(text):
    return sid.polarity_scores(text)['compound'] if text else 0.0

udf_tb_pol = udf(get_tb_polarity, FloatType())
udf_tb_sub = udf(get_tb_subjectivity, FloatType())
udf_vader = udf(get_vader_compound, FloatType())

# ==========================================
# KHỐI XỬ LÝ CHÍNH
# ==========================================
def process_sentiment():
    # 1. Đọc bảng chứa dữ liệu NLP (bao gồm cả tokens và lemmas) từ bước trước
    # GIẢ ĐỊNH: Bảng này được lưu ở processed_zone.daily_news_nlp
    # Nếu nhóm bạn lưu ở tên khác, hãy sửa lại đường dẫn này:
    source_table = "my_catalog.processed_zone.daily_news_nlp"
    target_table = "my_catalog.processed_zone.comprehensive_sentiment_scores"
    
    try:
        df = spark.read.table(source_table)
        
        # (Tùy chọn) Lọc chỉ lấy dữ liệu của ngày hôm nay nếu bảng này chứa toàn bộ lịch sử
        # df = df.filter(col("published_at").cast("date") == RUN_DATE.replace("/", "-"))

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
            \
            .withColumn("tb_pol_title_token", udf_tb_pol(col("title_tokens_str"))) \
            .withColumn("tb_sub_title_token", udf_tb_sub(col("title_tokens_str"))) \
            .withColumn("vd_comp_title_token", udf_vader(col("title_tokens_str"))) \
            \
            .withColumn("tb_pol_summary_lemma", udf_tb_pol(col("summary_lemmas_str"))) \
            .withColumn("tb_sub_summary_lemma", udf_tb_sub(col("summary_lemmas_str"))) \
            .withColumn("vd_comp_summary_lemma", udf_vader(col("summary_lemmas_str"))) \
            \
            .withColumn("tb_pol_summary_token", udf_tb_pol(col("summary_tokens_str"))) \
            .withColumn("tb_sub_summary_token", udf_tb_sub(col("summary_tokens_str"))) \
            .withColumn("vd_comp_summary_token", udf_vader(col("summary_tokens_str")))

        print("🏷️ Đang gắn nhãn Tích cực/Tiêu cực/Trung tính...")
        # Lấy ngưỡng (Threshold)
        TB_POS, TB_NEG = 0.1, -0.1
        VD_POS, VD_NEG = 0.05, -0.05
        TB_SUB = 0.5  # Phải khai báo biến này để phân tách Chủ quan/Khách quan

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
            \
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
            \
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
            \
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

        # Chọn lọc TẤT CẢ các cột cần thiết cho cả Lemma và Token
        final_cols = [
            "id", "published_at", "title", "summary",
            
            # --- CỘT TITLE LEMMA ---
            "tb_pol_title_lemma", "tb_label_title_lemma",
            "tb_sub_title_lemma", "tb_sub_label_title_lemma",
            "vd_comp_title_lemma", "vd_label_title_lemma",
            
            # --- CỘT TITLE TOKEN ---
            "tb_pol_title_token", "tb_label_title_token",
            "tb_sub_title_token", "tb_sub_label_title_token",
            "vd_comp_title_token", "vd_label_title_token",
            
            # --- CỘT SUMMARY LEMMA ---
            "tb_pol_summary_lemma", "tb_label_summary_lemma",
            "tb_sub_summary_lemma", "tb_sub_label_summary_lemma",
            "vd_comp_summary_lemma", "vd_label_summary_lemma",
            
            # --- CỘT SUMMARY TOKEN ---
            "tb_pol_summary_token", "tb_label_summary_token",
            "tb_sub_summary_token", "tb_sub_label_summary_token",
            "vd_comp_summary_token", "vd_label_summary_token"
        ]
        
        df_final = df_labeled.select(*final_cols)

        print(f"💾 Đang ghi {df_final.count()} dòng dữ liệu vào Iceberg: {target_table}...")
        df_final.write \
            .format("iceberg") \
            .mode("append") \
            .saveAsTable(target_table)
            
        print("✅ Lưu dữ liệu thành công!")
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy Sentiment Pipeline: {e}")

if __name__ == "__main__":
    process_sentiment()
    spark.stop()