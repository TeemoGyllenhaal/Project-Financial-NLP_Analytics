import os
import sys
import re
from datetime import datetime 

BUCKET_NAME = "raw-financial-data"
RUN_DATE = datetime.now().strftime("%Y/%m/%d")

print(f"📌 Đang thiết lập cấu hình cho BUCKET: {BUCKET_NAME} | NGÀY: {RUN_DATE}")

# =========================================================
# 1. ÉP VERSION VÀ THIẾT LẬP MÔI TRƯỜNG PYSPARK
# =========================================================
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

# =========================================================
# 2. KHỞI TẠO SPARK SESSION
# =========================================================
from pyspark.sql import SparkSession
# IMPORT THÊM CÁC HÀM XỬ LÝ CHUỖI VÀ TÊN FILE
from pyspark.sql.functions import col, udf, to_timestamp, to_date, input_file_name, regexp_extract
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

spark = SparkSession.builder \
    .appName("Processor_daily") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.memory.offHeap.enabled", "false") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "dataNLPmining-lab") \
    .config("spark.hadoop.fs.s3a.secret.key", "dataNLPmining-lab") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.my_catalog.type", "hadoop") \
    .config("spark.sql.catalog.my_catalog.warehouse", f"s3a://{BUCKET_NAME}/iceberg_warehouse_daily") \
    .getOrCreate()

# Vá lỗi thời gian Hadoop
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

print("✅ Khởi tạo Spark và môi trường hoàn tất!")
spark.sql("CREATE NAMESPACE IF NOT EXISTS my_catalog.processed_zone")

# ==========================================
# HÀM UDF XỬ LÝ NGÔN NGỮ (spaCy)
# ==========================================
print("🧠 Đã thiết lập hàm NLP UDF (Lazy Load)...")

nlp_schema = StructType([
    StructField("tokens", ArrayType(StringType()), False),
    StructField("lemmas", ArrayType(StringType()), False)
])

_nlp_model = None

def extract_tokens_and_lemmas(text):
    global _nlp_model
    import spacy
    import re
    
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    if not text: return {"tokens": [], "lemmas": []}
        
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text: return {"tokens": [], "lemmas": []}
        
    doc = _nlp_model(text)
    tokens_list = []
    lemmas_list = []
    for token in doc:
        if not token.is_stop and len(token.text.strip()) > 1:
            tokens_list.append(token.text)
            tokens_list.append(token.lemma_)
            
    return {"tokens": tokens_list, "lemmas": lemmas_list}

dual_nlp_udf = udf(extract_tokens_and_lemmas, nlp_schema)

# ==========================================
# PIPELINE 1: XỬ LÝ DỮ LIỆU TIN TỨC
# ==========================================
def process_daily_news():
    print(f"🚀 ĐANG XỬ LÝ TIN TỨC NGÀY {RUN_DATE}...")
    path_news = f"s3a://{BUCKET_NAME}/raw_zone_finnhub_daily/news_data_finnhub/{RUN_DATE}/*.json"
    
    try:
        df_news_raw = spark.read.format("json").load(path_news)
        
        # 💡 TUYỆT CHIÊU: Lấy tên file hiện tại và trích xuất ra mã ticker
        # Ví dụ: file_path là "s3a://.../AAPL.json" -> ticker sẽ là "AAPL"
        df_news_with_ticker = df_news_raw.withColumn(
            "ticker", 
            regexp_extract(input_file_name(), r'([^/]+)\.json$', 1)
        )
        
        df_news_clean = df_news_with_ticker.select(
            col("ticker"),  # Thêm cột ticker vào dữ liệu sạch
            col("id"),
            col("headline").alias("title"),
            col("summary"),
            to_timestamp(col("datetime")).alias("published_at")
        )
        
        df_processed = df_news_clean \
            .withColumn("title_nlp", dual_nlp_udf(col("title"))) \
            .withColumn("summary_nlp", dual_nlp_udf(col("summary")))
            
        df_final = df_processed.select(
            col("ticker"), col("id"), col("published_at"), col("title"), col("summary"),
            col("title_nlp.tokens").alias("title_tokens"),
            col("title_nlp.lemmas").alias("title_lemmas"),
            col("summary_nlp.tokens").alias("summary_tokens"),
            col("summary_nlp.lemmas").alias("summary_lemmas")
        )
        
        # 💡 YÊU CẦU LƯU TÁCH BIỆT: Sử dụng partitionBy("ticker")
        df_final.write \
            .format("iceberg") \
            .partitionBy("ticker") \
            .mode("append") \
            .saveAsTable("my_catalog.processed_zone.daily_news_nlp")
            
        print(f"   ✅ Đã APPEND xong bảng News! (Thêm {df_final.count()} dòng, tự động chia folder theo mã)")
    except Exception as e:
        print(f"   -> ⚠️ Lỗi hoặc không có dữ liệu tin tức trong ngày {RUN_DATE}: {e}")

# ==========================================
# PIPELINE 2: XỬ LÝ DỮ LIỆU CHỨNG KHOÁN
# ==========================================
def process_daily_market():
    print(f"\n🚀 ĐANG XỬ LÝ GIÁ CHỨNG KHOÁN NGÀY {RUN_DATE}...")
    path_market = f"s3a://{BUCKET_NAME}/raw_zone_finnhub_daily/market_data/{RUN_DATE}/*.json"
    
    try:
        df_market_raw = spark.read.format("json").load(path_market)
        
        # Tương tự, lấy ticker từ tên file
        df_market_with_ticker = df_market_raw.withColumn(
            "ticker", 
            regexp_extract(input_file_name(), r'([^/]+)\.json$', 1)
        )
        
        df_market_clean = df_market_with_ticker.select(
            col("ticker"), # Giữ lại mã chứng khoán
            to_date(col("Date")).alias("trade_date"),
            col("Open").alias("open_price"),
            col("High").alias("high_price"),
            col("Low").alias("low_price"),
            col("Close").alias("close_price"),
            col("Volume").alias("volume")
        )
        
        # Ghi APPEND vào Iceberg và phân tách vật lý bằng partition
        df_market_clean.write \
            .format("iceberg") \
            .partitionBy("ticker") \
            .mode("append") \
            .saveAsTable("my_catalog.processed_zone.daily_market_prices")
            
        print(f"   ✅ Đã APPEND xong bảng Market! (Thêm {df_market_clean.count()} dòng, tự động chia folder theo mã)")
    except Exception as e:
        print(f"   -> ⚠️ Lỗi hoặc không có dữ liệu chứng khoán trong ngày {RUN_DATE}: {e}")

# ==========================================
# KHỐI THỰC THI CHÍNH
# ==========================================
if __name__ == "__main__":
    process_daily_news()
    process_daily_market()
    print("\n🎉 HOÀN TẤT XỬ LÝ DATA CHO NGÀY HÔM NAY!")
    spark.stop()