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
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ==========================================
# CẤU HÌNH THỜI GIAN & MINIO
# ==========================================
if len(sys.argv) > 1:
    RUN_DATE = sys.argv[1] 
else:
    RUN_DATE = datetime.now().strftime("%Y/%m/%d")

print(f"🗓️ ĐANG TẠO TÍN HIỆU GIAO DỊCH (SIGNAL) CHO NGÀY: {RUN_DATE}")

MINIO_ENDPOINT = os.getenv("MINIO_URL", "http://minio:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "dataNLPmining-lab")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "dataNLPmining-lab")
BUCKET_NAME = "raw-financial-data"

spark = SparkSession.builder \
    .appName(f"Signal_Generator_{RUN_DATE.replace('/', '_')}") \
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

def generate_signals():
    market_table = "my_catalog.processed_zone.daily_market_prices"
    sentiment_table = "my_catalog.processed_zone.comprehensive_sentiment_scores"
    signal_output_table = "my_catalog.processed_zone.trading_signals"

    try:
        print("📥 Đang đọc dữ liệu Giá và Điểm Cảm xúc...")
        df_market = spark.read.table(market_table)
        df_sentiment = spark.read.table(sentiment_table)

        # Đổi tên cột ngày để join cho dễ
        df_sentiment = df_sentiment.withColumn("trade_date", F.to_date("published_at"))

        print("🔗 Đang ghép nối (Join) Dữ liệu CHÍNH XÁC THEO MÃ CHỨNG KHOÁN (TICKER)...")
        # Join bằng cả 'ticker' và 'trade_date'
        df_joined = df_sentiment.join(
            df_market,
            on=["ticker", "trade_date"],
            how="inner"
        ).dropDuplicates(["id"])

        # Nhóm theo Ticker và Ngày để lấy trung bình cảm xúc của ngày hôm đó
        df_daily = df_joined.groupBy("ticker", "trade_date", "close_price").agg(
            F.mean("vd_comp_summary_token").alias("avg_vader_score"),
            F.count("id").alias("news_count")
        )

        print("📈 Đang tính toán SMA 20 và Z-Score Sentiment THEO TỪNG MÃ CỔ PHIẾU...")
        # Thêm partitionBy("ticker") để máy tính hiểu phải vẽ đường SMA riêng cho từng hãng
        window_20 = Window.partitionBy("ticker").orderBy("trade_date").rowsBetween(-19, Window.currentRow)

        df_processed = df_daily.withColumn(
            "sma_20", F.avg("close_price").over(window_20)
        ).withColumn(
            "sentiment_mean_20", F.avg("avg_vader_score").over(window_20)
        ).withColumn(
            "sentiment_std_20", F.stddev("avg_vader_score").over(window_20)
        )

        df_processed = df_processed.withColumn(
            "sentiment_z_score", 
            (F.col("avg_vader_score") - F.col("sentiment_mean_20")) / F.when(F.col("sentiment_std_20") == 0, 1).otherwise(F.col("sentiment_std_20"))
        )

        z_threshold = 1.5

        # Logic Tín hiệu
        df_signals = df_processed.withColumn(
            "signal",
            F.when(
                (F.col("close_price") > F.col("sma_20")) & (F.col("sentiment_z_score") > z_threshold), 1
            ).when(
                (F.col("close_price") < F.col("sma_20")) & (F.col("sentiment_z_score") < -z_threshold), -1
            ).otherwise(0)
        )

        # Xóa những ngày đầu tiên chưa đủ 20 ngày để tính SMA
        df_signals = df_signals.dropna(subset=["sma_20", "sentiment_z_score"])

        # Thêm 'ticker' vào danh sách cột
        final_columns = [
            "ticker", "trade_date", "close_price", "sma_20", 
            "avg_vader_score", "sentiment_z_score", "news_count", "signal"
        ]
        df_final = df_signals.select(*final_columns)

        print(f"💾 Đang ghi {df_final.count()} dòng Tín hiệu vào Iceberg: {signal_output_table}...")
        
        # Lưu phân mảnh (Partition) theo ticker và dùng Overwrite
        df_final.write \
            .format("iceberg") \
            .partitionBy("ticker") \
            .mode("overwrite") \
            .saveAsTable(signal_output_table)
            
        print("✅ TẠO VÀ LƯU TÍN HIỆU GIAO DỊCH (ĐÃ CHIA TICKER) THÀNH CÔNG!")

    except Exception as e:
        print(f"❌ LỖI TẠO SIGNAL: {e}")

if __name__ == "__main__":
    # DỌN DẸP BẢNG CŨ TRƯỚC KHI LƯU CẤU TRÚC PHÂN MẢNH MỚI
    print("🧹 Đang dọn dẹp bảng Trading Signal cũ...")
    spark.sql("DROP TABLE IF EXISTS my_catalog.processed_zone.trading_signals")
    
    generate_signals()
    spark.stop()