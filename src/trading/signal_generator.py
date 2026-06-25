import os
import sys
from datetime import datetime

# --- FIX LỖI MÔI TRƯỜNG PYSPARK ---
conda_site_packages = "/opt/conda/lib/python3.13/site-packages"
if conda_site_packages not in sys.path:
    sys.path.insert(0, conda_site_packages)
sys.path = [p for p in sys.path if "/usr/local/spark" not in p]

os.environ["SPARK_HOME"] = os.path.join(conda_site_packages, "pyspark")
os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.config.setting import Settings

# --- CẤU HÌNH THỜI GIAN ---
if len(sys.argv) > 1:
    RUN_DATE = sys.argv[1] 
else:
    RUN_DATE = datetime.now().strftime("%Y/%m/%d")

print(f"🗓️ ĐANG TẠO TÍN HIỆU GIAO DỊCH (SIGNAL) CHO NGÀY: {RUN_DATE}")

# Khởi tạo Spark kết nối với MinIO & Iceberg
spark = SparkSession.builder \
    .appName(f"Signal_Generator_{RUN_DATE.replace('/', '_')}") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0") \
    .config("spark.hadoop.fs.s3a.endpoint", Settings.MINIO_URL) \
    .config("spark.hadoop.fs.s3a.access.key", Settings.MINIO_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", Settings.MINIO_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.my_catalog.type", "hadoop") \
    .config("spark.sql.catalog.my_catalog.warehouse", Settings.WAREHOUSE_PATH) \
    .getOrCreate()

def generate_signals():
    # Tên bảng theo file của bạn
    market_table = "my_catalog.processed_zone.daily_market_prices"
    sentiment_table = "my_catalog.processed_zone.comprehensive_sentiment_scores"
    signal_output_table = "my_catalog.processed_zone.trading_signals"

    try:
        print("📥 Đang đọc dữ liệu Giá và Điểm Cảm xúc...")
        df_market = spark.read.table(market_table)
        df_sentiment = spark.read.table(sentiment_table)

        df_sentiment = df_sentiment.withColumn("date_only", F.to_date("published_at"))

        print("🔗 Đang ghép nối (Join) Dữ liệu...")
        df_joined = df_sentiment.join(
            df_market,
            df_sentiment.date_only == df_market.trade_date,
            "inner"
        ).dropDuplicates(["id"])

        df_clean = df_joined.filter(F.col("close_price") < 1000)
        
        df_daily = df_clean.groupBy("trade_date", "close_price").agg(
            F.mean("vd_comp_summary_token").alias("avg_vader_score"),
            F.count("id").alias("news_count")
        ).orderBy("trade_date")

        print("📈 Đang tính toán SMA 20 và Z-Score Sentiment...")
        window_20 = Window.orderBy("trade_date").rowsBetween(-19, Window.currentRow)

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

        df_signals = df_processed.withColumn(
            "signal",
            F.when(
                (F.col("close_price") > F.col("sma_20")) & (F.col("sentiment_z_score") > z_threshold), 1
            ).when(
                (F.col("close_price") < F.col("sma_20")) & (F.col("sentiment_z_score") < -z_threshold), -1
            ).otherwise(0)
        )

        df_signals = df_signals.dropna(subset=["sma_20", "sentiment_z_score"])

        final_columns = [
            "trade_date", "close_price", "sma_20", 
            "avg_vader_score", "sentiment_z_score", "news_count", "signal"
        ]
        df_final = df_signals.select(*final_columns)

        print(f"💾 Đang ghi {df_final.count()} dòng Tín hiệu vào Iceberg: {signal_output_table}...")
        df_final.write.format("iceberg").mode("overwrite").saveAsTable(signal_output_table)
        print("✅ TẠO VÀ LƯU TÍN HIỆU GIAO DỊCH THÀNH CÔNG!")

    except Exception as e:
        print(f"❌ LỖI TẠO SIGNAL: {e}")

if __name__ == "__main__":
    generate_signals()
    spark.stop()