import os
import sys
import numpy as np
from datetime import datetime
from itertools import chain

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
# IMPORT THƯ VIỆN SPARK & ML
# ==========================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, split, expr, create_map, lit
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml.functions import vector_to_array  # <-- THÊM IMPORT NÀY

# ==========================================
# CẤU HÌNH THỜI GIAN & MINIO
# ==========================================
if len(sys.argv) > 1:
    RUN_DATE = sys.argv[1] 
else:
    RUN_DATE = datetime.now().strftime("%Y/%m/%d")

print(f"🗓️ ĐANG CHẠY MÔ HÌNH TÓM TẮT CHỦ ĐỀ (LDA) CHO NGÀY: {RUN_DATE}")

MINIO_ENDPOINT = os.getenv("MINIO_URL", "http://minio:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "dataNLPmining-lab")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "dataNLPmining-lab")
BUCKET_NAME = "raw-financial-data"

spark = SparkSession.builder \
    .appName(f"LDA_Topic_Modeling_By_Ticker_{RUN_DATE.replace('/', '_')}") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
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
# CẤU HÌNH THAM SỐ LDA
# ==========================================
NUM_TOPICS = 10        # Số lượng chủ đề kỳ vọng
MAX_ITER = 20          # Số vòng lặp huấn luyện
NUM_KEYWORDS = 5       # Số lượng từ khóa tóm tắt

def run_lda_pipeline():
    input_table = "my_catalog.processed_zone.comprehensive_sentiment_scores"
    
    try:
        df = spark.read.table(input_table)
        df = df.dropna(subset=["vd_comp_title_token", "vd_comp_summary_token"])
        
        # Tiền xử lý văn bản chung cho toàn bộ DataFrame
        df = df.withColumn("combined_tokens_str", concat_ws(" ", col("vd_comp_title_token"), col("vd_comp_summary_token")))
        df = df.withColumn("tokens_array", split(col("combined_tokens_str"), " "))
        
        # 1. Lấy danh sách tất cả các mã (tickers) có trong dữ liệu
        print("\n--- 1. Quét danh sách các mã cổ phiếu (Tickers) ---")
        distinct_tickers_rows = df.select("ticker").distinct().collect()
        tickers_list = [row['ticker'] for row in distinct_tickers_rows if row['ticker'] is not None]
        print(f"📌 Tìm thấy {len(tickers_list)} mã cổ phiếu cần xử lý.")

        final_result_df = None # DataFrame lưu trữ tổng hợp kết quả của tất cả các mã

        # 2. Vòng lặp huấn luyện LDA cho TỪNG MÃ CỔ PHIẾU
        for idx, current_ticker in enumerate(tickers_list, 1):
            print(f"\n[{idx}/{len(tickers_list)}] 🚀 Đang xử lý LDA cho mã: {current_ticker}")
            
            # Lọc dữ liệu theo mã hiện tại
            df_ticker = df.filter(col("ticker") == current_ticker)
            doc_count = df_ticker.count()
            
            # Bỏ qua nếu mã này có quá ít bài viết (không đủ để gom cụm)
            if doc_count < 5:
                print(f"   ⚠️ Bỏ qua {current_ticker} vì chỉ có {doc_count} bài viết (cần tối thiểu 5).")
                continue
            
            # Tự động giảm số lượng Topic nếu số lượng bài báo ít hơn NUM_TOPICS
            actual_k = min(NUM_TOPICS, doc_count // 2)
            actual_k = max(2, actual_k) # Đảm bảo có ít nhất 2 cụm

            # Tạo CountVectorizer
            cv = CountVectorizer(inputCol="tokens_array", outputCol="features", vocabSize=5000, minDF=1.0)
            cv_model = cv.fit(df_ticker)
            df_features = cv_model.transform(df_ticker)
            vocab = cv_model.vocabulary

            # Huấn luyện LDA
            lda = LDA(k=actual_k, maxIter=MAX_ITER, featuresCol="features")
            lda_model = lda.fit(df_features)

            # Trích xuất từ khóa cho từng Chủ đề
            topics = lda_model.describeTopics(maxTermsPerTopic=NUM_KEYWORDS).collect()
            topic_summary_dict = {}
            for row in topics:
                topic_id = row['topic']
                words = [vocab[idx] for idx in row['termIndices']]
                topic_summary_dict[topic_id] = ", ".join(words)
            
            # Phân loại bài báo
            df_ticker_result = lda_model.transform(df_features)

            # =========================================================================
            # KỸ THUẬT NÂNG CAO: Thay thế UDF bằng Native Spark SQL để tránh lỗi Closure
            # =========================================================================
            
            # BƯỚC 1: Ép kiểu Vector sang Array thuần túy
            df_ticker_result = df_ticker_result.withColumn(
                "topicDistributionArray", 
                vector_to_array(col("topicDistribution"))
            )

            # BƯỚC 2: Lấy ID của topic có xác suất cao nhất từ mảng vừa tạo
            df_ticker_result = df_ticker_result.withColumn(
                "dominant_topic_id", 
                expr("array_position(topicDistributionArray, array_max(topicDistributionArray)) - 1")
            )

            # Map Topic ID sang danh sách từ khóa mà không cần dùng UDF
            mapping_expr = create_map([lit(x) for x in chain(*topic_summary_dict.items())])
            df_ticker_result = df_ticker_result.withColumn(
                "lda_summary_keywords", 
                mapping_expr[col("dominant_topic_id")]
            )
            
            # Vì ta không dùng dictionary cố định nữa, topic_name sẽ lấy luôn các từ khóa chính
            df_ticker_result = df_ticker_result.withColumn(
                "topic_name", 
                concat_ws(" - ", lit("Chủ đề"), col("dominant_topic_id"), col("lda_summary_keywords"))
            )

            # Dọn dẹp các cột trung gian
            df_ticker_result = df_ticker_result.drop("features", "topicDistribution", "topicDistributionArray")
            
            # Nối (Union) kết quả của mã hiện tại vào DataFrame tổng
            if final_result_df is None:
                final_result_df = df_ticker_result
            else:
                final_result_df = final_result_df.unionByName(df_ticker_result)

        # 3. Ghi dữ liệu vào Apache Iceberg MỘT LẦN DUY NHẤT VỚI PARTITION
        if final_result_df is not None:
            # 💡 CHỈ CHỌN NHỮNG CỘT CẦN THIẾT ĐỂ TRÁNH LỖI SCHEMA MISMATCH
            final_df_to_save = final_result_df.select(
                "ticker", "id", "published_at", "title", "summary",
                "dominant_topic_id", "lda_summary_keywords", "topic_name"
            )
            
            print(f"\n⏳ Đang ghi {final_df_to_save.count()} dòng dữ liệu tổng hợp vào Apache Iceberg...")
            print("🗂️ Chế độ lưu: Phân vùng tách biệt theo từng Ticker (partitionBy).")
            
            # Đổi tên bảng thành 'daily_news_topics_final' để tạo khuôn mới
            final_df_to_save.write \
                .format("iceberg") \
                .partitionBy("ticker") \
                .mode("append") \
                .saveAsTable("my_catalog.processed_zone.daily_news_topics_final")
                
            print("🎉 HOÀN TẤT LDA CHẠY TỰ ĐỘNG THEO TỪNG MÃ VÀ ĐÃ LƯU VÀO MINIO!")
        else:
            print("⚠️ Không có dữ liệu hợp lệ nào được xử lý.")
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy LDA Pipeline: {e}")

if __name__ == "__main__":
    run_lda_pipeline()
    spark.stop()