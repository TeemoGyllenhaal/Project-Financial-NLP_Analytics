import os
import sys
import re
import numpy as np
from datetime import datetime

# ==========================================
# FIX LỖI: ÉP HỆ THỐNG DÙNG PYSPARK 3.5.1
# ==========================================
conda_site_packages = "/opt/conda/lib/python3.13/site-packages"
if conda_site_packages not in sys.path:
    sys.path.insert(0, conda_site_packages)
sys.path = [p for p in sys.path if "/usr/local/spark" not in p]

os.environ["SPARK_HOME"] = os.path.join(conda_site_packages, "pyspark")
os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

# ==========================================
# IMPORT THƯ VIỆN SPARK & ML
# ==========================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat_ws, split
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA

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
    .appName(f"LDA_Topic_Modeling_{RUN_DATE.replace('/', '_')}") \
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
NUM_TOPICS = 10        # Số lượng chủ đề
MAX_ITER = 20          # Số vòng lặp huấn luyện
NUM_KEYWORDS = 5       # Số lượng từ khóa tóm tắt

def run_lda_pipeline():
    # 1. Bảng dữ liệu đầu vào (Lấy từ bảng Sentiment tổng hợp ở bước trước)
    # LƯU Ý: Nếu tên bảng của bạn khác, hãy sửa ở đây
    input_table = "my_catalog.processed_zone.comprehensive_sentiment_scores"
    output_table = "my_catalog.processed_zone.news_lda_summarized"
    
    try:
        df = spark.read.table(input_table)
        
        # Chỉ lấy dữ liệu của ngày hôm nay nếu chạy Daily
        # df = df.filter(col("published_at").cast("date") == RUN_DATE.replace("/", "-"))
        
        # Lọc bỏ dòng rỗng
        df = df.dropna(subset=["vd_comp_title_token", "vd_comp_summary_token"])
        
        print("\n--- 1. Tiền xử lý văn bản ---")
        # Gộp title và summary lại, sau đó split thành mảng (ArrayType) để đưa vào CountVectorizer
        df = df.withColumn("combined_tokens_str", concat_ws(" ", col("vd_comp_title_token"), col("vd_comp_summary_token")))
        df = df.withColumn("tokens_array", split(col("combined_tokens_str"), " "))

        print("\n--- 2. Tạo CountVectorizer (Ma trận tần suất từ) ---")
        cv = CountVectorizer(inputCol="tokens_array", outputCol="features", vocabSize=10000, minDF=2.0)
        cv_model = cv.fit(df)
        df_features = cv_model.transform(df)
        vocab = cv_model.vocabulary

        print(f"\n--- 3. Bắt đầu huấn luyện LDA với {NUM_TOPICS} Chủ đề ---")
        lda = LDA(k=NUM_TOPICS, maxIter=MAX_ITER, featuresCol="features")
        lda_model = lda.fit(df_features)

        print("\n--- 4. Trích xuất từ khóa cho từng Chủ đề ---")
        topics = lda_model.describeTopics(maxTermsPerTopic=NUM_KEYWORDS).collect()
        topic_summary_dict = {}
        
        for row in topics:
            topic_id = row['topic']
            words = [vocab[idx] for idx in row['termIndices']]
            topic_summary_dict[topic_id] = ", ".join(words)
            print(f"   Chủ đề {topic_id}: {topic_summary_dict[topic_id]}")

        print("\n--- 5. Phân loại bài báo và gắn Tóm tắt ---")
        df_result = lda_model.transform(df_features)

        # UDF Lấy ID chủ đề chiếm tỷ trọng cao nhất
        @udf(returnType=IntegerType())
        def get_dominant_topic(topic_distribution):
            return int(np.argmax(topic_distribution))

        # UDF Lấy từ khóa tóm tắt
        @udf(returnType=StringType())
        def get_topic_summary(topic_id):
            return topic_summary_dict.get(topic_id, "Không xác định")

        # Áp dụng hàm
        df_result = df_result.withColumn("dominant_topic_id", get_dominant_topic(col("topicDistribution")))
        df_result = df_result.withColumn("lda_summary_keywords", get_topic_summary(col("dominant_topic_id")))

        # --- LƯU Ý QUAN TRỌNG VỀ TÊN CHỦ ĐỀ (HARDCODE) ---
        # Tên chủ đề được gán cứng dựa trên kết quả chạy thử nghiệm trước đây.
        # Khi chạy trên dữ liệu mới mỗi ngày, mô hình có thể tự định nghĩa lại chủ đề khác đi.
        topic_names_dict = {
            0: "Cổ phiếu Cổ tức & Tăng trưởng",
            1: "Xu hướng AI & Vốn hóa tỷ đô",
            2: "So sánh Năng lực Cạnh tranh",
            3: "Phân tích Biến động Giá Cổ phiếu",
            4: "Chỉ số Vĩ mô & Hàng hóa",
            5: "Khuyến nghị từ Chuyên gia",
            6: "Đầu tư Năng lượng & Công nghệ Mới",
            7: "Tin tức Sự kiện & Ra mắt trong ngày",
            8: "Tác động của Yếu tố Chính trị - Vĩ mô",
            9: "Mùa Báo cáo Tài chính Quý"
        }

        @udf(returnType=StringType())
        def get_topic_name(topic_id):
            return topic_names_dict.get(topic_id, "Chủ đề Khác")

        df_result = df_result.withColumn("topic_name", get_topic_name(col("dominant_topic_id")))

        # Xóa các cột trung gian nặng nề của ML để tiết kiệm ổ cứng
        df_final = df_result.drop("combined_tokens_str", "tokens_array", "features", "topicDistribution")

        print(f"\n⏳ Đang ghi dữ liệu vào Apache Iceberg: {output_table} ...")
        # Dùng 'append' nếu chạy daily, hoặc 'overwrite' nếu đang test toàn bộ
        df_final.write.format("iceberg").mode("append").saveAsTable(output_table)

        print("🎉 HOÀN TẤT LDA CHẠY TỰ ĐỘNG!")
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy LDA Pipeline: {e}")

if __name__ == "__main__":
    run_lda_pipeline()
    spark.stop()