import os
import sys
import re
import boto3
from datetime import datetime
import warnings

# Tắt các cảnh báo phiền phức từ pyLDAvis / Sklearn / Matplotlib
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1. ÉP VERSION PYSPARK (PHẢI CHẠY ĐẦU TIÊN TRƯỚC KHI IMPORT)
# =========================================================
modules_to_remove = [mod for mod in sys.modules if mod.startswith('pyspark') or mod.startswith('py4j')]
for mod in modules_to_remove: 
    del sys.modules[mod]

sys.path = [p for p in sys.path if "/usr/local/spark" not in p]
if "PYTHONPATH" in os.environ: 
    del os.environ["PYTHONPATH"]
    
conda_site_packages = "/opt/conda/lib/python3.13/site-packages"
if conda_site_packages not in sys.path: 
    sys.path.insert(0, conda_site_packages)
    
os.environ["SPARK_HOME"] = os.path.join(conda_site_packages, "pyspark")
os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"


# =========================================================
# 2. IMPORT CÁC THƯ VIỆN
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.lda_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws


# =========================================================
# 3. HÀM KHỞI TẠO SPARK
# =========================================================
def init_spark():
    """Khởi tạo cấu hình kết nối tới MinIO và Iceberg"""
    spark_session = SparkSession.builder \
        .appName("pyLDAvis_wordCloud_Pipeline") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.memory.offHeap.enabled", "true") \
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
        .config("spark.sql.catalog.my_catalog.warehouse", "s3a://raw-financial-data/iceberg_warehouse_daily") \
        .getOrCreate()

    # Vá lỗi thời gian Hadoop
    hadoop_conf = spark_session._jsc.hadoopConfiguration()
    iterator = hadoop_conf.iterator()
    while iterator.hasNext():
        entry = iterator.next()
        val = str(entry.getValue()).strip().lower()
        match = re.fullmatch(r"(\d+)([smhd])", val)
        if match:
            num, unit = int(match.group(1)), match.group(2)
            ms_val = num * 1000 if unit == 's' else num * 60000 if unit == 'm' else num * 3600000 if unit == 'h' else num * 86400000
            hadoop_conf.set(entry.getKey(), str(ms_val))
            
    return spark_session


# =========================================================
# 4. LUỒNG CHẠY CHÍNH (MAIN)
# =========================================================
if __name__ == "__main__":
    # Khởi tạo Spark
    spark = init_spark()
    print("✅ Khởi tạo Spark và môi trường hoàn tất!")

    # Cấu hình ngày chạy hệ thống và thư mục đầu ra
    ngay_chay = datetime.now().strftime("%Y_%m_%d")
    thu_muc_bao_cao = "bao_cao_daily"
    
    if not os.path.exists(thu_muc_bao_cao):
        os.makedirs(thu_muc_bao_cao)
        
    duong_dan_wordcloud = os.path.join(thu_muc_bao_cao, f"wordcloud_{ngay_chay}.png")
    duong_dan_ldavis = os.path.join(thu_muc_bao_cao, f"ldavis_{ngay_chay}.html")


    # -----------------------------------------------------
    # CẤU HÌNH MINIO S3 ĐỂ UPLOAD BÁO CÁO
    # -----------------------------------------------------
    s3_client = boto3.client(
        's3', 
        endpoint_url='http://minio:9000', 
        aws_access_key_id='dataNLPmining-lab', 
        aws_secret_access_key='dataNLPmining-lab'
    )
    BUCKET_NAME = 'raw-financial-data'
    
    # -----------------------------------------------------
    # 1. KÉO DỮ LIỆU TỪ ICEBERG VỀ PYTHON (DRIVER)
    # -----------------------------------------------------
    table_name = "my_catalog.processed_zone.news_lda_summarized"
    print("📥 1. Đang kéo 10.000 bản tin mẫu từ Iceberg về RAM...")

    # ĐỔI TÊN CỘT Ở ĐÂY: Dùng trực tiếp cột "title" (hoặc "summary")
    df_sample = spark.table(table_name) \
        .select("title") \
        .filter(col("title").isNotNull()) \
        .limit(10000)

    # Vì cột 'title' đã là dạng chuỗi (String), ta KHÔNG CẦN concat_ws nữa.
    # Lấy thẳng dữ liệu ra thành một list các chuỗi thuần túy của Python:
    docs = df_sample.rdd.map(lambda x: str(x[0])).collect()
    
    print(f"✅ Đã kéo thành công {len(docs)} bản tin!")
    if len(docs) > 0:
        # -----------------------------------------------------
        # TÁC VỤ 1: VẼ VÀ LƯU ẢNH WORDCLOUD DAILY
        # -----------------------------------------------------
        print("🎨 2. Đang tạo và ghi ảnh WordCloud...")
        all_words = " ".join(docs)
        
        if all_words.strip():
            wordcloud = WordCloud(
                width=1200, height=600, 
                background_color='white', colormap='viridis',     
                max_words=200, collocations=False      
            ).generate(all_words)

            fig = plt.figure(figsize=(16, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"WORDCLOUD - NGÀY {ngay_chay}", fontsize=20, fontweight='bold', pad=20)
            plt.savefig(duong_dan_wordcloud, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"🎉 Đã lưu ảnh WordCloud thành công tại: {duong_dan_wordcloud}")
        else:
            print("⚠️ Không có văn bản hợp lệ để vẽ WordCloud.")

        # -----------------------------------------------------
        # TÁC VỤ 2: HUẤN LUYỆN LDA VÀ LƯU FILE HTML PYLDAVIS DAILY
        # -----------------------------------------------------
        print("🧮 3. Đang xây dựng Ma trận từ vựng (DTM)...")
        tf_vectorizer = CountVectorizer(max_df=0.9, min_df=5)
        dtm = tf_vectorizer.fit_transform(docs)

        print("🧠 4. Đang mô phỏng thuật toán LDA (10 Chủ đề)...")
        lda_model = LatentDirichletAllocation(
            n_components=10,        # Chia làm 10 chủ đề
            max_iter=10,            # Số vòng lặp tối đa
            learning_method='online', 
            random_state=42         # Giữ nguyên seed định dạng biểu đồ
        )
        lda_model.fit(dtm)
        print("✅ Huấn luyện LDA hoàn tất!")

        print("📊 5. Đang kết xuất biểu đồ tương tác pyLDAvis...")
        # Tạo dữ liệu trực quan bằng hàm lda_model của Sklearn
        vis_data = pyLDAvis.lda_model.prepare(
            lda_model, 
            dtm, 
            tf_vectorizer, 
            mds='tsne'
        )

        # Lưu ngầm thành file HTML theo ngày
        pyLDAvis.save_html(vis_data, duong_dan_ldavis)
        print(f"🎉 Đã lưu báo cáo tương tác LDAvis thành công tại: {duong_dan_ldavis}")

        # -----------------------------------------------------
        # 3. UPLOAD BÁO CÁO LÊN MINIO (S3)
        # -----------------------------------------------------
        print("\n☁️ Đang đẩy báo cáo lên MinIO...")
        try:
            # Upload ảnh Wordcloud
            s3_path_img = f"reports/daily_{ngay_chay}/wordcloud.png"
            s3_client.upload_file(duong_dan_wordcloud, BUCKET_NAME, s3_path_img)
            print(f"   ✅ Đã đưa WordCloud lên S3: s3a://{BUCKET_NAME}/{s3_path_img}")

            # Upload file HTML LDA
            s3_path_html = f"reports/daily_{ngay_chay}/ldavis.html"
            s3_client.upload_file(duong_dan_ldavis, BUCKET_NAME, s3_path_html)
            print(f"   ✅ Đã đưa LDAvis lên S3: s3a://{BUCKET_NAME}/{s3_path_html}")
        except Exception as e:
            print(f"   ❌ Lỗi khi upload lên S3: {e}")
        
        
    else:
        print("⚠️ Không tìm thấy bản tin hợp lệ nào để xử lý.")

    # Dừng Spark để giải phóng tài nguyên hệ thống
    spark.stop()
    print("👋 Đã ngắt kết nối Spark Session.")