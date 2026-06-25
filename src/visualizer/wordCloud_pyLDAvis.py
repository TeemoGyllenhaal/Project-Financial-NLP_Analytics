import os
import sys
import re
import boto3
from datetime import datetime
import warnings

# Tắt các cảnh báo phiền phức
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 1. ÉP VERSION PYSPARK 
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
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType


# =========================================================
# 3. HÀM KHỞI TẠO SPARK
# =========================================================
def init_spark():
    spark_session = SparkSession.builder \
        .appName(f"Visualizer_WordCloud_LDAvis_{datetime.now().strftime('%Y%m%d')}") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
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

    return spark_session


# =========================================================
# 4. LUỒNG CHẠY CHÍNH (MAIN)
# =========================================================
if __name__ == "__main__":
    spark = init_spark()
    print("✅ Khởi tạo Spark hoàn tất!")

    ngay_chay = datetime.now().strftime("%Y_%m_%d")
    thu_muc_bao_cao = f"bao_cao_daily_{ngay_chay}"
    if not os.path.exists(thu_muc_bao_cao):
        os.makedirs(thu_muc_bao_cao)

    # -----------------------------------------------------
    # CẤU HÌNH MINIO S3
    # -----------------------------------------------------
    s3_client = boto3.client(
        's3', 
        endpoint_url='http://minio:9000', 
        aws_access_key_id='dataNLPmining-lab', 
        aws_secret_access_key='dataNLPmining-lab'
    )
    BUCKET_NAME = 'raw-financial-data'
    
    # -----------------------------------------------------
    # 1. ĐỌC BẢNG DỮ LIỆU ĐÃ PHÂN CHIA CHỦ ĐỀ TỪ BƯỚC TRƯỚC
    # -----------------------------------------------------
    input_table = "my_catalog.processed_zone.daily_news_topics_final"
    output_table = "my_catalog.processed_zone.visualize"
    
    print(f"📥 1. Đang đọc dữ liệu từ Iceberg table: {input_table}...")
    try:
        df_source = spark.table(input_table).filter(col("title").isNotNull())
        
        # Lấy danh sách các mã Ticker hiện có
        distinct_tickers = [row['ticker'] for row in df_source.select("ticker").distinct().collect() if row['ticker']]
        print(f"📌 Tìm thấy {len(distinct_tickers)} mã cổ phiếu cần tạo biểu đồ.")
        
        # Danh sách chứa kết quả để ghi vào bảng visualize sau cùng
        visualize_records = []

        # -----------------------------------------------------
        # 2. VÒNG LẶP XỬ LÝ TỪNG TICKER
        # -----------------------------------------------------
        for idx, ticker in enumerate(distinct_tickers, 1):
            print(f"\n[{idx}/{len(distinct_tickers)}] 🎨 Đang vẽ biểu đồ cho mã: {ticker}")
            
            # Lọc dữ liệu của mã hiện tại & giới hạn số lượng để tránh tràn RAM
            df_ticker = df_source.filter(col("ticker") == ticker).select("title").limit(2000)
            docs = df_ticker.rdd.map(lambda x: str(x[0])).collect()
            
            if len(docs) < 5:
                print(f"   ⚠️ Bỏ qua {ticker} vì chỉ có {len(docs)} bài viết (cần tối thiểu 5).")
                continue

            try:
                # Định nghĩa đường dẫn file local và S3
                local_wordcloud = os.path.join(thu_muc_bao_cao, f"wordcloud_{ticker}.png")
                local_ldavis = os.path.join(thu_muc_bao_cao, f"ldavis_{ticker}.html")
                
                s3_path_img = f"reports/daily_{ngay_chay}/ticker={ticker}/wordcloud.png"
                s3_path_html = f"reports/daily_{ngay_chay}/ticker={ticker}/ldavis.html"

                # --- A. TẠO WORDCLOUD ---
                all_words = " ".join(docs)
                wordcloud = WordCloud(
                    width=800, height=400, background_color='white', 
                    colormap='viridis', max_words=100
                ).generate(all_words)

                fig = plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f"WORDCLOUD - {ticker} ({ngay_chay})", fontsize=16, fontweight='bold', pad=15)
                plt.savefig(local_wordcloud, bbox_inches='tight', dpi=100)
                plt.close(fig) # Cực kỳ quan trọng: Giải phóng RAM đồ họa

                # --- B. TẠO PYLDAVIS (SKLEARN) ---
                tf_vectorizer = CountVectorizer(max_df=0.9, min_df=2)
                dtm = tf_vectorizer.fit_transform(docs)

                # Giảm số topic xuống 3-5 cho từng mã để biểu đồ không bị rối rắm
                n_topics = min(5, max(2, len(docs) // 3)) 
                lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5, random_state=42)
                lda_model.fit(dtm)

                vis_data = pyLDAvis.lda_model.prepare(lda_model, dtm, tf_vectorizer, mds='tsne')
                pyLDAvis.save_html(vis_data, local_ldavis)

                # --- C. UPLOAD LÊN MINIO ---
                s3_client.upload_file(local_wordcloud, BUCKET_NAME, s3_path_img)
                s3_client.upload_file(local_ldavis, BUCKET_NAME, s3_path_html)
                
                # --- D. GHI NHẬN KẾT QUẢ VÀO LIST ---
                visualize_records.append({
                    "ticker": ticker,
                    "report_date": ngay_chay,
                    "wordcloud_url": f"s3a://{BUCKET_NAME}/{s3_path_img}",
                    "ldavis_url": f"s3a://{BUCKET_NAME}/{s3_path_html}"
                })
                print(f"   ✅ Đã upload thành công Wordcloud & LDAvis cho {ticker}")

            except Exception as e:
                print(f"   ❌ Lỗi khi xử lý biểu đồ cho mã {ticker}: {e}")
                continue # Bỏ qua mã lỗi, chạy tiếp mã khác

        # -----------------------------------------------------
        # 3. GHI LOG ĐƯỜNG DẪN VÀO ICEBERG TABLE VỚI PARTITION
        # -----------------------------------------------------
        if visualize_records:
            print(f"\n⏳ Đang lưu metadata đường dẫn của {len(visualize_records)} mã vào bảng Iceberg...")
            
            # Định nghĩa cấu trúc bảng (Schema)
            schema = StructType([
                StructField("ticker", StringType(), True),
                StructField("report_date", StringType(), True),
                StructField("wordcloud_url", StringType(), True),
                StructField("ldavis_url", StringType(), True)
            ])
            
            # Chuyển List of Dictionaries thành Spark DataFrame
            df_visualize = spark.createDataFrame(visualize_records, schema)
            
            # 💡 YÊU CẦU LƯU TÁCH BIỆT: Sử dụng partitionBy("ticker")
            df_visualize.write \
                .format("iceberg") \
                .partitionBy("ticker") \
                .mode("append") \
                .saveAsTable(output_table)
                
            print(f"🎉 HOÀN TẤT LƯU BẢNG: {output_table}")
        else:
            print("⚠️ Không có biểu đồ nào được tạo thành công để lưu.")

    except Exception as e:
        print(f"❌ Lỗi truy cập bảng nguồn: {e}")

    spark.stop()
    print("👋 Đã ngắt kết nối Spark Session.")