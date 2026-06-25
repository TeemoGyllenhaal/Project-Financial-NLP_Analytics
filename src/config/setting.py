import os

class Settings:
    # --- CẤU HÌNH MINIO ---
    # Sử dụng os.getenv(tên_biến, giá_trị_mặc_định)
    MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "dataNLPmining-lab")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "dataNLPmining-lab")
    BUCKET_NAME = os.getenv("BUCKET_NAME", "raw-financial-data")

    # --- CẤU HÌNH SPARK & ICEBERG ---
    # Các thông số cần thiết để Spark kết nối tới Iceberg Catalog
    SPARK_APP_NAME = "Financial_NLP_Pipeline"
    CATALOG_NAME = "my_catalog"
    # Đường dẫn warehouse trong MinIO
    WAREHOUSE_PATH = f"s3a://{BUCKET_NAME}/iceberg_warehouse_daily"

    # --- ĐƯỜNG DẪN DỮ LIỆU THÔ (RAW ZONES) ---
    # Gom nhóm các path để dễ quản lý trong code Crawler/Processor
    RAW_NEWS_PATH = f"s3a://{BUCKET_NAME}/raw_zone_finhub_daily/news_data_finnhub"
    RAW_MARKET_PATH = f"s3a://{BUCKET_NAME}/raw_zone_finhub_daily/market_data"

    # --- NGƯỠNG CẢM XÚC (SENTIMENT THRESHOLDS) ---
    # Để sau này bạn muốn đổi ngưỡng thì chỉ cần sửa ở đây là xong
    TB_POS = 0.1
    TB_NEG = -0.1
    VD_POS = 0.05
    VD_NEG = -0.05
    TB_SUB = 0.5  # Ngưỡng Khách quan/Chủ quan

# Khởi tạo object để các file khác dễ dàng import
settings = Settings()