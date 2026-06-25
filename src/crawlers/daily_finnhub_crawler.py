import sys
import time
import json
import boto3
import yfinance as yf
import pandas as pd
import requests
import finnhub
import os
import sys
import re
from datetime import datetime, timedelta
from botocore.client import Config

# IMPORT CẤU HÌNH TỪ FILE SETTINGS
# Đảm bảo bạn đã thêm thư mục gốc vào sys.path nếu cần thiết
from src.config.setting import settings

# ==========================================
# CẤU HÌNH MINIO (DÙNG SETTINGS)
# ==========================================
s3_client = boto3.client(
    's3', 
    endpoint_url=settings.MINIO_URL, 
    aws_access_key_id=settings.MINIO_ACCESS_KEY, 
    aws_secret_access_key=settings.MINIO_SECRET_KEY, 
    config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}), 
    region_name='us-east-1'
)

# ==========================================
# CẤU HÌNH FINNHUB API
# ==========================================
# Khuyên dùng: Nên thêm API_KEY vào settings.py thay vì để lộ trong code
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d8p78f9r01qp954vlhggd8p78f9r01qp954vlhh0")
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ==========================================
# HÀM LƯU TRỮ VÀO MINIO (DÙNG SETTINGS)
# ==========================================
def upload_to_minio(data, folder, ticker):
    # Lấy ngày từ tham số dòng lệnh hoặc mặc định là hôm nay
    run_date = sys.argv[1].replace("-", "/") if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    
    # Sử dụng các đường dẫn đã khai báo trong settings
    # Lưu ý: RAW_NEWS_PATH và RAW_MARKET_PATH đã bao gồm cấu trúc s3a://...
    # Chúng ta cần tách phần path để phù hợp với s3_client.put_object
    object_key = f"raw_zone_finnhub_daily/{folder}/{run_date}/{ticker}.json" 
    
    json_bytes = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
    try:
        s3_client.put_object(
            Bucket=settings.BUCKET_NAME, 
            Key=object_key, 
            Body=json_bytes, 
            ContentType='application/json'
        )
        print(f"   -> ✅ Thành công lưu tại: s3://{settings.BUCKET_NAME}/{object_key}")
    except Exception as e:
        print(f"   -> ❌ Lỗi upload lên MinIO: {e}")

# ==========================================
# HÀM CÀO DỮ LIỆU DAILY (1 NGÀY)
# ==========================================
def crawl_stock_price_daily(ticker_symbol):
    print(f"⏳ Đang cào dữ liệu GIÁ (1 ngày) cho mã {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    # Rút ngắn period từ 1y xuống 1d
    df = ticker.history(period="1d", interval="1d")
    if df.empty:
        return
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    upload_to_minio(data=df.to_dict(orient="records"), folder="market_data", ticker=ticker_symbol)

def crawl_stock_news_finnhub_daily(ticker_symbol):
    print(f"📰 Đang cào dữ liệu TIN TỨC (1 ngày) cho mã {ticker_symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1) # Chỉ lùi về 1 ngày trước
    
    _from = start_date.strftime("%Y-%m-%d")
    _to = end_date.strftime("%Y-%m-%d")
    
    try:
        # Gọi API 1 lần duy nhất, không cần vòng lặp chunk 30 ngày
        news = finnhub_client.company_news(ticker_symbol, _from=_from, to=_to)
        if not news:
            print(f"   -> ℹ️ Không có tin tức mới cho {ticker_symbol} trong ngày qua.")
            return
            
        print(f"   -> Đã lấy được {len(news)} bài báo cho {ticker_symbol}.")
        upload_to_minio(data=news, folder="news_data_finnhub", ticker=ticker_symbol)
        
    except Exception as e:
        print(f"   -> ❌ Lỗi gọi API Finnhub cho {_from} đến {_to}: {e}")

# ==========================================
# HÀM LẤY DANH SÁCH MÃ S&P 500
# ==========================================
def get_sp500_tickers():
    print("⏳ Đang tải danh sách 500 mã cổ phiếu S&P 500 từ Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

# ==========================================
# KHỐI THỰC THI CHÍNH
# ==========================================
if __name__ == "__main__":
    print(f"=== BẮT ĐẦU CHẠY PIPELINE DAILY VÀO BUCKET: {settings.BUCKET_NAME} ===")
    
    danh_sach_ma = get_sp500_tickers()
    
    # LƯU Ý: Khuyến nghị chạy thử với 3-5 mã trước khi cào toàn bộ 500 mã
    # danh_sach_ma = danh_sach_ma[:5] 
    
    print(f"Tổng cộng có {len(danh_sach_ma)} mã cổ phiếu sẽ được cào.")
    
    for i, ma in enumerate(danh_sach_ma):
        print(f"\n--- Đang xử lý mã thứ {i+1}/{len(danh_sach_ma)}: {ma} ---")
        try:
            crawl_stock_price_daily(ma)
            time.sleep(1) 
            
            crawl_stock_news_finnhub_daily(ma)
            
            # API Finnhub Free giới hạn 60 request/phút. 
            # Mỗi mã gọi 1 API news -> 12 giây nghỉ đảm bảo an toàn tuyệt đối.
            print("   -> ⏳ Đang đợi 2 giây để tránh bị khóa API...")
            time.sleep(2) 
            
        except finnhub.FinnhubAPIException as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                print("   -> ⚠️ Chạm ngưỡng API! Tạm dừng hệ thống 60 giây...")
                time.sleep(60)
            else:
                print(f"   -> ❌ Lỗi API không xác định với mã {ma}: {e}")
        except Exception as e:
            print(f"   -> ❌ Lỗi hệ thống với mã {ma}: {e}")
            continue 
            
    print("\n🎉 HOÀN TẤT CÀO TOÀN BỘ DỮ LIỆU DAILY!")