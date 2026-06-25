import sys
import time
import json
import boto3
import yfinance as yf
import pandas as pd
import requests
import finnhub
import os
import re
from datetime import datetime, timedelta
from botocore.client import Config

# IMPORT CẤU HÌNH TỪ FILE SETTINGS
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
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d8p78f9r01qp954vlhggd8p78f9r01qp954vlhh0")
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ==========================================
# HÀM LƯU TRỮ VÀO MINIO
# ==========================================
def upload_to_minio(data, folder, ticker):
    run_date = sys.argv[1].replace("-", "/") if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
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
# HÀM CÀO DỮ LIỆU (20 NGÀY)
# ==========================================
def crawl_stock_price_20days(ticker_symbol):
    print(f"⏳ Đang cào dữ liệu GIÁ (20 ngày gần nhất) cho mã {ticker_symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20)
    
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
    
    if df.empty:
        return
        
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    upload_to_minio(data=df.to_dict(orient="records"), folder="market_data", ticker=ticker_symbol)

def crawl_stock_news_finnhub_20days(ticker_symbol):
    print(f"📰 Đang cào dữ liệu TIN TỨC (20 ngày gần nhất) cho mã {ticker_symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20) # Lấy dữ liệu 20 ngày
    
    _from = start_date.strftime("%Y-%m-%d")
    _to = end_date.strftime("%Y-%m-%d")
    
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=_from, to=_to)
        if not news:
            print(f"   -> ℹ️ Không có tin tức mới cho {ticker_symbol} trong 20 ngày qua.")
            return
            
        print(f"   -> Đã lấy được {len(news)} bài báo cho {ticker_symbol}.")
        upload_to_minio(data=news, folder="news_data_finnhub", ticker=ticker_symbol)
        
    except Exception as e:
        print(f"   -> ❌ Lỗi gọi API Finnhub cho {_from} đến {_to}: {e}")
        raise e # Re-raise để khối try-except bên ngoài bắt được mã lỗi 429 nếu có

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
    print(f"=== BẮT ĐẦU CHẠY PIPELINE (20 DAYS) VÀO BUCKET: {settings.BUCKET_NAME} ===")
    
    danh_sach_ma = get_sp500_tickers()
    print(f"Tổng cộng có {len(danh_sach_ma)} mã cổ phiếu sẽ được cào.")
    
    # Số giây tối thiểu cần thiết cho mỗi mã để đảm bảo an toàn API Finnhub (60 requests / 60 seconds)
    # Cài đặt ở mức 1.05 giây để an toàn tuyệt đối, tránh burst limit
    SAFE_DELAY_PER_TICKER = 1.05 
    
    for i, ma in enumerate(danh_sach_ma):
        print(f"\n--- Đang xử lý mã thứ {i+1}/{len(danh_sach_ma)}: {ma} ---")
        start_time = time.time()
        
        try:
            # 1. Cào Giá
            crawl_stock_price_20days(ma)
            
            # 2. Cào Tin tức
            crawl_stock_news_finnhub_20days(ma)
            
            # 3. Tính toán độ trễ động (Dynamic Delay)
            elapsed_time = time.time() - start_time
            if elapsed_time < SAFE_DELAY_PER_TICKER:
                sleep_time = SAFE_DELAY_PER_TICKER - elapsed_time
                print(f"   -> ⚡ Đã xử lý trong {elapsed_time:.2f}s. Tự động đợi thêm {sleep_time:.2f}s để tối ưu rate limit...")
                time.sleep(sleep_time)
            else:
                print(f"   -> ⚡ Đã xử lý trong {elapsed_time:.2f}s. Không cần đợi thêm.")
                
        except finnhub.FinnhubAPIException as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                print("   -> ⚠️ Chạm ngưỡng API Finnhub! Tạm dừng hệ thống 60 giây...")
                time.sleep(60)
            else:
                print(f"   -> ❌ Lỗi API không xác định với mã {ma}: {e}")
        except Exception as e:
            print(f"   -> ❌ Lỗi hệ thống với mã {ma}: {e}")
            continue 
            
    print("\n🎉 HOÀN TẤT CÀO TOÀN BỘ DỮ LIỆU 20 NGÀY!")