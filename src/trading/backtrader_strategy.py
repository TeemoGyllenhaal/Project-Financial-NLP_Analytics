import os
import sys
import traceback
import gc
from datetime import datetime

# ==============================================================================
# --- BỘ VÁ LỖI TOÀN DIỆN CHO BACKTRADER VỚI PYTHON 3.10+ / MATPLOTLIB MỚI ---
# ==============================================================================
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Iterator = collections.abc.Iterator
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping

import numpy as np
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'long'): np.long = int

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.axes as maxes
if not hasattr(maxes, 'SubplotBase'): maxes.SubplotBase = maxes.Axes
if not hasattr(maxes, 'Subplot'): maxes.Subplot = maxes.Axes

import matplotlib.cm as mcm
if not hasattr(mcm, 'get_cmap'):
    if hasattr(matplotlib, 'colormaps'): mcm.get_cmap = matplotlib.colormaps.get_cmap
    else: mcm.get_cmap = plt.get_cmap

import matplotlib.dates as mdates
import warnings
mdates.warnings = warnings
mdates.HOURS_PER_DAY = 24.0
mdates.MINUTES_PER_DAY = 1440.0
mdates.SEC_PER_DAY = 86400.0
class MockRRuleWrapper:
    def __init__(self, *args, **kwargs): pass
if not hasattr(mdates, 'rrulewrapper'): mdates.rrulewrapper = MockRRuleWrapper

import cycler
if not hasattr(matplotlib, 'cycler'): matplotlib.cycler = cycler.cycler

import backtrader.plot.plot as bt_plot
if hasattr(bt_plot, 'Plot_OldSync'): bt_plot.Plot_OldSync.show = lambda self: None
if hasattr(bt_plot, 'Plot'): bt_plot.Plot.show = lambda self: None
# ==============================================================================

import backtrader as bt
import pandas as pd

# --- FIX LỖI MÔI TRƯỜNG PYSPARK ---
modules_to_remove = [mod for mod in sys.modules if mod.startswith('pyspark') or mod.startswith('py4j')]
for mod in modules_to_remove: 
    del sys.modules[mod]

sys.path = [p for p in sys.path if "/usr/local/spark" not in p]
if "PYTHONPATH" in os.environ: del os.environ["PYTHONPATH"]
    
airflow_site_packages = "/home/airflow/.local/lib/python3.10/site-packages"
if airflow_site_packages not in sys.path: sys.path.insert(0, airflow_site_packages)
    
os.environ["SPARK_HOME"] = os.path.join(airflow_site_packages, "pyspark")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from src.config.setting import Settings

if len(sys.argv) > 1:
    RUN_DATE = sys.argv[1] 
else:
    RUN_DATE = datetime.now().strftime("%Y/%m/%d")

print(f"🗓️ ĐANG CHẠY TIẾN TRÌNH BACKTEST ĐẾN NGÀY: {RUN_DATE}")

spark = SparkSession.builder \
    .appName(f"Backtest_Engine_{RUN_DATE.replace('/', '_')}") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0") \
    .config("spark.hadoop.fs.s3a.endpoint", Settings.MINIO_URL) \
    .config("spark.hadoop.fs.s3a.access.key", Settings.MINIO_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", Settings.MINIO_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.my_catalog.type", "hadoop") \
    .config("spark.sql.catalog.my_catalog.warehouse", Settings.WAREHOUSE_PATH) \
    .getOrCreate()


# ĐỊNH NGHĨA CUSTOM DATA FEED CHO BACKTRADER
class SignalData(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('datetime', None),
        ('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close'),
        ('volume', 'volume'), ('openinterest', -1),
        ('signal', 'signal'), 
    )

# ĐỊNH NGHĨA CHIẾN LƯỢC QUẢN LÝ DANH MỤC (PORTFOLIO STRATEGY)
class PortfolioSentimentStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        pass # Tắt log để giữ terminal sạch sẽ

    def __init__(self):
        # Lưu trữ order cho TỪNG data feed (từng mã cổ phiếu)
        self.orders = {data._name: None for data in self.datas}

    def next(self):
        # Lặp qua tất cả các mã cổ phiếu đang được nạp vào
        for data in self.datas:
            ticker = data._name
            
            # Nếu đang có lệnh chờ khớp của mã này thì bỏ qua
            if self.orders[ticker]:
                continue

            # data.signal[0] là giá trị tín hiệu của mã hiện tại ở ngày hiện tại
            if not self.getposition(data): # Chưa giữ cổ phiếu này
                if data.signal[0] == 1:
                    # Lệnh buy mặc định dùng sizer (50 cổ phiếu)
                    self.orders[ticker] = self.buy(data=data)
            else: # Đang giữ cổ phiếu này
                if data.signal[0] == -1:
                    self.orders[ticker] = self.sell(data=data)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        ticker = order.data._name
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[ticker] = None # Xóa trạng thái chờ


def run_portfolio_backtest():
    signal_table = "my_catalog.processed_zone.trading_signals"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    save_dir = os.path.join(project_root, "bao_cao_daily", RUN_DATE)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        print("📥 Đang lấy dữ liệu Tín hiệu từ Data Lake...")
        target_date = RUN_DATE.replace('/', '-')
        
        df_base = spark.read.table(signal_table) \
            .withColumn("date_only", F.to_date(F.col("trade_date"))) \
            .filter(F.col("date_only") <= F.lit(target_date))
            
        print("🔄 Đang tải toàn bộ dữ liệu thị trường về RAM...")
        raw_pandas_df = df_base.toPandas()
        
        if raw_pandas_df.empty:
            print(f"⚠️ Không có dữ liệu giao dịch nào trước ngày {target_date}!")
            return

        raw_pandas_df['trade_date'] = pd.to_datetime(raw_pandas_df['trade_date'])
        tickers = raw_pandas_df['ticker'].dropna().unique()
        
        # ==========================================
        # KHỞI TẠO 1 CEREBRO DUY NHẤT CHO TỔNG VỐN
        # ==========================================
        cerebro = bt.Cerebro()
        cerebro.addstrategy(PortfolioSentimentStrategy)

        INITIAL_CASH = 10000.0
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=0.001) 
        # Đặt khối lượng mua mặc định cho mỗi lần vào lệnh là 20 cổ phiếu (bạn có thể tự chỉnh)
        cerebro.addsizer(bt.sizers.FixedSize, stake=20) 

        print(f"🎯 NẠP DỮ LIỆU CỦA {len(tickers)} MÃ VÀO CEREBRO...")
        for ticker in tickers:
            df_ticker = raw_pandas_df[raw_pandas_df['ticker'] == ticker].copy()
            df_ticker.sort_values('trade_date', inplace=True)
            
            if len(df_ticker) < 5:
                continue

            bt_df = df_ticker.copy()
            bt_df.set_index('trade_date', inplace=True)
            for col in ['open', 'high', 'low', 'close']:
                bt_df[col] = bt_df['close_price']
            bt_df['volume'] = bt_df.get('news_count', 0)
            
            # Khởi tạo data feed
            data = SignalData(dataname=bt_df, name=ticker)
            # Nạp vào chung 1 cerebro
            cerebro.adddata(data)

        # ==========================================
        # CHẠY MÔ PHỎNG TỔNG QUÁT
        # ==========================================
        print('='*50)
        print(f'🚀 BẮT ĐẦU BACKTEST DANH MỤC TỔNG (PORTFOLIO)')
        print(f'💰 Vốn khởi điểm: ${INITIAL_CASH:.2f}')
        print('='*50)

        cerebro.run()

        final_value = cerebro.broker.getvalue()
        pnl = final_value - INITIAL_CASH
        pnl_percent = (pnl / INITIAL_CASH) * 100

        print('='*50)
        print(f'🏁 KẾT THÚC BACKTEST DANH MỤC')
        print(f'💰 Vốn cuối cùng: ${final_value:.2f}')
        if pnl >= 0:
            print(f'📈 TỔNG LỢI NHUẬN RÒNG (PnL): +${pnl:.2f} (+{pnl_percent:.2f}%)')
        else:
            print(f'📉 TỔNG THUA LỖ RÒNG (PnL): -${abs(pnl):.2f} ({pnl_percent:.2f}%)')
        print('='*50)

        # ==========================================
        # XUẤT ĐỒ THỊ CUSTOM CHO TỪNG MÃ (KHÔNG CHẠY LẠI CEREBRO)
        # ==========================================
        print("\n📊 Đang vẽ và lưu biểu đồ Custom cho từng mã...")
        z_threshold = 1.5
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Chỉ vẽ Custom plot, không vẽ Backtrader plot vì Backtrader plot
        # gom 500 mã vào 1 hình sẽ làm Crash máy tính ngay lập tức.
        for idx, ticker in enumerate(tickers, 1):
            df_ticker = raw_pandas_df[raw_pandas_df['ticker'] == ticker].copy()
            df_ticker.sort_values('trade_date', inplace=True)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1.2]})

            ax1.plot(df_ticker['trade_date'], df_ticker['close_price'], label='Giá (Close)', color='#1f77b4', linewidth=1.5)
            ax1.plot(df_ticker['trade_date'], df_ticker['sma_20'], label='SMA 20', color='#ff7f0e', linestyle='--', linewidth=1.5)
            ax1.set_title(f'[{ticker}] TÍN HIỆU CẢM XÚC TIN TỨC + SMA ({target_date})', fontsize=12, fontweight='bold')
            
            ax1.fill_between(df_ticker['trade_date'], df_ticker['close_price'], df_ticker['sma_20'], 
                             where=(df_ticker['close_price'] > df_ticker['sma_20']), color='green', alpha=0.1)
            ax1.fill_between(df_ticker['trade_date'], df_ticker['close_price'], df_ticker['sma_20'], 
                             where=(df_ticker['close_price'] < df_ticker['sma_20']), color='red', alpha=0.1)

            buy_signals = df_ticker[df_ticker['signal'] == 1]
            sell_signals = df_ticker[df_ticker['signal'] == -1]
            offset = df_ticker['close_price'].mean() * 0.02 

            ax1.scatter(buy_signals['trade_date'], buy_signals['close_price'] - offset, marker='^', color='green', s=100, label='MUA', zorder=5)
            ax1.scatter(sell_signals['trade_date'], sell_signals['close_price'] + offset, marker='v', color='red', s=100, label='BÁN', zorder=5)
            ax1.legend(loc='upper left', fontsize=9)

            colors = ['green' if x > 0 else 'red' for x in df_ticker['sentiment_z_score']]
            ax2.bar(df_ticker['trade_date'], df_ticker['sentiment_z_score'], color=colors, alpha=0.6, width=1.0)
            ax2.axhline(0, color='black', linewidth=1)
            ax2.axhline(z_threshold, color='green', linestyle=':', linewidth=1)
            ax2.axhline(-z_threshold, color='red', linestyle=':', linewidth=1)
            ax2.set_ylabel('Z-Score', fontsize=9)
            
            report_path_custom = os.path.join(save_dir, f"{ticker}_custom_plot.png")
            plt.tight_layout()
            plt.savefig(report_path_custom, bbox_inches='tight', dpi=70)
            plt.close(fig)
            gc.collect()

        print("🎉 HOÀN TẤT! Báo cáo danh mục đã được lưu thành công.")

    except Exception as e:
        print(f"❌ LỖI TRONG QUÁ TRÌNH BACKTEST:")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run_portfolio_backtest()
    spark.stop()