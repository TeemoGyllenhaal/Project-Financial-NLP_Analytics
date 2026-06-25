import os
import sys
from datetime import datetime
import backtrader as bt
import pandas as pd
import matplotlib

# QUAN TRỌNG CHO AIRFLOW: Chuyển backend đồ họa sang 'Agg' để không yêu cầu giao diện (UI)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- FIX LỖI MÔI TRƯỜNG PYSPARK ---
conda_site_packages = "/opt/conda/lib/python3.13/site-packages"
if conda_site_packages not in sys.path:
    sys.path.insert(0, conda_site_packages)
sys.path = [p for p in sys.path if "/usr/local/spark" not in p]

os.environ["SPARK_HOME"] = os.path.join(conda_site_packages, "pyspark")
os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

from pyspark.sql import SparkSession
from src.config.setting import Settings

if len(sys.argv) > 1:
    RUN_DATE = sys.argv[1] 
else:
    RUN_DATE = datetime.now().strftime("%Y/%m/%d")

print(f"🗓️ ĐANG CHẠY BACKTEST CHO NGÀY: {RUN_DATE}")

# Khởi tạo Spark để lấy dữ liệu từ Iceberg
spark = SparkSession.builder \
    .appName(f"Backtest_{RUN_DATE.replace('/', '_')}") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
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

# ĐỊNH NGHĨA CUSTOM DATA FEED (Chứa Tín hiệu)
class SignalData(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('datetime', None),
        ('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close'),
        ('volume', 'volume'), ('openinterest', -1),
        ('signal', 'signal'), 
    )

# ĐỊNH NGHĨA CHIẾN LƯỢC GIAO DỊCH
class SentimentSMAStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datasignal = self.datas[0].signal
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.datasignal[0] == 1:
                self.log(f'>>> TÍN HIỆU MUA: Đặt lệnh tại {self.dataclose[0]:.2f}')
                self.order = self.buy()
        else:
            if self.datasignal[0] == -1:
                self.log(f'<<< TÍN HIỆU BÁN: Đặt lệnh tại {self.dataclose[0]:.2f}')
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'✅ KHỚP LỆNH MUA | Giá: {order.executed.price:.2f} | Phí: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'✅ KHỚP LỆNH BÁN | Giá: {order.executed.price:.2f} | Phí: {order.executed.comm:.2f}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('❌ Lệnh bị hủy / Từ chối')
        self.order = None

def run_backtest():
    signal_table = "my_catalog.processed_zone.trading_signals"
    
    try:
        print("📥 Đang lấy dữ liệu Tín hiệu từ Data Lake...")
        df_spark = spark.read.table(signal_table).orderBy("trade_date")
        bt_df = df_spark.toPandas()
        
        # Chuẩn hóa Pandas DataFrame cho Backtrader
        bt_df['trade_date'] = pd.to_datetime(bt_df['trade_date'])
        bt_df.set_index('trade_date', inplace=True)
        
        for col in ['open', 'high', 'low', 'close']:
            bt_df[col] = bt_df['close_price']
        bt_df['volume'] = 0 
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SentimentSMAStrategy)
        
        data = SignalData(dataname=bt_df)
        cerebro.adddata(data)
        
        INITIAL_CASH = 10000.0
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=0.001) 
        cerebro.addsizer(bt.sizers.FixedSize, stake=50) 
        
        print('='*50)
        print(f'🚀 BẮT ĐẦU BACKTEST TRÊN DỮ LIỆU THẬT')
        print(f'💰 Vốn khởi điểm: ${cerebro.broker.getvalue():.2f}')
        print('='*50)
        
        cerebro.run()
        
        final_value = cerebro.broker.getvalue()
        pnl = final_value - INITIAL_CASH
        pnl_percent = (pnl / INITIAL_CASH) * 100
        
        print('='*50)
        print(f'🏁 KẾT THÚC BACKTEST')
        print(f'💰 Vốn cuối cùng: ${final_value:.2f}')
        if pnl >= 0:
            print(f'📈 LỢI NHUẬN RÒNG (PnL): +${pnl:.2f} (+{pnl_percent:.2f}%)')
        else:
            print(f'📉 THUA LỖ RÒNG (PnL): -${abs(pnl):.2f} ({pnl_percent:.2f}%)')
        print('='*50)

        # ----------------------------------------------------
        # VẼ BIỂU ĐỒ VÀ LƯU THÀNH FILE ẢNH (Bỏ qua plt.show())
        # ----------------------------------------------------
        print("📊 Đang vẽ biểu đồ kết quả Backtest...")
        plt.rcParams['figure.figsize'] = [16, 8]
        plt.rcParams['figure.dpi'] = 100
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # Nối thêm tên thư mục lưu báo cáo vào thư mục gốc
        save_dir = os.path.join(project_root, "bao_cao_daily")
        
        # Tạo thư mục nếu nó chưa tồn tại
        os.makedirs(save_dir, exist_ok=True)
        
        report_path = f"{save_dir}/backtest_result_{RUN_DATE.replace('/', '_')}.png"
        
        # Vẽ biểu đồ ẩn và lấy figures
        figs = cerebro.plot(iplot=False, style='candlestick', barup='green', bardown='red')
        figs[0][0].savefig(report_path)
        
        print(f"✅ Đã lưu biểu đồ Backtest thành công tại: {report_path}")
    except Exception as e:
        print(f"❌ LỖI QUÁ TRÌNH BACKTEST: {e}")

if __name__ == "__main__":
    run_backtest()
    spark.stop()