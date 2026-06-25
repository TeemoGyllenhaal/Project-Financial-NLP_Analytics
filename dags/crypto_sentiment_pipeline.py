from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# ==========================================
# 1. CẤU HÌNH MẶC ĐỊNH CHO DAG
# ==========================================
default_args = {
    'owner': 'nhom_5_do_an_7',          # Tên người sở hữu (hoặc tên nhóm)
    'depends_on_past': False,           # Không phụ thuộc vào ngày chạy hôm trước
    'start_date': datetime(2026, 6, 20),# Ngày bắt đầu cho phép hệ thống chạy
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,                       # Số lần thử lại nếu task bị lỗi
    'retry_delay': timedelta(minutes=1),# Thời gian chờ giữa các lần thử lại
}

# ==========================================
# 2. KHỞI TẠO DAG
# ==========================================
# schedule_interval: '0 17 * * *' nghĩa là chạy vào 17:00 (5h chiều) mỗi ngày
dag = DAG(
    dag_id='crypto_sentiment_trading_pipeline_2',
    default_args=default_args,
    description='Pipeline tự động Cào dữ liệu -> NLP -> Chấm điểm -> Trading',
    schedule_interval='0 17 * * *', 
    catchup=False,                      # Không chạy bù các ngày trong quá khứ khi mới bật lên
    tags=['NLP', 'Trading', 'BigData'],
)

# ==========================================
# 3. ĐỊNH NGHĨA CÁC TASK (TÁC VỤ)
# ==========================================
# LƯU Ý: {{ ds }} là một biến đặc biệt (Macro) của Airflow. 
# Nó sẽ tự động biến thành chuỗi ngày tháng (VD: '2026-06-24') khi chạy, 
# khớp hoàn hảo với sys.argv[1] mà chúng ta đã code trong các file Python.

# Task 1: Thu thập tin tức và giá (Crawl)
crawl_task = BashOperator(
    task_id='crawl_daily_data',
    bash_command='python /opt/airflow/src/crawlers/daily_finnhub_crawler.py {{ ds }}',
    dag=dag,
)

# Task 2: Tiền xử lý NLP bằng PySpark & spaCy
nlp_process_task = BashOperator(
    task_id='spark_nlp_processing',
    bash_command='python /opt/airflow/src/processing/daily_spark_processor.py {{ ds }}',
    dag=dag,
)

# Task 3: Chấm điểm cảm xúc (TextBlob & VADER)
sentiment_task = BashOperator(
    task_id='sentiment_scoring',
    bash_command='python /opt/airflow/src/models/sentiment_scoring_pipeline.py {{ ds }}',
    dag=dag,
)

# Task 4: Tóm tắt chủ đề bằng LDA
lda_task = BashOperator(
    task_id='lda_topic_modeling',
    bash_command='python /opt/airflow/src/models/lda_topic_modeler.py {{ ds }}',
    dag=dag,
)

# Task 4.5: Vẽ Wordcloud và pyLDAvis
viz_task = BashOperator(
    task_id='visualize',
    bash_command='python /opt/airflow/src/visualizer/wordCloud_pyLDAvis.py {{ ds }}',
    dag=dag,
)


# Task 5: Tạo tín hiệu Mua/Bán (Signal)
signal_task = BashOperator(
    task_id='generate_trading_signals',
    bash_command='python /opt/airflow/src/trading/signal_generator.py {{ ds }}',
    dag=dag,
)

# Task 6: Chạy mô phỏng Backtesting
backtest_task = BashOperator(
    task_id='run_backtrader_simulation',
    bash_command='python /opt/airflow/src/trading/backtrader_strategy.py {{ ds }}',
    dag=dag,
)

# ==========================================
# 4. THIẾT LẬP THỨ TỰ CHẠY (NỐI DÒNG)
# ==========================================

crawl_task >> nlp_process_task

# Sau khi NLP xong, rẽ nhánh:
# Nhánh 1: Sentiment -> Signal -> Backtest
# Nhánh 2: LDA -> Visualize
nlp_process_task >> sentiment_task >> signal_task >> backtest_task
nlp_process_task >> lda_task >> viz_task