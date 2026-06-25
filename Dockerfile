# Sử dụng Image gốc của Airflow
FROM apache/airflow:2.9.1-python3.10

# Chuyển sang quyền Root để cài đặt Java
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường cho Java 
# (Lưu ý: Mình đã sửa lại thành java-17 cho khớp với bản cài đặt ở trên thay vì java-11)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Chuyển lại về user airflow để thao tác an toàn
USER airflow

# Copy file requirements.txt từ máy tính vào thư mục tạm của container
COPY requirements.txt /tmp/requirements.txt

# Cài đặt toàn bộ thư viện trong file requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Tải model xử lý ngôn ngữ tiếng Anh cơ bản cho spaCy
RUN python -m spacy download en_core_web_sm