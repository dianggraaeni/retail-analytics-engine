# config/config.py
import os

# =======================
# Kafka Configuration
# =======================
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092' # Bisa juga os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = 'retail_data_stream'

# =======================
# Batch Configuration
# =======================
BATCH_SIZE = 1000              # Jumlah records per batch
BATCH_TIME_WINDOW = 60         # Dalam detik (seconds)

# =======================
# Data Paths (untuk di dalam Container Spark)
# =======================
# Path root tempat aplikasi di-mount di dalam container
APP_ROOT_IN_CONTAINER = '/tmp/app' # Sesuaikan jika berbeda di docker-compose.yml

DATA_DIR_IN_CONTAINER = os.path.join(APP_ROOT_IN_CONTAINER, 'data')

RAW_DATA_PATH = os.path.join(DATA_DIR_IN_CONTAINER, 'raw', 'online_retail_II.csv') # Untuk Producer
BATCH_DATA_PATH = os.path.join(DATA_DIR_IN_CONTAINER, 'batches') # Untuk Consumer dan Spark (jika baca batch)
MODEL_PATH = os.path.join(DATA_DIR_IN_CONTAINER, 'models') # Untuk Spark menyimpan model & API memuat model
TRAINING_DATA_PATH = os.path.join(DATA_DIR_IN_CONTAINER, 'training_data') # Untuk Spark melatih dari data CSV manual

# =======================
# Model Configuration
# =======================
MODEL_CONFIGS = {
    'linear_regression': {
        'batch_scheme': 'incremental',
        'target': 'total_amount'
    },
    'random_forest': {
        'batch_scheme': 'time_window',
        'target': 'quantity_category'
    },
    'logistic_regression': {
        'batch_scheme': 'fixed_size',
        'target': 'high_value_customer'
    }
}

# Untuk API yang berjalan di host, mungkin perlu path host juga
# HOST_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# MODEL_PATH_HOST = os.path.join(HOST_PROJECT_ROOT, 'data', 'models')