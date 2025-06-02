# kafka/consumer.py

import json
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import subprocess
import sys

# --- Penentuan Path yang Benar untuk Host & Impor Config ---
try:
    PROJECT_ROOT_ON_HOST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT_ON_HOST not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_ON_HOST)

    HOST_BATCH_DATA_DIR = os.path.join(PROJECT_ROOT_ON_HOST, 'data', 'batches')
    
    from config.config import (
        KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, 
        BATCH_SIZE, BATCH_TIME_WINDOW,
        # Path ini adalah untuk referensi APA YANG AKAN DIGUNAKAN SPARK DI CONTAINER
        # Consumer sendiri tidak menggunakannya untuk menulis.
        BATCH_DATA_PATH as SPARK_EXPECTED_BATCH_PATH, 
        MODEL_PATH as SPARK_EXPECTED_MODEL_PATH
    )

    print(f"CONSUMER: Using KAFKA_BOOTSTRAP_SERVERS: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"CONSUMER: Using KAFKA_TOPIC: {KAFKA_TOPIC}")
    print(f"CONSUMER: Will save batches to HOST path: {HOST_BATCH_DATA_DIR}")
    print(f"CONSUMER: Spark will read batches from CONTAINER path (from config): {SPARK_EXPECTED_BATCH_PATH}")

except ImportError:
    print("CONSUMER CRITICAL ERROR: Could not import configurations from config.config.")
    sys.exit(1)
except Exception as e:
    print(f"CONSUMER CRITICAL ERROR during initial setup: {e}")
    sys.exit(1)
# --- Akhir Penentuan Path & Impor Config ---

class RetailDataConsumer:
    def __init__(self):
        # Ganti group_id jika Anda ingin consumer membaca ulang semua pesan dari awal topik
        self.consumer_group_id = 'retail_analytics_consumer_group_v3' 
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',
                enable_auto_commit=True,      
                group_id=self.consumer_group_id,
                consumer_timeout_ms=20000 # Timeout 20 detik jika tidak ada pesan
            )
            print(f"CONSUMER: Kafka Consumer initialized for group '{self.consumer_group_id}'. Waiting for messages...")
        except Exception as e:
            print(f"CONSUMER CRITICAL ERROR: Failed to initialize Kafka Consumer: {e}")
            sys.exit(1)
        
        self.batch_buffer = []
        self.batch_counter = 0 
        self.last_batch_process_time = datetime.now()
        
        os.makedirs(HOST_BATCH_DATA_DIR, exist_ok=True)
        self.spark_log_dir = os.path.join(PROJECT_ROOT_ON_HOST, 'logs', 'spark_jobs_from_consumer')
        os.makedirs(self.spark_log_dir, exist_ok=True)
    
    def save_current_batch(self):
        if not self.batch_buffer:
            return False

        current_batch_to_save = list(self.batch_buffer)
        self.batch_buffer.clear()
        
        df = pd.DataFrame(current_batch_to_save)
        # Nama kolom di sini seharusnya sudah huruf kecil dari producer
        
        filename = f"batch_{self.batch_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath_on_host = os.path.join(HOST_BATCH_DATA_DIR, filename)
        
        try:
            df.to_csv(filepath_on_host, index=False)
            print(f"CONSUMER: Saved batch {self.batch_counter} with {len(current_batch_to_save)} records to HOST: {filepath_on_host}")
            self.batch_counter += 1
            return True
        except Exception as e:
            print(f"CONSUMER ERROR saving batch file {filepath_on_host}: {e}")
            self.batch_buffer.extend(current_batch_to_save)
            return False

    def should_create_batch(self):
        size_condition = len(self.batch_buffer) >= BATCH_SIZE
        time_condition = (datetime.now() - self.last_batch_process_time).total_seconds() >= BATCH_TIME_WINDOW
        return len(self.batch_buffer) > 0 and (size_condition or time_condition)
    
    def trigger_model_training(self, completed_batch_id):
        print(f"\n--- CONSUMER: Triggering Spark Model Training (after batch {completed_batch_id} was saved) ---")
        
        spark_script_dir_on_host = os.path.join(PROJECT_ROOT_ON_HOST, 'spark')
        scripts_info = [
            ("Linear Regression", os.path.join(spark_script_dir_on_host, 'linear_regression_model.py')),
            ("Logistic Regression", os.path.join(spark_script_dir_on_host, 'logistic_regression_model.py')),
            ("Random Forest", os.path.join(spark_script_dir_on_host, 'random_forest_model.py')),
        ]
        
        # !!! PENTING: SESUAIKAN KONFIGURASI SPARK-SUBMIT INI !!!
        # Gunakan nilai yang telah Anda temukan paling stabil.
        spark_submit_base_cmd = [
            "spark-submit",
            "--master", os.getenv("SPARK_MASTER_URL_FOR_SUBMIT", "spark://localhost:7077"), # Dari host ke Docker
            "--deploy-mode", "client",
            "--driver-memory", os.getenv("SPARK_DRIVER_MEMORY_SUBMIT", "2g"), 
            # Jika Anda menggunakan worker terpisah dan mode cluster sebenarnya:
            # "--executor-memory", os.getenv("SPARK_EXECUTOR_MEMORY_SUBMIT", "2g"),
            # "--num-executors", os.getenv("SPARK_NUM_EXECUTORS_SUBMIT", "1"),
            # "--executor-cores", os.getenv("SPARK_EXECUTOR_CORES_SUBMIT", "1"),
            "--conf", f"spark.sql.shuffle.partitions={os.getenv('SPARK_SHUFFLE_PARTITIONS_SUBMIT', '30')}"
        ]

        for model_type, script_path in scripts_info:
            if not os.path.exists(script_path):
                print(f"  CONSUMER ERROR: Spark script for {model_type} not found at: {script_path}. Skipping.")
                continue

            cmd_to_run = spark_submit_base_cmd + [script_path]
            # Skrip Spark akan mengambil BATCH_DATA_PATH dan MODEL_PATH dari config.py nya sendiri (path container)
            
            script_name = os.path.basename(script_path)
            print(f"  CONSUMER: Submitting Spark job for: {script_name} ({model_type})")
            print(f"    Command: {' '.join(cmd_to_run)}")
            
            try:
                log_file_name = f"{script_name.replace('.py', '')}_batch_{completed_batch_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
                log_file_path = os.path.join(self.spark_log_dir, log_file_name)

                with open(log_file_path, 'w') as log_f:
                    process = subprocess.Popen(cmd_to_run, stdout=log_f, stderr=subprocess.STDOUT)
                print(f"  CONSUMER: Spark job for {script_name} submitted. PID: {process.pid}. Log: {log_file_path}")
            except FileNotFoundError:
                print(f"  CONSUMER ERROR: 'spark-submit' command not found. Ensure Spark is installed on host and in PATH.")
            except Exception as e:
                print(f"  CONSUMER ERROR submitting Spark job {script_name}: {e}")
        print(f"--- CONSUMER: Spark Model Training Triggers for batch {completed_batch_id} Sent ---")

    def consume_data(self):
        print(f"CONSUMER: Starting data consumption loop for topic '{KAFKA_TOPIC}' (Group ID: '{self.consumer_group_id}'). Press Ctrl+C to stop.")
        total_messages_in_session = 0
        
        try:
            while True: 
                messages_in_current_poll = 0
                # Loop for akan timeout berdasarkan consumer_timeout_ms jika tidak ada pesan
                for message in self.consumer: 
                    messages_in_current_poll +=1
                    if message and message.value:
                        record = message.value
                        self.batch_buffer.append(record)
                        total_messages_in_session += 1

                        if len(self.batch_buffer) > 0 and len(self.batch_buffer) % (BATCH_SIZE // 10) == 0 : # Log lebih jarang
                             print(f"CONSUMER: Buffer size: {len(self.batch_buffer)}/{BATCH_SIZE}. Total processed: {total_messages_in_session}.")
                    
                    if self.should_create_batch():
                        batch_id_before_save = self.batch_counter
                        if self.save_current_batch():
                            self.last_batch_process_time = datetime.now()
                            self.trigger_model_training(batch_id_before_save)
                
                # Setelah loop for (mungkin karena timeout jika tidak ada pesan baru)
                if messages_in_current_poll == 0:
                    print(f"CONSUMER: No messages received in the last {self.consumer.config['consumer_timeout_ms']/1000}s. Checking time-based batch.")

                if self.should_create_batch(): # Cek lagi untuk time-based batching, jika ada sisa di buffer
                    print("CONSUMER: Condition met for time-based batching (or size if messages came fast before timeout).")
                    batch_id_before_save = self.batch_counter
                    if self.save_current_batch():
                        self.last_batch_process_time = datetime.now()
                        self.trigger_model_training(batch_id_before_save)
                
                if not self.batch_buffer and messages_in_current_poll == 0:
                    print(f"CONSUMER: Buffer empty and no new messages. Topic '{KAFKA_TOPIC}' might be idle or producer finished.")
                    # Anda bisa tambahkan logika untuk break loop jika kondisi tertentu terpenuhi,
                    # misalnya, jika producer mengirim pesan "selesai". Untuk sekarang, akan terus polling.
                    # time.sleep(5) # Jeda tambahan jika benar-benar tidak ada aktivitas


        except KeyboardInterrupt:
            print("\nCONSUMER: Loop interrupted by user (Ctrl+C).")
        except KafkaError as ke:
            print(f"CONSUMER: Kafka related error in consumer loop: {ke}")
        except Exception as e:
            print(f"CONSUMER: An unexpected error occurred in consumer loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("CONSUMER: Shutting down...")
            if self.batch_buffer:
                print(f"CONSUMER: Saving final {len(self.batch_buffer)} records from buffer before exiting...")
                self.save_current_batch() 
            if hasattr(self, 'consumer') and self.consumer:
                self.consumer.close(timeout=10) 
                print("CONSUMER: Kafka Consumer closed.")
            print(f"CONSUMER: Total messages processed in this session: {total_messages_in_session}.")

if __name__ == "__main__":
    print("Initializing Retail Data Consumer Application...")
    consumer_app = RetailDataConsumer()
    consumer_app.consume_data()