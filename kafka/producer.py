# kafka/producer.py

import pandas as pd
import json
import time
import random
from kafka.errors import KafkaTimeoutError 
from kafka import KafkaProducer
from datetime import datetime
import sys
import os

# --- Penentuan Path yang Benar untuk Host & Impor Config ---
try:
    PROJECT_ROOT_ON_HOST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT_ON_HOST not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_ON_HOST)

    HOST_RAW_DATA_PATH = os.path.join(PROJECT_ROOT_ON_HOST, 'data', 'raw', 'online_retail_II.csv')
    
    from config.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
    # BATCH_SIZE dari config mungkin tidak relevan untuk producer, tapi kita bisa pakai untuk info
    # from config.config import BATCH_SIZE as PRODUCER_BATCH_INFO_SIZE 
    
    print(f"PRODUCER: Using KAFKA_BOOTSTRAP_SERVERS: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"PRODUCER: Using KAFKA_TOPIC: {KAFKA_TOPIC}")
    print(f"PRODUCER: Using RAW_DATA_PATH on host: {HOST_RAW_DATA_PATH}")

except ImportError:
    print("PRODUCER CRITICAL ERROR: Could not import Kafka configurations from config.config.")
    print(f"Ensure config/config.py exists in project root '{PROJECT_ROOT_ON_HOST}'.")
    sys.exit(1)
except Exception as e:
    print(f"PRODUCER CRITICAL ERROR during initial setup: {e}")
    sys.exit(1)
# --- Akhir Penentuan Path & Impor Config ---

class RetailDataProducer:
    def __init__(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                acks='all', 
                retries=3,  
                linger_ms=20, # Sedikit lebih lama untuk batching internal Kafka
                batch_size=16384 * 2 # Ukuran batch internal Kafka (bytes)
            )
            print("PRODUCER: Kafka Producer initialized successfully.")
        except Exception as e:
            print(f"PRODUCER CRITICAL ERROR: Failed to initialize Kafka Producer: {e}")
            print(f"Check if Kafka server is running at {KAFKA_BOOTSTRAP_SERVERS}.")
            sys.exit(1)
        
    def preprocess_chunk(self, df_chunk):
        if df_chunk.empty:
            return df_chunk

        # print(f"PRODUCER: Preprocessing chunk with {len(df_chunk)} rows...")
        df_processed = df_chunk.copy()

        # Standarisasi nama kolom dari CSV ke huruf kecil jika belum
        df_processed.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_processed.columns]

        # Hapus transaksi yang dibatalkan (asumsi kolom 'invoice' ada)
        if 'invoice' in df_processed.columns:
            df_processed['invoice'] = df_processed['invoice'].astype(str)
            df_processed = df_processed[~df_processed['invoice'].str.startswith('c', na=False)] # Perhatikan 'c' kecil
        
        # Hapus baris dengan Customer ID kosong (asumsi kolom 'customer_id' ada)
        if 'customer_id' in df_processed.columns:
            df_processed = df_processed.dropna(subset=['customer_id'])
            if df_processed.empty: return df_processed
            df_processed['customer_id'] = df_processed['customer_id'].astype(str)
        
        # Hapus baris dengan Quantity atau Price tidak valid
        if 'quantity' in df_processed.columns and 'price' in df_processed.columns:
            df_processed = df_processed.dropna(subset=['quantity', 'price'])
            if df_processed.empty: return df_processed
            # Pastikan tipe numerik sebelum perbandingan
            df_processed['quantity'] = pd.to_numeric(df_processed['quantity'], errors='coerce')
            df_processed['price'] = pd.to_numeric(df_processed['price'], errors='coerce')
            df_processed = df_processed.dropna(subset=['quantity', 'price']) # Drop lagi jika coerce menghasilkan NaN
            if df_processed.empty: return df_processed
            df_processed = df_processed[df_processed['quantity'] > 0]
            if df_processed.empty: return df_processed
            df_processed = df_processed[df_processed['price'] > 0]
            if df_processed.empty: return df_processed
        else:
            print("PRODUCER WARNING: 'quantity' or 'price' column missing, cannot calculate total_amount or filter by them.")
            return pd.DataFrame() # Kembalikan DataFrame kosong jika kolom penting hilang

        # Hitung total_amount
        df_processed['total_amount'] = df_processed['quantity'] * df_processed['price']
        
        # Fitur waktu
        if 'invoicedate' in df_processed.columns: # Asumsi nama kolom dari CSV sudah jadi 'invoicedate'
            try:
                df_processed['invoice_date_dt'] = pd.to_datetime(df_processed['invoicedate'])
                df_processed['month'] = df_processed['invoice_date_dt'].dt.month
                df_processed['day'] = df_processed['invoice_date_dt'].dt.day
                df_processed['hour'] = df_processed['invoice_date_dt'].dt.hour
            except Exception as e:
                print(f"PRODUCER WARNING: Error processing 'invoicedate' in chunk: {e}.")
                df_processed['month'] = 0; df_processed['day'] = 0; df_processed['hour'] = 0
                df_processed['invoice_date_dt'] = pd.NaT
        else:
            df_processed['month'] = 0; df_processed['day'] = 0; df_processed['hour'] = 0; df_processed['invoice_date_dt'] = pd.NaT

        # Quantity categories
        if 'quantity' in df_processed.columns:
            df_processed_qc = df_processed.dropna(subset=['quantity'])
            if not df_processed_qc.empty:
                labels = ['low', 'medium', 'high', 'bulk']
                df_processed['quantity_category'] = pd.cut(df_processed_qc['quantity'], 
                                                       bins=[-float('inf'), 1, 5, 20, float('inf')], 
                                                       labels=labels, right=False, include_lowest=True)
                # Pastikan tipe kategorikal dan tambahkan 'unknown' jika belum ada
                df_processed['quantity_category'] = pd.Categorical(df_processed['quantity_category'], categories=labels + ['unknown'], ordered=False)
                df_processed['quantity_category'] = df_processed['quantity_category'].fillna('unknown')
            else:
                df_processed['quantity_category'] = 'unknown'
        else:
            df_processed['quantity_category'] = 'unknown'

        # High value customer - Disederhanakan/dihilangkan di producer karena kompleksitas chunking
        # Lebih baik dihitung di Spark jika memungkinkan, atau gunakan threshold pre-calculated
        df_processed['high_value_customer'] = 0 # Default
        
        return df_processed
    
    def stream_data(self):
        print("PRODUCER: Loading dataset in chunks...")
        chunk_size = 50000  # Baca 50,000 baris per chunk, sesuaikan jika perlu
        total_records_streamed = 0
        global_record_counter = 0 # Untuk key jika customer_id tidak ada

        try:
            if not os.path.exists(HOST_RAW_DATA_PATH):
                print(f"PRODUCER CRITICAL ERROR: Dataset file not found at {HOST_RAW_DATA_PATH}")
                return

            for df_chunk_raw in pd.read_csv(HOST_RAW_DATA_PATH, encoding='ISO-8859-1', 
                                         chunksize=chunk_size,
                                         # Baca semua sebagai string dulu untuk fleksibilitas, casting di preprocess
                                         dtype=str 
                                         ):
                
                if df_chunk_raw.empty:
                    print("PRODUCER: Read an empty chunk, skipping.")
                    continue
                
                print(f"\nPRODUCER: Processing new chunk of raw size {len(df_chunk_raw)}...")
                df_chunk_processed = self.preprocess_chunk(df_chunk_raw)

                if df_chunk_processed.empty:
                    print("PRODUCER: Chunk is empty after preprocessing, skipping.")
                    continue
                
                print(f"PRODUCER: Streaming {len(df_chunk_processed)} processed records from current chunk...")
                for _, row in df_chunk_processed.iterrows():
                    invoice_date_iso = None
                    if 'invoice_date_dt' in row and pd.notnull(row['invoice_date_dt']):
                        try:
                            invoice_date_iso = row['invoice_date_dt'].isoformat()
                        except AttributeError: # Jika invoice_date_dt bukan datetime object
                            invoice_date_iso = str(row['invoice_date_dt']) # Fallback ke string

                    record = {
                        'invoice': row.get('invoice', ''),
                        'stock_code': row.get('stockcode', ''), # Sesuaikan dengan hasil tolower()
                        'description': row.get('description', ''),
                        'quantity': float(row.get('quantity', 0.0)),
                        'invoice_date': invoice_date_iso,
                        'price': float(row.get('price', 0.0)),
                        'customer_id': str(row.get('customer_id', '')),
                        'country': row.get('country', ''),
                        'total_amount': float(row.get('total_amount', 0.0)),
                        'month': int(row.get('month', 0)),
                        'day': int(row.get('day', 0)),
                        'hour': int(row.get('hour', 0)),
                        'quantity_category': str(row.get('quantity_category', 'unknown')),
                        'high_value_customer': int(row.get('high_value_customer', 0)),
                        'timestamp': datetime.now().isoformat(),
                    }
                    
                    kafka_key = record['customer_id'] if record['customer_id'] else str(global_record_counter)
                    try:
                        self.producer.send(KAFKA_TOPIC, key=kafka_key, value=record)
                    except KafkaTimeoutError:
                        print(f"PRODUCER KafkaTimeoutError sending record (global_record_counter {global_record_counter}).")
                        continue 
                    except Exception as e:
                        print(f"PRODUCER Error sending record (global_record_counter {global_record_counter}) to Kafka: {e}")
                        continue
                    
                    total_records_streamed += 1
                    global_record_counter += 1

                    if total_records_streamed % 10000 == 0: # Log lebih jarang
                        print(f"PRODUCER: Sent {total_records_streamed} total records so far...")
                        self.producer.flush(timeout=10)

                # Delay antar chunk, bukan antar record
                # time.sleep(0.1) # Optional: delay antar chunk

        except FileNotFoundError:
            print(f"PRODUCER CRITICAL ERROR: Dataset file not found at {HOST_RAW_DATA_PATH}")
            return
        except pd.errors.EmptyDataError:
            print("PRODUCER CRITICAL ERROR: Dataset file is empty or all chunks were empty.")
            return
        except Exception as e:
            print(f"PRODUCER CRITICAL ERROR during data streaming: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"\nPRODUCER: Finished streaming all data. Total records streamed: {total_records_streamed}")
        try:
            self.producer.flush(timeout=60) 
            print("PRODUCER: All messages flushed to Kafka.")
        except KafkaTimeoutError:
            print("PRODUCER: KafkaTimeoutError during final flush.")
        except Exception as e:
            print(f"PRODUCER: Error during final flush: {e}")
        finally:
            self.producer.close()
            print("PRODUCER: Kafka Producer closed.")

if __name__ == "__main__":
    producer = RetailDataProducer()
    producer.stream_data()