|           Nama               |     NRP    |
|            --                |     --     |
| Riakiyatul Nur Oktarani      | 5027231013 |
| Dian Anggraeni Putri         | 5027231016 |
| Acintya Edria Sudarsono      | 5027231020 |

# Retail Analytics Engine: Sistem Big Data dengan Kafka, Spark, dan REST API

Sistem ini mensimulasikan arsitektur pemrosesan data stream secara real-time menggunakan dataset *Online Retail II*. Data dari dataset dibaca seolah-olah merupakan stream berkelanjutan, diproses dalam batch, digunakan untuk melatih model Machine Learning secara periodik, dan hasilnya disajikan melalui REST API untuk prediksi.

![image](https://github.com/user-attachments/assets/5f6a4f6a-2d1d-4835-a34e-496d85573621)

## 1. Arsitektur & Alur Kerja

Sistem ini terdiri dari beberapa komponen utama yang bekerja secara berurutan untuk mengubah data mentah menjadi insight yang dapat diakses.

```mermaid
graph TD
    subgraph "Data Source"
        A[1. Dataset CSV]
    end

    subgraph "Data Ingestion (Kafka)"
        B((Kafka Producer))
        C{Kafka Topic: retail_data_stream}
    end

    subgraph "Batch Processing (Python)"
        D((Kafka Consumer))
        E[Batch Files (.csv)]
    end

    subgraph "Model Training (Apache Spark)"
        F[Spark ML Training Jobs]
        G[(Trained Models)]
    end

    subgraph "Serving Layer (API)"
        H{REST API (Flask)}
        I[Client / Postman]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    I --> H
    H --> I

```

---

## 2. Struktur Direktori Proyek

```
retail-analytics-engine/
│
├── README.md                      # Dokumentasi utama proyek
├── run_system.sh                  # Skrip utama untuk menjalankan dan menghentikan sistem
├── docker-compose.yml             # Konfigurasi Docker untuk Kafka & Spark Cluster
├── test_api.py                    # Skrip untuk pengujian otomatis semua endpoint API
│
├── config/                        # Konfigurasi terpusat
│   └── config.py                  # Variabel lingkungan dan parameter sistem
│
├── data/                          # Direktori data (tidak termasuk dalam Git)
│   ├── raw/                       # Data mentah
│   │   └── online_retail_II.csv   # Diunduh secara manual
│   ├── batches/                   # File batch yang dihasilkan oleh Consumer
│   └── models/                    # Model ML yang telah dilatih oleh Spark
│
├── kafka/                         # Komponen Kafka Producer & Consumer
│   ├── producer.py
│   └── consumer.py
│
├── spark/                         # Skrip training Spark ML
│   ├── linear_regression_model.py
│   ├── random_forest_model.py
│   └── logistic_regression_model.py
│
├── api/                           # Komponen REST API (Flask)
│   ├── app.py
│   └── requirements.txt
│
└── logs/                          # Log yang dihasilkan sistem (tidak termasuk dalam Git)
    ├── spark_jobs_from_consumer/  # Log spesifik dari setiap job Spark
    └── ...
```

---

## 3. Setup dan Penggunaan

### Prasyarat
- **Docker & Docker Compose**: Untuk menjalankan Kafka dan Spark.
- **Python 3.8+ & Pip**: Untuk menjalankan semua skrip non-Docker.
- **Apache Spark 3.4+ (Instalasi Lokal)**: Diperlukan agar perintah `spark-submit` bisa dijalankan dari host machine. Pastikan direktori `bin` Spark sudah ditambahkan ke `PATH` environment variable sistem Anda.

### Panduan Instalasi & Menjalankan

1.  **Clone Repository**
    ```bash
    git clone <repository-url>
    cd retail-analytics-engine
    ```

2.  **Siapkan Dataset**
    -   Unduh dataset [Online Retail II dari Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).
    -   Ekstrak dan letakkan file `online_retail_II.csv` di dalam direktori `data/raw/`.

3.  **Setup Virtual Environment & Dependensi**
    ```bash
    # Buat dan aktifkan virtual environment (direkomendasikan)
    python -m venv venv
    source venv/Scripts/activate  # Untuk Windows (Git Bash)
    # source venv/bin/activate    # Untuk macOS/Linux

    # Install semua dependensi
    pip install -r api/requirements.txt
    ```

4.  **Jalankan Sistem**
    Skrip `run_system.sh` akan mengelola semua komponen secara otomatis.

    ```bash
    # Berikan izin eksekusi pada skrip
    chmod +x run_system.sh

    # Jalankan semua komponen (Kafka, Spark, Producer, Consumer, API)
    ./run_system.sh
    ```
    - **Kafka UI (Kafdrop)**: `http://localhost:9090`
    - **API Server**: `http://localhost:5000`

5.  **Hentikan Sistem**
    Tekan `Ctrl+C` di terminal tempat `run_system.sh` berjalan. Skrip akan menangani proses shutdown secara bersih.

---

## 4. Strategi Training Model

Model dilatih secara periodik setiap kali batch data baru disimpan, dengan strategi berikut:

| Jenis Model | Versi Model | Data yang Digunakan | Direktori Model |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | v1 | 1/3 data pertama | `linear_model_1` |
| (Regresi) | v2 | 2/3 data pertama | `linear_model_2` |
| | v3 | Semua data | `linear_model_3` |
| **Random Forest** | v1 | Data dari 5 menit pertama | `rf_model_window_1` |
| (Klasifikasi) | v2 | Data dari 10 menit pertama | `rf_model_window_2` |
| | v3 | Data dari 15 menit pertama | `rf_model_window_3` |
| **Logistic Regression**| v1 | Batch data ke-N | `logistic_model_batch_1`|
| (Klasifikasi) | v2 | Batch data ke-N+1 | `logistic_model_batch_2`|
| | v3 | Batch data ke-N+2 | `logistic_model_batch_3`|

---

## 5. Pemetaan Endpoint API

Gunakan Postman atau `curl` untuk mengirim request `POST` ke endpoint berikut dengan body JSON.

| Tujuan Prediksi | Endpoint | Model | Fitur Input (JSON Body) |
| :--- | :--- | :--- | :--- |
| **Prediksi Harga Total**| `/predict/total_amount/v1` | Linear Reg V1 | `quantity`, `price`, `month`, `day`, `hour` |
| | `/predict/total_amount/v2` | Linear Reg V2 | `quantity`, `price`, `month`, `day`, `hour` |
| | `/predict/total_amount/v3` | Linear Reg V3 | `quantity`, `price`, `month`, `day`, `hour` |
| **Klasifikasi Kuantitas**| `/predict/quantity_category/v1`| Random Forest V1| `quantity`, `price`, `total_amount`, `month`, `day`, `hour` |
| | `/predict/quantity_category/v2`| Random Forest V2| `quantity`, `price`, `total_amount`, `month`, `day`, `hour` |
| | `/predict/quantity_category/v3`| Random Forest V3| `quantity`, `price`, `total_amount`, `month`, `day`, `hour` |
| **Prediksi Pelanggan High-Value** | `/predict/high_value_customer/v1` | Logistic Reg V1 | `quantity`, `price`, `total_amount`, `month`, `day`, `hour` |
| | `/predict/high_value_customer/v2` | Logistic Reg V2 | `quantity`, `price`, `total_amount`, `month`, `day`, `hour` |
| | `/predict/high_value_customer/v3` | Logistic Reg V3 | `quantity`, `price`, `total_amount`, `month`, `day`, `hour` |

#### Contoh Request `curl`:
```bash
curl -X POST http://localhost:5000/predict/total_amount/v3 \
-H "Content-Type: application/json" \
-d '{
    "quantity": 10,
    "price": 25.50,
    "month": 6,
    "day": 11,
    "hour": 14
}'
```

---

## 6. Testing

Untuk menjalankan pengujian otomatis pada semua endpoint API, gunakan skrip `test_api.py`.

```bash
python test_api.py
```

![Hasil Testing API](https://github.com/user-attachments/assets/6af8166f-d645-4bf9-ba26-864748c5e684)

