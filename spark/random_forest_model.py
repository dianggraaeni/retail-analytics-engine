# /tmp/app/spark/random_forest_model.py

import argparse
import os
import sys
import glob
from datetime import datetime, timedelta # Untuk logika time window jika diperlukan

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, min as spark_min, max as spark_max, lit, to_timestamp
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString # StandardScaler mungkin tidak terlalu krusial untuk RF
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# --- Konfigurasi Impor ---
try:
    PROJECT_ROOT_IN_CONTAINER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT_IN_CONTAINER not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_IN_CONTAINER)
    
    from config.config import MODEL_PATH as DEFAULT_MODEL_PATH_CFG
    from config.config import BATCH_DATA_PATH as DEFAULT_BATCH_DATA_PATH_CFG
    
    print(f"Successfully imported from config.py for Random Forest:")
    print(f"  DEFAULT_MODEL_PATH_CFG: {DEFAULT_MODEL_PATH_CFG}")
    print(f"  DEFAULT_BATCH_DATA_PATH_CFG: {DEFAULT_BATCH_DATA_PATH_CFG}")

except ImportError as e:
    print(f"CRITICAL ERROR (RF): Could not import paths from config.config.")
    print(f"Ensure config.py is in '{PROJECT_ROOT_IN_CONTAINER}/config/' and defines MODEL_PATH and BATCH_DATA_PATH.")
    print(f"ImportError: {e}")
    sys.exit(1)
except AttributeError as e:
    print(f"CRITICAL ERROR (RF): A required path variable (MODEL_PATH or BATCH_DATA_PATH) is missing from config.py.")
    print(f"AttributeError: {e}")
    sys.exit(1)

class RandomForestModelTrainer:
    def __init__(self, spark_session, model_base_dir, batch_data_base_dir):
        self.spark = spark_session
        self.model_base_dir = model_base_dir
        self.batch_data_base_dir = batch_data_base_dir
        os.makedirs(self.model_base_dir, exist_ok=True)
        print(f"RandomForestModelTrainer initialized.")
        print(f"  Models will be saved to: {self.model_base_dir}")
        print(f"  Batch data will be read from: {self.batch_data_base_dir}")

    def _load_minimal_batch_data(self): # <<< NAMA FUNGSI DIUBAH UNTUK KEJELASAN TES
        """Loads ONLY THE FIRST batch file and LIMITS it to a small number of rows for testing."""
        all_batch_files_glob = os.path.join(self.batch_data_base_dir, "batch_*.csv")
        sorted_batch_file_paths = sorted(glob.glob(all_batch_files_glob))

        if not sorted_batch_file_paths:
            print("MINIMAL TEST: No batch files found. Cannot proceed.")
            return None

        # HANYA AMBIL FILE BATCH PERTAMA
        first_batch_file_path = [sorted_batch_file_paths[0]] 
        print(f"MINIMAL TEST: Attempting to load only the first batch file: {first_batch_file_path[0]}")

        df_temp = None
        try:
            df_temp = self.spark.read.csv(first_batch_file_path[0], header=True, inferSchema=True)
            
            if df_temp.isEmpty():
                print(f"MINIMAL TEST: DataFrame from {first_batch_file_path[0]} is empty.")
                return None

            # Pastikan kolom timestamp ada dan bertipe Timestamp
            if "timestamp" not in df_temp.columns:
                print(f"MINIMAL TEST WARNING: Column 'timestamp' not found in {first_batch_file_path[0]}. Creating dummy timestamp_col.")
                df_temp = df_temp.withColumn("timestamp_col", lit(datetime.now()).cast("timestamp")) # Dummy jika tidak ada
            else:
                df_temp = df_temp.withColumn("timestamp_col", to_timestamp(col("timestamp")))
                if df_temp.filter(col("timestamp_col").isNull()).count() > 0 and \
                   df_temp.filter(col("timestamp").isNotNull()).count() > 0:
                    print(f"MINIMAL TEST WARNING: Some 'timestamp' values in {first_batch_file_path[0]} could not be converted.")
            
            # Casting kolom lain (sesuaikan dengan kebutuhan dan nama kolom CSV)
            # Nama kolom sudah diasumsikan huruf kecil dari CSV
            if "quantity" in df_temp.columns: df_temp = df_temp.withColumn("quantity", col("quantity").cast("double"))
            if "price" in df_temp.columns: df_temp = df_temp.withColumn("price", col("price").cast("double"))
            if "total_amount" in df_temp.columns: df_temp = df_temp.withColumn("total_amount", col("total_amount").cast("double"))
            if "month" in df_temp.columns: df_temp = df_temp.withColumn("month", col("month").cast("integer"))
            if "day" in df_temp.columns: df_temp = df_temp.withColumn("day", col("day").cast("integer"))
            if "hour" in df_temp.columns: df_temp = df_temp.withColumn("hour", col("hour").cast("integer"))
            if "quantity_category" not in df_temp.columns:
                 print(f"MINIMAL TEST WARNING: Target column 'quantity_category' not found in {first_batch_file_path[0]}.")
        
        except Exception as e:
            print(f"MINIMAL TEST ERROR: Could not read or process CSV file {first_batch_file_path[0]}: {e}")
            return None
        
        # PILIH HANYA KOLOM YANG DIPERLUKAN
        cols_to_keep = ["timestamp_col", "quantity_category", "quantity", "price", "total_amount", "month", "day", "hour"]
        actual_cols_in_df = df_temp.columns
        final_cols_to_select = [c for c in cols_to_keep if c in actual_cols_in_df]

        if not final_cols_to_select or "quantity_category" not in final_cols_to_select or "timestamp_col" not in final_cols_to_select: # Minimal target dan timestamp
            print(f"MINIMAL TEST ERROR: Not all required 'cols_to_keep' (esp. target/timestamp) are present. Selected: {final_cols_to_select}. Available: {actual_cols_in_df}")
            return None
        
        limited_df = df_temp.select(*final_cols_to_select)
        
        if limited_df.isEmpty():
            print("MINIMAL TEST: DataFrame is empty after selecting columns.")
            return None

        print(f"MINIMAL TEST: Count before limit: {limited_df.count()}")
        limited_df = limited_df.limit(1000000) 
        print(f"MINIMAL TEST: Limited count: {limited_df.count()}")
        
        # Hapus distinct dan orderBy untuk tes paling ringan
        return limited_df

    def _preprocess_data_rf(self, df):
        # ... (Kode _preprocess_data_rf Anda yang sudah ada, pastikan nama kolom huruf kecil) ...
        # Contoh (bagian pentingnya):
        print("Preprocessing data for Random Forest...")
        if df is None or df.isEmpty(): print("ERROR: DataFrame is None or empty in _preprocess_data_rf."); return None
        target_col = "quantity_category"
        if target_col not in df.columns: 
            print(f"ERROR: Target column '{target_col}' not found."); return None
        df = df.dropna(subset=[target_col])
        if df.isEmpty(): print(f"DataFrame empty after dropping NaNs in {target_col}."); return None
        
        numeric_features = ["quantity", "price", "total_amount", "month", "day", "hour"]
        for feature in numeric_features:
            if feature not in df.columns:
                print(f"WARNING: Numeric feature '{feature}' not found. Will not be used.")
                continue 
            df = df.withColumn(feature, when(col(feature).isNull(), 0.0).otherwise(col(feature)))
        print("Preprocessing complete for RF.")
        df.printSchema()
        return df


    def train_and_evaluate_model_rf(self, data_for_training, model_name_suffix):
        # ... (Kode train_and_evaluate_model_rf Anda yang sudah ada, pastikan nama kolom huruf kecil) ...
        # Contoh (bagian pentingnya):
        model_output_name = f"rf_model_{model_name_suffix}"
        print(f"\n--- Training {model_output_name} ---")
        if data_for_training is None or data_for_training.isEmpty():
            print(f"No data for training {model_output_name}. Skipping.")
            return

        processed_df = self._preprocess_data_rf(data_for_training)
        if processed_df is None or processed_df.isEmpty():
            print(f"Preprocessing failed for {model_output_name}. Skipping.")
            return

        feature_columns_to_use = ["quantity", "price", "total_amount", "month", "day", "hour"]
        actual_present_features = [f for f in feature_columns_to_use if f in processed_df.columns]
        if not actual_present_features:
             print(f"ERROR: No usable features for {model_output_name}. Skipping.")
             return
        print(f"Using features for {model_output_name}: {actual_present_features}")

        assembler = VectorAssembler(inputCols=actual_present_features, outputCol="features", handleInvalid="keep")
        label_indexer = StringIndexer(inputCol="quantity_category", outputCol="label", handleInvalid="keep")
        rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10, maxDepth=5, seed=42) # numTrees & maxDepth dikurangi untuk tes
        pipeline_stages = [assembler, label_indexer, rf_classifier]
        pipeline = Pipeline(stages=pipeline_stages)
        
        # Pastikan ada cukup data untuk split setelah limit
        if processed_df.count() < 10: # Perlu setidaknya beberapa baris untuk split
            print(f"Not enough data ({processed_df.count()} rows) after limit to perform train/test split for {model_output_name}. Skipping training.")
            return

        train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
        if train_df.isEmpty():
            print(f"ERROR: Training data is empty after split for {model_output_name}. Cannot train.")
            return
        
        print(f"Training {model_output_name} with {train_df.count()} records, testing with {test_df.count()} records.")
        try:
            pipeline_model = pipeline.fit(train_df)
            model_save_path = os.path.join(self.model_base_dir, model_output_name)
            print(f"Saving model {model_output_name} to: {model_save_path}")
            pipeline_model.write().overwrite().save(model_save_path)
            print(f"Model {model_output_name} saved successfully.")
            if not test_df.isEmpty():
                predictions = pipeline_model.transform(test_df)
                evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
                accuracy = evaluator.evaluate(predictions)
                evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
                f1_score = evaluator_f1.evaluate(predictions)
                print(f"--- Evaluation Metrics for {model_output_name} ---")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1 Score: {f1_score:.4f}")
            else:
                print(f"Test data is empty for {model_output_name}. Evaluation skipped.")
        except Exception as e:
            print(f"ERROR during training or evaluation of {model_output_name}: {e}")
            import traceback
            traceback.print_exc()


    def run_minimal_time_window_scenario(self, window_duration_minutes=5): # <<< NAMA FUNGSI DIUBAH
        """Trains ONLY ONE model based on the first time window of minimal data."""
        print(f"\n=== MINIMAL RF Training Scenario (First Window of {window_duration_minutes} min) ===")
        
        # _load_minimal_batch_data() akan mengembalikan DataFrame yang sudah sangat terbatas
        minimal_data_df = self._load_minimal_batch_data() 
        
        if minimal_data_df is None or minimal_data_df.isEmpty():
            print("MINIMAL TEST: No batch data loaded. Cannot run time window scenario.")
            return
        
        if "timestamp_col" not in minimal_data_df.columns:
            print("MINIMAL TEST ERROR: 'timestamp_col' is required but not found. Skipping RF training.")
            return

        # Karena data sudah sangat terbatas (dari 1 batch, di-limit 500),
        # kita bisa langsung gunakan data ini seolah-olah itu adalah window pertama.
        # Logika window yang kompleks tidak terlalu relevan lagi dengan data yang sudah di-limit.
        
        # Jika masih ingin simulasi filter window pada data yang sudah di-limit:
        # min_ts_row = minimal_data_df.agg(spark_min("timestamp_col").alias("min_ts")).collect()
        # if not min_ts_row or min_ts_row[0]["min_ts"] is None:
        #     print("MINIMAL TEST: Could not determine min timestamp from limited data. Using all limited data.")
        #     data_model_1 = minimal_data_df
        # else:
        #     first_timestamp = min_ts_row[0]["min_ts"]
        #     end_time_window_1 = first_timestamp + timedelta(minutes=window_duration_minutes)
        #     data_model_1 = minimal_data_df.filter(col("timestamp_col") <= end_time_window_1)
        #     print(f"MINIMAL TEST: Model 1 using data from limited set up to: {end_time_window_1}")

        # Langsung gunakan data yang sudah sangat di-limit untuk training
        data_model_1 = minimal_data_df
        
        self.train_and_evaluate_model_rf(data_model_1, f"minimal_window_test")

        # Skenario 2 dan 3 DIKOMENTARI untuk tes minimalis ini
        print("=== Minimal RF Training Scenario Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest Models for Retail Data from Batches.")
    parser.add_argument("--batch_data_dir", type=str, help="Directory containing batch CSV files.")
    parser.add_argument("--model_output_dir", type=str, help="Directory to save trained models.")
    parser.add_argument("--window_minutes", type=int, default=5, help="Duration of each time window in minutes (used if processing full data).")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("RetailRandomForestMinimalTest").getOrCreate() # Nama App diubah

    batch_dir_to_use = args.batch_data_dir if args.batch_data_dir else DEFAULT_BATCH_DATA_PATH_CFG
    model_dir_to_use = args.model_output_dir if args.model_output_dir else DEFAULT_MODEL_PATH_CFG
    window_duration = args.window_minutes # Ini mungkin tidak terlalu dipakai jika data sudah di-limit
    
    print(f"Final batch data directory being used: {batch_dir_to_use}")
    print(f"Final model output directory being used: {model_dir_to_use}")
    print(f"Time window duration (parameter): {window_duration} minutes")

    if not batch_dir_to_use or not os.path.exists(batch_dir_to_use):
         print(f"ERROR: Batch data directory '{batch_dir_to_use}' does not exist or not provided correctly!")
         spark.stop()
         sys.exit(1)
    if not model_dir_to_use:
        print(f"ERROR: Model output directory not specified!")
        spark.stop()
        sys.exit(1)

    trainer = RandomForestModelTrainer(spark, model_dir_to_use, batch_dir_to_use)
    # Panggil metode yang sudah dimodifikasi untuk tes minimalis
    trainer.run_minimal_time_window_scenario(window_duration_minutes=window_duration) 

    spark.stop()
    print("Spark session stopped successfully (Random Forest Minimal Test).")