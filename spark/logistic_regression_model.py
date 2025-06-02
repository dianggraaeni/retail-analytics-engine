# /tmp/app/spark/logistic_regression_model.py

import argparse
import os
import sys
import glob

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression # Impor yang benar
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# --- Konfigurasi Impor ---
try:
    PROJECT_ROOT_IN_CONTAINER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT_IN_CONTAINER not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_IN_CONTAINER)
    
    from config.config import MODEL_PATH as DEFAULT_MODEL_PATH_CFG
    from config.config import BATCH_DATA_PATH as DEFAULT_BATCH_DATA_PATH_CFG # Menggunakan BATCH_DATA_PATH
    
    print(f"Successfully imported from config.py for Logistic Regression:")
    print(f"  DEFAULT_MODEL_PATH_CFG: {DEFAULT_MODEL_PATH_CFG}")
    print(f"  DEFAULT_BATCH_DATA_PATH_CFG: {DEFAULT_BATCH_DATA_PATH_CFG}")

except ImportError as e:
    print(f"CRITICAL ERROR (LR): Could not import paths from config.config.")
    print(f"Ensure config.py is in '{PROJECT_ROOT_IN_CONTAINER}/config/' and defines MODEL_PATH and BATCH_DATA_PATH.")
    print(f"ImportError: {e}")
    sys.exit(1)
except AttributeError as e:
    print(f"CRITICAL ERROR (LR): A required path variable (MODEL_PATH or BATCH_DATA_PATH) is missing from config.py.")
    print(f"AttributeError: {e}")
    sys.exit(1)


class LogisticRegressionModelTrainer:
    def __init__(self, spark_session, model_base_dir, batch_data_base_dir):
        self.spark = spark_session
        self.model_base_dir = model_base_dir
        self.batch_data_base_dir = batch_data_base_dir # Path ke folder /batches
        os.makedirs(self.model_base_dir, exist_ok=True)
        print(f"LogisticRegressionModelTrainer initialized.")
        print(f"  Models will be saved to: {self.model_base_dir}")
        print(f"  Batch data will be read from: {self.batch_data_base_dir}")

    def _load_specific_batches(self, batch_file_paths_list):
        """Loads data from a specific list of full batch file paths."""
        all_dfs = []
        for file_path in batch_file_paths_list: # Sekarang menerima list full path
            if not os.path.exists(file_path):
                print(f"WARNING: Batch file '{file_path}' not found. Skipping.")
                continue
            
            print(f"Reading data from batch file: {file_path}")
            try:
                df = self.spark.read.csv(file_path, header=True, inferSchema=True)
                # Sesuaikan nama kolom dengan CSV (semua huruf kecil)
                # Target untuk model ini adalah 'high_value_customer'
                required_cols = ["quantity", "price", "total_amount", "month", "day", "hour", "high_value_customer"]
                
                for c in required_cols:
                    if c not in df.columns:
                        print(f"WARNING: Column '{c}' not found in {file_path}.")
                
                # Casting (contoh, sesuaikan dengan kebutuhan)
                if "quantity" in df.columns: df = df.withColumn("quantity", col("quantity").cast("double"))
                if "price" in df.columns: df = df.withColumn("price", col("price").cast("double"))
                if "total_amount" in df.columns: df = df.withColumn("total_amount", col("total_amount").cast("double"))
                if "month" in df.columns: df = df.withColumn("month", col("month").cast("integer"))
                if "day" in df.columns: df = df.withColumn("day", col("day").cast("integer"))
                if "hour" in df.columns: df = df.withColumn("hour", col("hour").cast("integer"))
                if "high_value_customer" in df.columns: df = df.withColumn("high_value_customer", col("high_value_customer").cast("integer")) # Asumsi 0 atau 1
                
                all_dfs.append(df)
            except Exception as e:
                print(f"ERROR: Could not read or process CSV file {file_path}: {e}")
        
        if not all_dfs:
            print("No dataframes were loaded from batches. Cannot proceed.")
            return None
        
        combined_df = all_dfs[0]
        for i in range(1, len(all_dfs)): combined_df = combined_df.unionByName(all_dfs[i])
        return combined_df.distinct()

    def _preprocess_data(self, df):
        print("Preprocessing data for Logistic Regression...")
        if df is None: print("ERROR: DataFrame is None in _preprocess_data."); return None
        
        target_col = "high_value_customer"
        if target_col not in df.columns: 
            print(f"ERROR: Target column '{target_col}' not found."); return None
        df = df.dropna(subset=[target_col]) # Hapus baris jika target null
        if df.isEmpty(): print(f"DataFrame empty after dropping NaNs in {target_col}."); return None
        
        # Cek apakah target adalah biner (0 dan 1)
        distinct_targets = df.select(target_col).distinct().collect()
        target_values = [row[target_col] for row in distinct_targets]
        if not all(value in [0, 1] for value in target_values):
            print(f"WARNING: Target column '{target_col}' is not strictly binary (0 or 1). Values found: {target_values}. This might affect Logistic Regression.")
            # Anda mungkin perlu melakukan mapping tambahan di sini jika nilainya bukan 0/1

        feature_cols = ["quantity", "price", "total_amount", "month", "day", "hour"]
        for feature in feature_cols:
            if feature not in df.columns:
                print(f"WARNING: Feature '{feature}' not found. Will not be used.")
                continue 
            df = df.withColumn(feature, when(col(feature).isNull(), 0.0).otherwise(col(feature)))
        
        print("Preprocessing complete.")
        df.printSchema()
        return df

    def train_and_evaluate_model(self, data_for_training, model_name_suffix):
        model_output_name = f"logistic_model_{model_name_suffix}" # Nama model disesuaikan
        print(f"\n--- Training {model_output_name} ---")

        if data_for_training is None or data_for_training.isEmpty():
            print(f"No data provided or data is empty for training {model_output_name}. Skipping.")
            return

        processed_df = self._preprocess_data(data_for_training)
        if processed_df is None or processed_df.isEmpty():
            print(f"Data preprocessing failed or resulted in empty DataFrame for {model_output_name}. Skipping.")
            return

        feature_columns_to_use = ["quantity", "price", "total_amount", "month", "day", "hour"]
        actual_present_features = [f for f in feature_columns_to_use if f in processed_df.columns]
        if not actual_present_features:
             print(f"ERROR: No usable features found in processed_df for {model_output_name}. Columns: {processed_df.columns}. Skipping.")
             return
        print(f"Using features for {model_output_name}: {actual_present_features}")

        assembler = VectorAssembler(inputCols=actual_present_features, outputCol="unscaled_features", handleInvalid="skip")
        scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
        # Gunakan LogisticRegression dengan labelCol yang benar
        lr_classifier = LogisticRegression(featuresCol="features", labelCol="high_value_customer")
        pipeline = Pipeline(stages=[assembler, scaler, lr_classifier])
        
        train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
        
        if train_df.isEmpty():
            print(f"ERROR: Training data is empty after split for {model_output_name}. Cannot train.")
            return
        
        print(f"Training {model_output_name} with {train_df.count()} records, testing with {test_df.count()} records (if test_df not empty).")

        try:
            print(f"Fitting pipeline for {model_output_name}...")
            pipeline_model = pipeline.fit(train_df)
            print(f"Pipeline fitting complete for {model_output_name}.")

            model_save_path = os.path.join(self.model_base_dir, model_output_name)
            print(f"Saving model {model_output_name} to: {model_save_path}")
            pipeline_model.write().overwrite().save(model_save_path)
            print(f"Model {model_output_name} saved successfully.")

            if not test_df.isEmpty():
                print(f"Evaluating {model_output_name}...")
                predictions = pipeline_model.transform(test_df)
                # Gunakan BinaryClassificationEvaluator dengan labelCol yang benar
                evaluator = BinaryClassificationEvaluator(labelCol="high_value_customer", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
                auc = evaluator.evaluate(predictions)
                
                # Anda juga bisa menghitung akurasi
                # evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="high_value_customer", predictionCol="prediction", metricName="accuracy")
                # accuracy = evaluator_accuracy.evaluate(predictions) # LogisticRegression juga menghasilkan kolom 'prediction'
                
                print(f"--- Evaluation Metrics for {model_output_name} ---")
                print(f"  Area Under ROC (AUC): {auc:.4f}")
                # print(f"  Accuracy: {accuracy:.4f}")
            else:
                print(f"Test data is empty for {model_output_name}. Evaluation skipped.")
        except Exception as e:
            print(f"ERROR during training or evaluation of {model_output_name}: {e}")
            import traceback
            traceback.print_exc()

    def run_fixed_size_batch_training_scenarios(self):
        """Train models with incrementally larger sets of batch files."""
        print("\n=== Starting Logistic Regression Training Scenarios (Fixed Size / Incremental Batches) ===")
        
        # Dapatkan semua file batch dan urutkan (berdasarkan nama, asumsi nama batch_0, batch_1, dst.)
        # Atau bisa juga diurutkan berdasarkan waktu modifikasi jika nama tidak berurutan
        all_batch_files_glob = os.path.join(self.batch_data_base_dir, "batch_*.csv")
        sorted_batch_file_paths = sorted(glob.glob(all_batch_files_glob))

        if not sorted_batch_file_paths:
            print("No batch files found. Cannot run training scenarios.")
            return

        # Skenario 1: Menggunakan batch pertama saja
        if len(sorted_batch_file_paths) >= 1:
            data_model_1 = self._load_specific_batches(sorted_batch_file_paths[:1]) # List berisi 1 file path
            self.train_and_evaluate_model(data_model_1, "batch_1_only")
        else:
            print("Not enough batch files for Model 1 (batch_1_only).")

        # Skenario 2: Menggunakan dua batch pertama
        if len(sorted_batch_file_paths) >= 2:
            data_model_2 = self._load_specific_batches(sorted_batch_file_paths[:2]) # List berisi 2 file path
            self.train_and_evaluate_model(data_model_2, "batches_1_to_2")
        else:
            print("Not enough batch files for Model 2 (batches_1_to_2).")

        # Skenario 3: Menggunakan tiga batch pertama
        if len(sorted_batch_file_paths) >= 3:
            data_model_3 = self._load_specific_batches(sorted_batch_file_paths[:3]) # List berisi 3 file path
            self.train_and_evaluate_model(data_model_3, "batches_1_to_3")
        else:
            print("Not enough batch files for Model 3 (batches_1_to_3).")
        
        print("=== All Logistic Regression Training Scenarios Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression Models for Retail Data from Batches.")
    parser.add_argument("--batch_data_dir", type=str, help="Directory containing batch CSV files.")
    parser.add_argument("--model_output_dir", type=str, help="Directory to save trained models.")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("RetailLogisticRegressionBatchTrain").getOrCreate()

    batch_dir_to_use = args.batch_data_dir if args.batch_data_dir else DEFAULT_BATCH_DATA_PATH_CFG
    model_dir_to_use = args.model_output_dir if args.model_output_dir else DEFAULT_MODEL_PATH_CFG
    
    print(f"Final batch data directory being used: {batch_dir_to_use}")
    print(f"Final model output directory being used: {model_dir_to_use}")

    if not batch_dir_to_use or not os.path.exists(batch_dir_to_use):
         print(f"ERROR: Batch data directory '{batch_dir_to_use}' does not exist or not provided correctly!")
         spark.stop()
         sys.exit(1)
    if not model_dir_to_use:
        print(f"ERROR: Model output directory not specified!")
        spark.stop()
        sys.exit(1)

    trainer = LogisticRegressionModelTrainer(spark, model_dir_to_use, batch_dir_to_use)
    trainer.run_fixed_size_batch_training_scenarios() # Panggil metode yang sesuai

    spark.stop()
    print("Spark session stopped successfully (Logistic Regression).")