# /tmp/app/spark/linear_regression_model.py

import argparse
import os
import sys
import glob

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# --- Konfigurasi Impor ---
try:
    PROJECT_ROOT_IN_CONTAINER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT_IN_CONTAINER not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_IN_CONTAINER)
    
    from config.config import MODEL_PATH as DEFAULT_MODEL_PATH_CFG
    from config.config import TRAINING_DATA_PATH as DEFAULT_TRAINING_DATA_PATH_CFG
    
    print(f"Successfully imported from config.py:")
    print(f"  DEFAULT_MODEL_PATH_CFG: {DEFAULT_MODEL_PATH_CFG}")
    print(f"  DEFAULT_TRAINING_DATA_PATH_CFG: {DEFAULT_TRAINING_DATA_PATH_CFG}")

except ImportError as e:
    print(f"CRITICAL ERROR: Could not import paths from config.config.")
    print(f"Ensure config.py is in '{PROJECT_ROOT_IN_CONTAINER}/config/' and defines MODEL_PATH and TRAINING_DATA_PATH.")
    print(f"Ensure '{PROJECT_ROOT_IN_CONTAINER}/config/' contains an __init__.py file if 'config' is treated as a package.")
    print(f"ImportError: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)
except AttributeError as e:
    print(f"CRITICAL ERROR: A required path variable (e.g., MODEL_PATH or TRAINING_DATA_PATH) is missing from config.py.")
    print(f"AttributeError: {e}")
    sys.exit(1)


class LinearRegressionModelTrainer:
    def __init__(self, spark_session, model_base_dir, training_data_base_dir):
        self.spark = spark_session
        self.model_base_dir = model_base_dir
        self.training_data_base_dir = training_data_base_dir
        os.makedirs(self.model_base_dir, exist_ok=True)
        print(f"LinearRegressionModelTrainer initialized.")
        print(f"  Models will be saved to: {self.model_base_dir}")
        print(f"  Training data will be read from: {self.training_data_base_dir}")

    def _load_and_combine_data(self, csv_file_names_list):
        all_dfs = []
        for file_name in csv_file_names_list:
            file_path = os.path.join(self.training_data_base_dir, file_name)
            if not os.path.exists(file_path):
                print(f"WARNING: Data file '{file_path}' not found. Skipping.")
                continue
            
            print(f"Reading data from: {file_path}")
            try:
                df = self.spark.read.csv(file_path, header=True, inferSchema=True)
                # Sesuaikan nama kolom dengan CSV (semua huruf kecil)
                required_numeric_cols = ["quantity", "price", "total_amount"]
                required_int_cols = ["month", "day", "hour"]
                for c in required_numeric_cols:
                    if c in df.columns: df = df.withColumn(c, col(c).cast("double"))
                    else: print(f"WARNING: Column '{c}' not found in {file_path} for numeric casting.")
                for c in required_int_cols:
                    if c in df.columns: df = df.withColumn(c, col(c).cast("integer"))
                    else: print(f"WARNING: Column '{c}' not found in {file_path} for integer casting.")
                all_dfs.append(df)
            except Exception as e:
                print(f"ERROR: Could not read or process CSV file {file_path}: {e}")
        
        if not all_dfs:
            print("No dataframes were loaded. Cannot proceed.")
            return None
        combined_df = all_dfs[0]
        for i in range(1, len(all_dfs)): combined_df = combined_df.unionByName(all_dfs[i])
        return combined_df.distinct()

    def _preprocess_data(self, df):
        print("Preprocessing data...")
        if df is None: print("ERROR: DataFrame is None in _preprocess_data."); return None
        # Sesuaikan nama kolom target
        if "total_amount" not in df.columns: print("ERROR: Target column 'total_amount' not found."); return None
        df = df.dropna(subset=["total_amount"])
        if df.isEmpty(): print("DataFrame empty after dropping NaNs in total_amount."); return None
        
        # Sesuaikan nama kolom fitur
        numeric_features = ["quantity", "price", "month", "day", "hour"]
        for feature in numeric_features:
            if feature not in df.columns:
                print(f"WARNING: Feature '{feature}' not found. Will not be used.")
                continue 
            df = df.withColumn(feature, when(col(feature).isNull(), 0.0).otherwise(col(feature)))
        print("Preprocessing complete.")
        df.printSchema()
        return df

    def train_and_evaluate_model(self, data_for_training, model_name_suffix):
        model_output_name = f"linear_model_{model_name_suffix}"
        print(f"\n--- Training {model_output_name} ---")

        if data_for_training is None or data_for_training.isEmpty():
            print(f"No data provided or data is empty for training {model_output_name}. Skipping.")
            return

        processed_df = self._preprocess_data(data_for_training)
        if processed_df is None or processed_df.isEmpty():
            print(f"Data preprocessing failed or resulted in empty DataFrame for {model_output_name}. Skipping.")
            return

        # Sesuaikan nama kolom fitur
        feature_columns_to_use = ["quantity", "price", "month", "day", "hour"]
        actual_present_features = [f for f in feature_columns_to_use if f in processed_df.columns]
        if not actual_present_features:
             print(f"ERROR: No usable features found in processed_df for {model_output_name}. Columns: {processed_df.columns}. Skipping.")
             return
        print(f"Using features for {model_output_name}: {actual_present_features}")

        assembler = VectorAssembler(inputCols=actual_present_features, outputCol="unscaled_features", handleInvalid="skip")
        scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
        # Sesuaikan nama kolom label
        lr = LinearRegression(featuresCol="features", labelCol="total_amount")
        pipeline = Pipeline(stages=[assembler, scaler, lr])
        
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
                # Sesuaikan nama kolom label
                evaluator_rmse = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
                rmse = evaluator_rmse.evaluate(predictions)
                evaluator_r2 = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2")
                r2 = evaluator_r2.evaluate(predictions)
                print(f"--- Evaluation Metrics for {model_output_name} ---")
                print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
                print(f"  R-squared (R2): {r2:.4f}")
            else:
                print(f"Test data is empty for {model_output_name}. Evaluation skipped.")
        except Exception as e:
            print(f"ERROR during training or evaluation of {model_output_name}: {e}")
            import traceback
            traceback.print_exc()

    def run_training_scenarios(self):
        print("\n=== Starting Linear Regression Training Scenarios ===")
        data_model_1 = self._load_and_combine_data(["linear_model_1.csv"])
        self.train_and_evaluate_model(data_model_1, "scenario1_from_file1")

        data_model_2 = self._load_and_combine_data(["linear_model_1.csv", "linear_model_2.csv"])
        self.train_and_evaluate_model(data_model_2, "scenario2_from_files12")

        data_model_3 = self._load_and_combine_data(["linear_model_1.csv", "linear_model_2.csv", "linear_model_3.csv"])
        self.train_and_evaluate_model(data_model_3, "scenario3_from_files123")
        
        print("=== All Linear Regression Training Scenarios Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Linear Regression Models for Retail Data.")
    parser.add_argument("--training_data_dir", type=str, help="Directory containing training CSV files.")
    parser.add_argument("--model_output_dir", type=str, help="Directory to save trained models.")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("RetailLinearRegressionManualTrain").getOrCreate()

    training_dir_to_use = args.training_data_dir if args.training_data_dir else DEFAULT_TRAINING_DATA_PATH_CFG
    model_dir_to_use = args.model_output_dir if args.model_output_dir else DEFAULT_MODEL_PATH_CFG
    
    print(f"Final training data directory being used: {training_dir_to_use}")
    print(f"Final model output directory being used: {model_dir_to_use}")

    if not training_dir_to_use or not os.path.exists(training_dir_to_use):
         print(f"ERROR: Training data directory '{training_dir_to_use}' does not exist or not provided correctly!")
         spark.stop()
         sys.exit(1)
    if not model_dir_to_use:
        print(f"ERROR: Model output directory not specified!")
        spark.stop()
        sys.exit(1)

    trainer = LinearRegressionModelTrainer(spark, model_dir_to_use, training_dir_to_use)
    trainer.run_training_scenarios()

    spark.stop()
    print("Spark session stopped successfully.")