from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import *

app = Flask(__name__)

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models (simplified for demo)"""
        # In real implementation, you would load Spark models
        # For demo, we'll simulate model responses
        self.models = {
            'linear_model_1': {'type': 'regression', 'version': 1},
            'linear_model_2': {'type': 'regression', 'version': 2},
            'linear_model_3': {'type': 'regression', 'version': 3},
            'rf_model_window_1': {'type': 'classification', 'version': 1},
            'rf_model_window_2': {'type': 'classification', 'version': 2},
            'rf_model_window_3': {'type': 'classification', 'version': 3},
            'logistic_model_batch_1': {'type': 'binary_classification', 'version': 1},
            'logistic_model_batch_2': {'type': 'binary_classification', 'version': 2},
            'logistic_model_batch_3': {'type': 'binary_classification', 'version': 3}
        }
        logger.info(f"Loaded {len(self.models)} models")

predictor = ModelPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(predictor.models)
    })

# Linear Regression Endpoints (3 models)
@app.route('/predict/total_amount/v1', methods=['POST'])
def predict_total_amount_v1():
    """Linear Regression Model 1 - Predict total amount using 1/3 data"""
    try:
        data = request.json
        
        # Validate input
        required_fields = ['quantity', 'price', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Simulate prediction (replace with actual Spark model inference)
        quantity = float(data['quantity'])
        price = float(data['price'])
        
        # Simple calculation for demo
        predicted_amount = quantity * price * (1 + np.random.normal(0, 0.1))
        
        return jsonify({
            'model': 'linear_model_1',
            'prediction': round(predicted_amount, 2),
            'confidence': 0.85,
            'features_used': required_fields,
            'model_version': 1
        })
        
    except Exception as e:
        logger.error(f"Error in predict_total_amount_v1: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/total_amount/v2', methods=['POST'])
def predict_total_amount_v2():
    """Linear Regression Model 2 - Predict total amount using 2/3 data"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        quantity = float(data['quantity'])
        price = float(data['price'])
        month = int(data['month'])
        
        # Improved prediction with more data
        seasonal_factor = 1.2 if month in [11, 12] else 1.0
        predicted_amount = quantity * price * seasonal_factor * (1 + np.random.normal(0, 0.08))
        
        return jsonify({
            'model': 'linear_model_2',
            'prediction': round(predicted_amount, 2),
            'confidence': 0.91,
            'features_used': required_fields,
            'model_version': 2,
            'seasonal_adjustment': seasonal_factor
        })
        
    except Exception as e:
        logger.error(f"Error in predict_total_amount_v2: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/total_amount/v3', methods=['POST'])
def predict_total_amount_v3():
    """Linear Regression Model 3 - Predict total amount using all data"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        quantity = float(data['quantity'])
        price = float(data['price'])
        month = int(data['month'])
        hour = int(data['hour'])
        
        # Most sophisticated prediction with all data
        seasonal_factor = 1.3 if month in [11, 12] else 1.0
        hourly_factor = 1.1 if 9 <= hour <= 17 else 0.9
        
        predicted_amount = quantity * price * seasonal_factor * hourly_factor * (1 + np.random.normal(0, 0.05))
        
        return jsonify({
            'model': 'linear_model_3',
            'prediction': round(predicted_amount, 2),
            'confidence': 0.95,
            'features_used': required_fields,
            'model_version': 3,
            'seasonal_adjustment': seasonal_factor,
            'hourly_adjustment': hourly_factor
        })
        
    except Exception as e:
        logger.error(f"Error in predict_total_amount_v3: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Random Forest Endpoints (3 models)
@app.route('/predict/quantity_category/v1', methods=['POST'])
def predict_quantity_category_v1():
    """Random Forest Model 1 - Predict quantity category using first time window"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'total_amount', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        quantity = float(data['quantity'])
        
        # Simple rule-based classification for demo
        if quantity <= 1:
            category = 'Low'
            confidence = 0.75
        elif quantity <= 5:
            category = 'Medium'
            confidence = 0.80
        elif quantity <= 20:
            category = 'High'
            confidence = 0.85
        else:
            category = 'Bulk'
            confidence = 0.90
        
        return jsonify({
            'model': 'rf_model_window_1',
            'predicted_category': category,
            'confidence': confidence,
            'probabilities': {
                'Low': 0.2 if category != 'Low' else 0.75,
                'Medium': 0.3 if category != 'Medium' else 0.80,
                'High': 0.3 if category != 'High' else 0.85,
                'Bulk': 0.2 if category != 'Bulk' else 0.90
            },
            'features_used': required_fields,
            'model_version': 1
        })
        
    except Exception as e:
        logger.error(f"Error in predict_quantity_category_v1: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/quantity_category/v2', methods=['POST'])
def predict_quantity_category_v2():
    """Random Forest Model 2 - Predict quantity category using two time windows"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'total_amount', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        quantity = float(data['quantity'])
        total_amount = float(data['total_amount'])
        
        # Enhanced classification considering total amount
        if quantity <= 1 and total_amount < 10:
            category = 'Low'
            confidence = 0.82
        elif quantity <= 5 and total_amount < 50:
            category = 'Medium'
            confidence = 0.87
        elif quantity <= 20 and total_amount < 200:
            category = 'High'
            confidence = 0.90
        else:
            category = 'Bulk'
            confidence = 0.93
        
        return jsonify({
            'model': 'rf_model_window_2',
            'predicted_category': category,
            'confidence': confidence,
            'probabilities': {
                'Low': 0.15 if category != 'Low' else 0.82,
                'Medium': 0.25 if category != 'Medium' else 0.87,
                'High': 0.35 if category != 'High' else 0.90,
                'Bulk': 0.25 if category != 'Bulk' else 0.93
            },
            'features_used': required_fields,
            'model_version': 2,
            'amount_threshold_used': True
        })
        
    except Exception as e:
        logger.error(f"Error in predict_quantity_category_v2: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/quantity_category/v3', methods=['POST'])
def predict_quantity_category_v3():
    """Random Forest Model 3 - Predict quantity category using all time windows"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'total_amount', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        quantity = float(data['quantity'])
        total_amount = float(data['total_amount'])
        month = int(data['month'])
        hour = int(data['hour'])
        
        # Most sophisticated classification with temporal features
        seasonal_boost = 1.2 if month in [11, 12] else 1.0
        time_boost = 1.1 if 9 <= hour <= 17 else 0.9
        
        adjusted_quantity = quantity * seasonal_boost * time_boost
        
        if adjusted_quantity <= 1.2 and total_amount < 12:
            category = 'Low'
            confidence = 0.89
        elif adjusted_quantity <= 6 and total_amount < 60:
            category = 'Medium'
            confidence = 0.92
        elif adjusted_quantity <= 24 and total_amount < 240:
            category = 'High'
            confidence = 0.95
        else:
            category = 'Bulk'
            confidence = 0.97
        
        return jsonify({
            'model': 'rf_model_window_3',
            'predicted_category': category,
            'confidence': confidence,
            'probabilities': {
                'Low': 0.1 if category != 'Low' else 0.89,
                'Medium': 0.2 if category != 'Medium' else 0.92,
                'High': 0.4 if category != 'High' else 0.95,
                'Bulk': 0.3 if category != 'Bulk' else 0.97
            },
            'features_used': required_fields,
            'model_version': 3,
            'temporal_adjustments': {
                'seasonal_boost': seasonal_boost,
                'time_boost': time_boost,
                'adjusted_quantity': round(adjusted_quantity, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in predict_quantity_category_v3: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Logistic Regression Endpoints (3 models)
@app.route('/predict/high_value_customer/v1', methods=['POST'])
def predict_high_value_customer_v1():
    """Logistic Regression Model 1 - Predict high-value customer using first batch"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'total_amount', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        total_amount = float(data['total_amount'])
        
        # Simple threshold-based classification
        threshold = 50.0
        is_high_value = 1 if total_amount > threshold else 0
        probability = min(0.95, max(0.05, total_amount / 100))
        
        return jsonify({
            'model': 'logistic_model_batch_1',
            'is_high_value_customer': bool(is_high_value),
            'probability': round(probability, 3),
            'confidence': 0.78,
            'threshold_used': threshold,
            'features_used': required_fields,
            'model_version': 1
        })
        
    except Exception as e:
        logger.error(f"Error in predict_high_value_customer_v1: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/high_value_customer/v2', methods=['POST'])
def predict_high_value_customer_v2():
    """Logistic Regression Model 2 - Predict high-value customer using two batches"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'total_amount', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        total_amount = float(data['total_amount'])
        quantity = float(data['quantity'])
        
        # Enhanced model considering quantity
        base_score = total_amount / 75
        quantity_bonus = quantity * 0.1
        final_score = base_score + quantity_bonus
        
        probability = 1 / (1 + np.exp(-final_score))  # Sigmoid function
        is_high_value = 1 if probability > 0.5 else 0
        
        return jsonify({
            'model': 'logistic_model_batch_2',
            'is_high_value_customer': bool(is_high_value),
            'probability': round(probability, 3),
            'confidence': 0.85,
            'score_components': {
                'base_score': round(base_score, 3),
                'quantity_bonus': round(quantity_bonus, 3),
                'final_score': round(final_score, 3)
            },
            'features_used': required_fields,
            'model_version': 2
        })
        
    except Exception as e:
        logger.error(f"Error in predict_high_value_customer_v2: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/high_value_customer/v3', methods=['POST'])
def predict_high_value_customer_v3():
    """Logistic Regression Model 3 - Predict high-value customer using three batches"""
    try:
        data = request.json
        
        required_fields = ['quantity', 'price', 'total_amount', 'month', 'day', 'hour']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        total_amount = float(data['total_amount'])
        quantity = float(data['quantity'])
        month = int(data['month'])
        hour = int(data['hour'])
        
        # Most sophisticated model with temporal features
        base_score = total_amount / 60
        quantity_bonus = quantity * 0.15
        seasonal_bonus = 0.3 if month in [11, 12] else 0
        time_bonus = 0.2 if 9 <= hour <= 17 else 0
        
        final_score = base_score + quantity_bonus + seasonal_bonus + time_bonus
        probability = 1 / (1 + np.exp(-final_score))
        is_high_value = 1 if probability > 0.45 else 0  # Lower threshold for more sensitivity
        
        return jsonify({
            'model': 'logistic_model_batch_3',
            'is_high_value_customer': bool(is_high_value),
            'probability': round(probability, 3),
            'confidence': 0.92,
            'score_components': {
                'base_score': round(base_score, 3),
                'quantity_bonus': round(quantity_bonus, 3),
                'seasonal_bonus': seasonal_bonus,
                'time_bonus': time_bonus,
                'final_score': round(final_score, 3)
            },
            'threshold': 0.45,
            'features_used': required_fields,
            'model_version': 3
        })
        
    except Exception as e:
        logger.error(f"Error in predict_high_value_customer_v3: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Model comparison endpoint
@app.route('/compare/models', methods=['POST'])
def compare_models():
    """Compare predictions across all models"""
    try:
        data = request.json
        
        results = {}
        
        # Linear regression comparison
        results['total_amount_predictions'] = {}
        for version in [1, 2, 3]:
            endpoint_map = {
                1: predict_total_amount_v1,
                2: predict_total_amount_v2,
                3: predict_total_amount_v3
            }
            
            with app.test_request_context(json=data):
                response = endpoint_map[version]()
                if hasattr(response, 'json'):
                    results['total_amount_predictions'][f'v{version}'] = response.json
        
        # Add metadata
        results['comparison_timestamp'] = datetime.now().isoformat()
        results['input_data'] = data
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in compare_models: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Batch processing status
@app.route('/status/batches', methods=['GET'])
def batch_status():
    """Get current batch processing status"""
    try:
        import glob
        batch_files = glob.glob(os.path.join(BATCH_DATA_PATH, "batch_*.csv"))
        
        status = {
            'total_batches': len(batch_files),
            'latest_batch': max(batch_files, key=os.path.getctime) if batch_files else None,
            'batch_directory': BATCH_DATA_PATH,
            'models_available': list(predictor.models.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error in batch_status: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)