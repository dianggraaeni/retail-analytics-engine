import requests
import json

BASE_URL = "http://localhost:5000"

def test_linear_regression():
    """Test Linear Regression endpoints"""
    print("Testing Linear Regression Models...")
    
    test_data = {
        "quantity": 5,
        "price": 15.50,
        "month": 12,
        "day": 15,
        "hour": 14
    }
    
    for version in [1, 2, 3]:
        url = f"{BASE_URL}/predict/total_amount/v{version}"
        response = requests.post(url, json=test_data)
        print(f"Linear Model V{version}: {response.json()}")

def test_random_forest():
    """Test Random Forest endpoints"""
    print("\nTesting Random Forest Models...")
    
    test_data = {
        "quantity": 8,
        "price": 12.75,
        "total_amount": 102.0,
        "month": 11,
        "day": 20,
        "hour": 15
    }
    
    for version in [1, 2, 3]:
        url = f"{BASE_URL}/predict/quantity_category/v{version}"
        response = requests.post(url, json=test_data)
        print(f"Random Forest V{version}: {response.json()}")

def test_logistic_regression():
    """Test Logistic Regression endpoints"""
    print("\nTesting Logistic Regression Models...")
    
    test_data = {
        "quantity": 3,
        "price": 25.00,
        "total_amount": 75.0,
        "month": 12,
        "day": 10,
        "hour": 16
    }
    
    for version in [1, 2, 3]:
        url = f"{BASE_URL}/predict/high_value_customer/v{version}"
        response = requests.post(url, json=test_data)
        print(f"Logistic Model V{version}: {response.json()}")

def test_system_status():
    """Test system status endpoints"""
    print("\nTesting System Status...")
    
    # Health check
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health: {response.json()}")
    
    # Batch status
    response = requests.get(f"{BASE_URL}/status/batches")
    print(f"Batch Status: {response.json()}")

if __name__ == "__main__":
    test_system_status()
    test_linear_regression()
    test_random_forest()
    test_logistic_regression()