#!/usr/bin/env python3
"""
CKD API Test Client
Demonstrates single predictions, batch JSON, and file uploads
"""

import requests
import json
import pandas as pd
import os

# API Configuration
API_BASE_URL = "http://localhost:5000"

def test_health():
    """Test API health check"""
    print("🏥 Testing API Health...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Get model information"""
    print("📊 Getting Model Information...")
    response = requests.get(f"{API_BASE_URL}/model/info")
    if response.status_code == 200:
        info = response.json()
        print("Model Type:", info['model_type'])
        print("Required Features:", info['features_required'])
        print("Clinical Thresholds:", json.dumps(info['clinical_thresholds'], indent=2))
    else:
        print(f"Error: {response.status_code} - {response.text}")
    print()

def test_single_prediction():
    """Test single patient prediction"""
    print("👤 Testing Single Patient Prediction...")
    
    # High-risk patient (elevated creatinine and urea)
    patient_data = {
        "patient_id": "TEST_001",
        "age": 68,
        "weight": 72,
        "creatinine": 240,  # High
        "sodium": 133,
        "potassium": 5.7,   # High
        "glucose": 9.0,
        "urea": 20,         # High
        "gender": "male"
    }
    
    response = requests.post(f"{API_BASE_URL}/predict/single", json=patient_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction successful!")
        print(f"Patient ID: {result['result']['patient_id']}")
        print(f"CKD Detected: {result['result']['prediction']['ckd_detected']}")
        print(f"CKD Probability: {result['result']['prediction']['ckd_probability']:.3f}")
        print(f"Risk Level: {result['result']['prediction']['risk_level']}")
        print(f"Clinical Flags: {result['result']['clinical_assessment']['flags']}")
        print(f"Recommendation: {result['result']['clinical_assessment']['recommendation']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.json()}")
    print()

def test_batch_json():
    """Test batch prediction with JSON array"""
    print("👥 Testing Batch JSON Prediction...")
    
    batch_data = {
        "patients": [
            {
                "patient_id": "BATCH_001",
                "age": 68, "weight": 72, "creatinine": 240,
                "sodium": 133, "potassium": 5.7, "glucose": 9.0,
                "urea": 20, "gender": "male"
            },
            {
                "patient_id": "BATCH_002", 
                "age": 30, "weight": 70, "creatinine": 85,
                "sodium": 139, "potassium": 4.3, "glucose": 5.1,
                "urea": 5.0, "gender": "male"
            },
            {
                "patient_id": "BATCH_003",
                "age": 55, "weight": 80, "creatinine": 210,
                "sodium": 136, "potassium": 5.3, "glucose": 7.5,
                "urea": 17, "gender": "female"
            }
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/predict/batch/json", json=batch_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Batch prediction successful!")
        print(f"Processed: {result['processed']} patients")
        print(f"Errors: {result['errors']}")
        
        print("\nResults:")
        for r in result['results']:
            print(f"  • {r['patient_id']}: CKD={r['ckd_detected']}, "
                  f"Prob={r['ckd_probability']:.3f}, Risk={r['risk_level']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.json()}")
    print()

def create_test_csv():
    """Create a test CSV file for batch upload"""
    print("📄 Creating Test CSV File...")
    
    test_data = [
        {
            "patient_id": "CSV_001", "age": 68, "weight": 72, "creatinine": 240,
            "sodium": 133, "potassium": 5.7, "glucose": 9.0, "urea": 20, "gender": "male"
        },
        {
            "patient_id": "CSV_002", "age": 30, "weight": 70, "creatinine": 85,
            "sodium": 139, "potassium": 4.3, "glucose": 5.1, "urea": 5.0, "gender": "male"
        },
        {
            "patient_id": "CSV_003", "age": 55, "weight": 80, "creatinine": 210,
            "sodium": 136, "potassium": 5.3, "glucose": 7.5, "urea": 17, "gender": "female"
        },
        {
            "patient_id": "CSV_004", "age": 40, "weight": 60, "creatinine": 95,
            "sodium": 140, "potassium": 4.0, "glucose": 4.8, "urea": 4.8, "gender": "female"
        },
        {
            "patient_id": "CSV_005", "age": 71, "weight": 65, "creatinine": 230,
            "sodium": 132, "potassium": 5.5, "glucose": 8.0, "urea": 19, "gender": "female"
        }
    ]
    
    df = pd.DataFrame(test_data)
    csv_path = "test_patients.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Test CSV created: {csv_path}")
    return csv_path

def test_batch_csv_upload(csv_path):
    """Test batch prediction with CSV file upload"""
    print("📤 Testing CSV Batch Upload...")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    with open(csv_path, 'rb') as f:
        files = {'file': (csv_path, f, 'text/csv')}
        response = requests.post(f"{API_BASE_URL}/predict/batch", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ CSV batch prediction successful!")
        print(f"Processed: {result['processed']} patients")
        print(f"Errors: {result['errors']}")
        
        print("\nResults:")
        for r in result['results']:
            print(f"  • {r['patient_id']}: CKD={r['ckd_detected']}, "
                  f"Prob={r['ckd_probability']:.3f}, Risk={r['risk_level']}")
            if r['clinical_flags']:
                print(f"    Flags: {', '.join(r['clinical_flags'])}")
    else:
        print(f"❌ Error: {response.status_code} - {response.json()}")
    print()

def run_all_tests():
    """Run comprehensive API tests"""
    print("🧪 CKD Prediction API - Comprehensive Test Suite")
    print("=" * 50)
    
    try:
        # Test basic functionality
        test_health()
        test_model_info()
        
        # Test prediction endpoints
        test_single_prediction()
        test_batch_json()
        
        # Test file upload
        csv_path = create_test_csv()
        test_batch_csv_upload(csv_path)
        
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"🧹 Cleaned up test file: {csv_path}")
        
        print("✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running on http://localhost:5000")
        print("   Start the server with: python ckd_prediction_api.py")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    run_all_tests()