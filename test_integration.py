#!/usr/bin/env python3
"""Quick integration test for HealthCare Analysis API"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_health_check():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ Health check: {response.status_code}")
        return response.status_code == 200
    except:
        print("✗ Backend not running")
        return False

def test_upload_health_data():
    """Test health data upload"""
    data = {
        "steps": 8500,
        "sleepHours": 7.5,
        "heartRate": 72,
        "calories": 2200,
        "waterIntake": 2.1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/uploadHealthData", json=data)
        print(f"✓ Health data upload: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Data ID: {result.get('data_id')}")
            return result.get('data_id')
    except Exception as e:
        print(f"✗ Health data upload failed: {e}")
    return None

def test_chat_ai():
    """Test AI chat endpoint"""
    try:
        response = requests.post(f"{BASE_URL}/api/chat", json={"message": "How is my sleep?"})
        print(f"✓ AI Chat: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Response: {result.get('response')[:50]}...")
    except Exception as e:
        print(f"✗ AI Chat failed: {e}")

def test_health_summary():
    """Test health summary endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/healthSummary")
        print(f"✓ Health Summary: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Health Score: {result.get('healthScore', 'N/A')}")
    except Exception as e:
        print(f"✗ Health Summary failed: {e}")

if __name__ == "__main__":
    print("Testing HealthCare Analysis Integration...")
    print("=" * 50)
    
    if test_health_check():
        data_id = test_upload_health_data()
        time.sleep(1)
        test_chat_ai()
        test_health_summary()
        print("\n✓ All integrations working!")
    else:
        print("\n✗ Start backend first: python health-backend-java/health-backend/app/integrated_main.py")