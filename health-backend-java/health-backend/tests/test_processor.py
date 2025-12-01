import requests
import pandas as pd
import io

# Mock data
csv_content = """user_id,date,metric,type,value,notes
u123,2025-11-01,steps,walk,1000,
u123,2025-11-01,heart_rate,resting,60,
u123,2025-11-01,sleep,night,8,
u123,2025-11-02,steps,walk,2000,
u123,2025-11-02,heart_rate,resting,65,
u123,2025-11-02,sleep,night,7,
u123,2025-11-03,steps,walk,500,
u123,2025-11-03,heart_rate,resting,110,fever
u123,2025-11-03,sleep,night,4,
"""

def test_processing():
    # We can test the processor logic directly without running the server
    from app.processor import get_trends_and_insights
    
    df = pd.read_csv(io.StringIO(csv_content))
    results = get_trends_and_insights(df)
    
    import json
    with open('processed_sample.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Summary:", results['summary'])
    print("Anomalies:", results['anomalies'])
    
    # Check for expected anomalies
    anomalies = results['anomalies']
    hr_anomaly = next((a for a in anomalies if a['metric'] == 'heart_rate'), None)
    assert hr_anomaly is not None, "Should detect heart rate anomaly > 100"
    print("Test Passed: Heart rate anomaly detected.")

if __name__ == "__main__":
    test_processing()
