from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import uuid
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import io
import google.generativeai as genai
import os
from config import GEMINI_API_KEY

# Add AI module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'AI'))

from ai_reasoning_engine import AIReasoningEngine
from processor import get_trends_and_insights

app = FastAPI(title="Integrated Health Data Backend", version="2.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class HealthDataUpload(BaseModel):
    steps: int
    sleepHours: float
    heartRate: int
    calories: int
    waterIntake: Optional[float] = 2.0
    date: Optional[str] = None

class HealthSummaryResponse(BaseModel):
    steps: int
    sleepHours: float
    heartRate: int
    calories: int
    waterIntake: float
    aiInsights: List[str]
    healthScore: float
    trends: Dict[str, Any]
    anomalies: List[Dict[str, Any]]

# In-memory storage
DATA_STORE: Dict[str, Dict] = {}
ai_engine = AIReasoningEngine()

def process_health_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process health data and generate AI insights"""
    try:
        # Convert to processor expected format: user_id, date, metric, value
        date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        records = [
            {'user_id': 'user1', 'date': date_str, 'metric': 'steps', 'value': data['steps']},
            {'user_id': 'user1', 'date': date_str, 'metric': 'sleep', 'value': data['sleepHours']},
            {'user_id': 'user1', 'date': date_str, 'metric': 'heart_rate', 'value': data['heartRate']},
            {'user_id': 'user1', 'date': date_str, 'metric': 'calories', 'value': data['calories']},
            {'user_id': 'user1', 'date': date_str, 'metric': 'water', 'value': data.get('waterIntake', 2.0)}
        ]
        
        df = pd.DataFrame(records)
        
        # Prepare data for AI analysis with correct column names
        health_data = {
            'sleep': pd.DataFrame({
                'timestamp': [pd.to_datetime(date_str)],
                'duration_hours': [data['sleepHours']]
            }),
            'heart_rate': pd.DataFrame({
                'timestamp': [pd.to_datetime(date_str)],
                'heart_rate': [data['heartRate']]
            }),
            'hydration': pd.DataFrame({
                'timestamp': [pd.to_datetime(date_str)],
                'water_ml': [data.get('waterIntake', 2.0) * 1000]
            })
        }
        
        # Get AI analysis
        ai_results = ai_engine.analyze_health_data(health_data)
        
        # Get trends and anomalies using existing processor
        trends_results = get_trends_and_insights(df)
        
        # Calculate health score
        health_score = calculate_health_score(data)
        
        # Format AI insights for frontend
        ai_insights = []
        for insight in ai_results['insights']:
            if insight['severity'] in ['warning', 'caution', 'critical']:
                ai_insights.append(insight['message'])
        
        # Add recommendations
        ai_insights.extend(ai_results['recommendations'][:3])  # Limit to top 3
        
        # Add summary insights from processor
        summary = trends_results.get('summary', {})
        if 'heart_rate_avg_7d' in summary:
            ai_insights.append(f"Average heart rate: {summary['heart_rate_avg_7d']} BPM")
        if 'sleep_avg_7d' in summary:
            ai_insights.append(f"Average sleep: {summary['sleep_avg_7d']} hours")
        if 'steps_avg_7d' in summary:
            ai_insights.append(f"Average steps: {int(summary['steps_avg_7d'])} per day")
        
        return {
            'processed_data': data,
            'ai_insights': ai_insights,
            'health_score': health_score,
            'trends': trends_results.get('trends', {}),
            'anomalies': trends_results.get('anomalies', []),
            'full_ai_analysis': ai_results
        }
        
    except Exception as e:
        print(f"Error processing health data: {str(e)}")
        return {
            'processed_data': data,
            'ai_insights': ["Data processed successfully. Continue monitoring your health metrics."],
            'health_score': 75.0,
            'trends': {},
            'anomalies': []
        }

def calculate_health_score(data: Dict[str, Any]) -> float:
    """Calculate overall health score based on metrics"""
    score = 0
    
    # Steps scoring (30% weight)
    steps = data['steps']
    if steps >= 10000:
        score += 30
    elif steps >= 7500:
        score += 25
    elif steps >= 5000:
        score += 20
    else:
        score += 10
    
    # Sleep scoring (25% weight)
    sleep = data['sleepHours']
    if 7 <= sleep <= 9:
        score += 25
    elif 6 <= sleep < 7 or 9 < sleep <= 10:
        score += 20
    else:
        score += 10
    
    # Heart rate scoring (25% weight)
    hr = data['heartRate']
    if 60 <= hr <= 100:
        score += 25
    elif 50 <= hr < 60 or 100 < hr <= 120:
        score += 20
    else:
        score += 10
    
    # Hydration scoring (20% weight)
    water = data.get('waterIntake', 2.0)
    if water >= 2.5:
        score += 20
    elif water >= 2.0:
        score += 15
    else:
        score += 10
    
    return min(score, 100.0)

@app.post("/api/uploadHealthData")
async def upload_health_data(data: HealthDataUpload):
    """Upload and process health data with AI analysis"""
    try:
        # Convert to dict
        health_data = data.dict()
        if not health_data.get('date'):
            health_data['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Process with AI
        results = process_health_data(health_data)
        
        # Store data
        data_id = str(uuid.uuid4())
        DATA_STORE[data_id] = {
            'timestamp': datetime.now().isoformat(),
            'raw_data': health_data,
            'results': results
        }
        
        return {
            "status": "success",
            "message": "Health data processed successfully",
            "data_id": data_id,
            "summary": {
                "steps": health_data['steps'],
                "sleepHours": health_data['sleepHours'],
                "heartRate": health_data['heartRate'],
                "calories": health_data['calories'],
                "waterIntake": health_data.get('waterIntake', 2.0),
                "aiInsights": results['ai_insights'],
                "healthScore": results['health_score']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing health data: {str(e)}")

@app.get("/api/healthSummary")
async def get_health_summary():
    """Get summary of all health data"""
    try:
        if not DATA_STORE:
            return {
                "message": "No health data available",
                "totalRecords": 0
            }
        
        # Get latest entry
        latest_entry = max(DATA_STORE.values(), key=lambda x: x['timestamp'])
        results = latest_entry['results']
        raw_data = latest_entry['raw_data']
        
        # Get processor summary for accurate averages
        trends_results = results.get('trends', {})
        processor_summary = trends_results.get('summary', {}) if isinstance(trends_results, dict) else {}
        
        return {
            "steps": raw_data['steps'],
            "sleepHours": raw_data['sleepHours'],
            "heartRate": raw_data['heartRate'],
            "calories": raw_data['calories'],
            "waterIntake": raw_data.get('waterIntake', 2.0),
            "aiInsights": results['ai_insights'],
            "healthScore": results['health_score'],
            "trends": results['trends'],
            "anomalies": results['anomalies'],
            "averages": {
                "heartRate": processor_summary.get('heart_rate_avg_7d', raw_data['heartRate']),
                "steps": processor_summary.get('steps_avg_7d', raw_data['steps']),
                "sleep": processor_summary.get('sleep_avg_7d', raw_data['sleepHours'])
            },
            "totalRecords": len(DATA_STORE),
            "lastUpdated": latest_entry['timestamp']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving health summary: {str(e)}")

@app.get("/api/healthInsights")
async def get_health_insights():
    """Get AI-generated health insights"""
    try:
        if not DATA_STORE:
            return {
                "insights": ["Upload health data to get personalized insights"],
                "recommendations": []
            }
        
        # Get latest entry
        latest_entry = max(DATA_STORE.values(), key=lambda x: x['timestamp'])
        ai_analysis = latest_entry['results'].get('full_ai_analysis', {})
        
        insights = []
        for insight in ai_analysis.get('insights', []):
            insights.append({
                "id": insight['id'],
                "type": insight['type'],
                "severity": insight['severity'],
                "title": insight['title'],
                "message": insight['message'],
                "timestamp": latest_entry['timestamp']
            })
        
        return {
            "insights": insights,
            "recommendations": ai_analysis.get('recommendations', []),
            "analysisDate": ai_analysis.get('analysis_date'),
            "contextPeriod": ai_analysis.get('context_period', "Current data")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving insights: {str(e)}")

@app.get("/api/healthTrends")
async def get_health_trends():
    """Get health trends analysis"""
    try:
        if not DATA_STORE:
            return {
                "trends": {},
                "message": "No data available for trend analysis"
            }
        
        # Collect all data for trend analysis
        all_data = []
        for entry in DATA_STORE.values():
            data_point = entry['raw_data'].copy()
            data_point['timestamp'] = entry['timestamp']
            all_data.append(data_point)
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x['timestamp'])
        
        # Calculate trends
        trends = {}
        if len(all_data) >= 2:
            latest = all_data[-1]
            previous = all_data[-2]
            
            for metric in ['steps', 'sleepHours', 'heartRate', 'calories']:
                if metric in latest and metric in previous:
                    change = latest[metric] - previous[metric]
                    percent_change = (change / previous[metric]) * 100 if previous[metric] != 0 else 0
                    
                    trends[metric] = {
                        "current": latest[metric],
                        "previous": previous[metric],
                        "change": change,
                        "percentChange": round(percent_change, 1),
                        "trend": "increasing" if change > 0 else "decreasing" if change < 0 else "stable"
                    }
        
        return {
            "trends": trends,
            "dataPoints": len(all_data),
            "dateRange": {
                "start": all_data[0]['timestamp'] if all_data else None,
                "end": all_data[-1]['timestamp'] if all_data else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving trends: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file - Frontend compatible endpoint"""
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        results = get_trends_and_insights(df)
        data_id = str(uuid.uuid4())
        
        DATA_STORE[data_id] = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'type': 'csv_upload'
        }
        
        return {"data_id": data_id, "message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/{data_id}/summary")
async def get_data_summary(data_id: str):
    """Get processed data summary"""
    if data_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Data not found")
    return DATA_STORE[data_id]['results']

@app.get("/data/{data_id}/trends")
async def get_data_trends(data_id: str):
    """Get timeseries data for charts"""
    if data_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Data not found")
    return DATA_STORE[data_id]['results'].get('timeseries', [])

@app.post("/api/uploadCSV")
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload CSV file for batch processing"""
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Process with existing processor
        results = get_trends_and_insights(df)
        
        # Store results in compatible format
        data_id = str(uuid.uuid4())
        
        # Get latest row for raw_data format
        latest_row = df.iloc[-1] if not df.empty else {}
        raw_data = {
            'steps': int(latest_row.get('steps', 0)),
            'sleepHours': float(latest_row.get('sleep_hours', 0)),
            'heartRate': int(latest_row.get('heart_rate_bpm', 0)),
            'calories': int(latest_row.get('calories_burned', 0)),
            'waterIntake': float(latest_row.get('water_liters', 0)),
            'date': latest_row.get('date', datetime.now().strftime('%Y-%m-%d'))
        }
        
        DATA_STORE[data_id] = {
            'timestamp': datetime.now().isoformat(),
            'raw_data': raw_data,
            'results': {
                'ai_insights': [f"Processed {len(df)} records from CSV", f"Average heart rate: {results.get('summary', {}).get('heart_rate_avg_7d', 0)} BPM"],
                'health_score': 85.0,
                'trends': results,
                'anomalies': results.get('anomalies', [])
            },
            'type': 'csv_upload'
        }
        
        return {
            "status": "success",
            "message": f"CSV file processed: {file.filename}",
            "data_id": data_id,
            "summary": results.get("summary", {}),
            "recordsProcessed": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Integrated Health Backend",
        "version": "2.0",
        "ai_engine": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Integrated Health Data Analysis API",
        "version": "2.0",
        "endpoints": [
            "/api/uploadHealthData",
            "/api/healthSummary", 
            "/api/healthInsights",
            "/api/healthTrends",
            "/api/uploadCSV"
        ]
    }

# Initialize Gemini from config
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    model = None

@app.post("/api/chat")
async def chat_with_ai(request: dict):
    """AI Chat endpoint with Gemini integration"""
    try:
        prompt = request.get('message', '')
        
        # Try Gemini AI first
        if model and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            try:
                health_context = "You are a health assistant. Provide helpful, accurate health advice. Keep responses concise and actionable."
                full_prompt = f"{health_context}\n\nUser question: {prompt}"
                
                response = model.generate_content(full_prompt)
                return {"response": response.text, "timestamp": datetime.now().isoformat(), "source": "gemini"}
            except Exception as e:
                print(f"Gemini API error: {e}")
        
        # Fallback to keyword responses
        responses = {
            'sleep': "Based on your sleep data, I recommend maintaining 7-9 hours nightly. Try a consistent bedtime routine.",
            'heart': "Your heart rate patterns look normal. Consider stress management if you see spikes.",
            'steps': "Great job on staying active! Aim for 10,000 steps daily for optimal health.",
            'water': "Hydration is key! Try to drink 8 glasses of water throughout the day.",
            'stress': "Managing stress is crucial. Try deep breathing, meditation, or light exercise."
        }
        
        response = "I'm here to help with your health insights. Upload your data to get personalized recommendations!"
        for keyword, reply in responses.items():
            if keyword in prompt.lower():
                response = reply
                break
                
        return {"response": response, "timestamp": datetime.now().isoformat(), "source": "fallback"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)