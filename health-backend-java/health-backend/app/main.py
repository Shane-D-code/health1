from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uuid
from typing import Dict
from .processor import get_trends_and_insights
from .models import UploadResponse, SummaryResponse
import sys
import os

# Add AI directory to path
ai_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'AI')
sys.path.append(ai_path)

try:
    from ai_reasoning_engine import AIReasoningEngine
except ImportError:
    AIReasoningEngine = None

app = FastAPI(title="Health Data Backend", version="1.0")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
DATA_STORE: Dict[str, Dict] = {}

@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Process
        results = get_trends_and_insights(df)
        
        # Generate ID and store
        data_id = str(uuid.uuid4())
        
        # Extract user_id if possible, else default
        user_id = "unknown"
        if 'user_id' in df.columns:
            user_id = str(df['user_id'].iloc[0])
            
        stored_data = {
            "user_id": user_id,
            "raw_filename": file.filename,
            "processed": results
        }
        DATA_STORE[data_id] = stored_data
        
        return {
            "status": "ok",
            "data_id": data_id,
            "summary": results["summary"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/{data_id}/summary")
async def get_summary(data_id: str):
    if data_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Data ID not found")
    
    data = DATA_STORE[data_id]
    processed = data["processed"]
    
    return {
        "user_id": data["user_id"],
        "summary": processed["summary"],
        "trends": processed["trends"],
        "anomalies": processed["anomalies"],
        "timeseries": processed["timeseries"],
        "data_id": data_id
    }

@app.get("/data/{data_id}/trends")
async def get_trends(data_id: str):
    if data_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Data ID not found")
        
    return DATA_STORE[data_id]["processed"]["timeseries"]

@app.get("/data/{data_id}/anomalies")
async def get_anomalies(data_id: str):
    if data_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Data ID not found")
        
    return DATA_STORE[data_id]["processed"]["anomalies"]

@app.get("/health")
async def health_check():
    return {"status": "ok"}
