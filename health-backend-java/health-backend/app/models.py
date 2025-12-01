from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Anomaly(BaseModel):
    date: str
    metric: str
    value: float
    reason: str
    z_score: Optional[float] = None

class Trend(BaseModel):
    metric: str
    trend: str
    change_percent: float
    note: Optional[str] = None

class SummaryResponse(BaseModel):
    user_id: str
    summary: Dict[str, float]
    trends: List[Trend]
    anomalies: List[Anomaly]
    data_id: str

class UploadResponse(BaseModel):
    status: str
    data_id: str
    summary: Dict[str, float]
