# Health-Care-Analysis

A comprehensive health data analysis platform with AI-powered insights, data processing backend, and interactive frontend dashboard.

## Architecture
- **AI Engine** (Python) - Analyzes health metrics and generates insights
- **Backend API** (FastAPI) - Processes CSV uploads and serves data
- **Frontend** (React/Vite) - Interactive dashboard with charts and visualizations

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm

### Setup
```bash
# Run complete setup
chmod +x setup_all.sh
./setup_all.sh
```

### Running the Application

1. **Start Backend** (Terminal 1):
```bash
cd health-backend-java/health-backend
source venv/bin/activate
uvicorn app.main:app --reload
```
Backend runs on: http://localhost:8000

2. **Start Frontend** (Terminal 2):
```bash
cd FrontEnd
npm run dev
```
Frontend runs on: http://localhost:5173

3. **Test AI Engine**:
```bash
cd AI
source venv/bin/activate
python sample_usage.py
```

## Data Format
Upload CSV files with columns:
- `user_id`: String identifier
- `date`: ISO date (YYYY-MM-DD)
- `metric`: steps, heart_rate, sleep, water
- `value`: Numeric value
- `type` (optional): Context info
- `notes` (optional): Free text

## API Endpoints
- `POST /upload` - Upload health data CSV
- `GET /data/{data_id}/summary` - Get analysis summary
- `GET /data/{data_id}/trends` - Get trend data
- `GET /data/{data_id}/anomalies` - Get anomaly detection

## Features
- 📊 Interactive health dashboards
- 🤖 AI-powered health insights
- 📈 Trend analysis and forecasting
- ⚠️ Anomaly detection
- 💡 Personalized recommendations
