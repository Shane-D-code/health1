# Health Data Backend API

Backend service for processing health metrics (steps, heart rate, sleep, water).

## Data Format (CSV)
Columns required:
- `user_id`: String identifier (e.g., "u123")
- `date`: ISO date or datetime (e.g., "2025-11-01" or "2025-11-01 08:00:00")
- `metric`: One of `steps`, `heart_rate`, `sleep`, `water`
- `value`: Numeric value
- `type` (optional): Context (e.g., "resting", "walking")
- `notes` (optional): Free text

### Units
- **steps**: count
- **heart_rate**: bpm
- **sleep**: hours
- **water**: ml

## API Endpoints

### `POST /upload`
Upload a CSV file.
**Response:**
```json
{
  "status": "ok",
  "data_id": "uuid...",
  "summary": { ... }
}
```

### `GET /data/{data_id}/summary`
Returns aggregated stats, trends, and anomalies.

### `GET /data/{data_id}/trends`
Returns time-series data for charting.

### `GET /data/{data_id}/anomalies`
Returns list of detected anomalies.

## Running Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run server:
   ```bash
   uvicorn app.main:app --reload
   ```
