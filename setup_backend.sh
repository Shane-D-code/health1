#!/bin/bash
cd health-backend-java/health-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "Backend setup complete. To start the server:"
echo "cd health-backend-java/health-backend"
echo "source venv/bin/activate"
echo "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"