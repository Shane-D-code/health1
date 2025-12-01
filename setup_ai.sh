#!/bin/bash
cd AI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "AI Engine setup complete. To test:"
echo "cd AI"
echo "source venv/bin/activate"
echo "python sample_usage.py"