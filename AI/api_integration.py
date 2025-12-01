from flask import Flask, request, jsonify
from ai_reasoning_engine import AIReasoningEngine
import pandas as pd
from datetime import datetime

app = Flask(__name__)
ai_engine = AIReasoningEngine()

@app.route('/analyze', methods=['POST'])
def analyze_health_data():
    """API endpoint for health data analysis"""
    try:
        data = request.json
        
        # Convert JSON data to DataFrames
        health_data = {}
        for data_type, records in data.items():
            if records:
                health_data[data_type] = pd.DataFrame(records)
        
        # Perform analysis
        results = ai_engine.analyze_health_data(health_data)
        
        return jsonify({
            'status': 'success',
            'data': results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Reasoning Engine',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)