import pandas as pd
from datetime import datetime, timedelta
from ai_reasoning_engine import AIReasoningEngine

def create_sample_data():
    """Generate sample health data for testing"""
    dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
    
    # Sample sleep data
    sleep_data = pd.DataFrame({
        'date': dates,
        'duration_hours': [6.5, 7.2, 5.8, 6.0, 7.5, 8.1, 6.2, 7.0, 5.5, 6.8],
        'bedtime': ['23:30', '22:45', '00:15', '23:50', '22:30', '22:00', '23:45', '22:30', '00:30', '23:15']
    })
    
    # Sample heart rate data
    hr_data = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(days=i, hours=h) for i in range(5) for h in [9, 15, 19]],
        'heart_rate': [72, 85, 78, 70, 95, 82, 68, 88, 76, 74, 110, 85, 69, 92, 79]
    })
    
    # Sample hydration data
    hydration_data = pd.DataFrame({
        'date': dates,
        'water_ml': [1800, 2200, 1500, 1600, 2100, 2400, 1700, 2000, 1400, 1900]
    })
    
    return {
        'sleep': sleep_data,
        'heart_rate': hr_data,
        'hydration': hydration_data
    }

def main():
    # Initialize the AI reasoning engine
    ai_engine = AIReasoningEngine()
    
    # Create sample data
    health_data = create_sample_data()
    
    # Analyze the data
    results = ai_engine.analyze_health_data(health_data)
    
    # Display results
    print("🤖 AI Health Analysis Report")
    print("=" * 40)
    print(f"Analysis Period: {results['context_period']}")
    print(f"Generated: {results['analysis_date'][:19]}")
    print()
    
    print("📊 Health Insights:")
    for insight in results['insights']:
        severity_emoji = {'good': '✅', 'warning': '⚠️', 'caution': '🔶', 'info': 'ℹ️'}
        emoji = severity_emoji.get(insight['severity'], '📋')
        print(f"{emoji} {insight['message']}")
    
    print("\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"• {rec}")

if __name__ == "__main__":
    main()