import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class AdvancedPatternAnalyzer:
    def __init__(self):
        self.pattern_thresholds = {
            'sleep_irregularity': 1.5,  # hours variance
            'hr_spike_threshold': 1.3,  # multiplier of baseline
            'trend_significance': 0.15  # 15% change
        }
    
    def detect_sleep_patterns(self, sleep_data: pd.DataFrame) -> Dict[str, any]:
        """Advanced sleep pattern detection"""
        if sleep_data.empty or len(sleep_data) < 3:
            return {'patterns': [], 'trends': 'insufficient_data'}
        
        patterns = []
        durations = sleep_data['duration_hours'].values
        
        # Detect weekend vs weekday patterns
        if 'date' in sleep_data.columns:
            sleep_data['weekday'] = pd.to_datetime(sleep_data['date']).dt.dayofweek
            weekday_sleep = sleep_data[sleep_data['weekday'] < 5]['duration_hours'].mean()
            weekend_sleep = sleep_data[sleep_data['weekday'] >= 5]['duration_hours'].mean()
            
            if abs(weekend_sleep - weekday_sleep) > 1:
                patterns.append(f"Weekend sleep differs by {abs(weekend_sleep - weekday_sleep):.1f} hours from weekdays")
        
        # Detect declining trend
        if len(durations) >= 5:
            recent_avg = np.mean(durations[-3:])
            earlier_avg = np.mean(durations[:3])
            change = (recent_avg - earlier_avg) / earlier_avg
            
            if change < -self.pattern_thresholds['trend_significance']:
                patterns.append(f"Sleep duration declining by {abs(change)*100:.0f}% over the period")
            elif change > self.pattern_thresholds['trend_significance']:
                patterns.append(f"Sleep duration improving by {change*100:.0f}% over the period")
        
        # Detect consistency issues
        variance = np.var(durations)
        if variance > self.pattern_thresholds['sleep_irregularity']:
            patterns.append("Highly irregular sleep schedule detected")
        
        return {
            'patterns': patterns,
            'avg_duration': np.mean(durations),
            'consistency_score': max(0, 100 - (variance * 20))  # 0-100 scale
        }
    
    def detect_heart_rate_anomalies(self, hr_data: pd.DataFrame) -> Dict[str, any]:
        """Advanced heart rate anomaly detection"""
        if hr_data.empty:
            return {'anomalies': [], 'risk_level': 'unknown'}
        
        anomalies = []
        hr_values = hr_data['heart_rate'].values
        
        # Statistical anomaly detection
        q75, q25 = np.percentile(hr_values, [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + (1.5 * iqr)
        
        outliers = hr_data[hr_data['heart_rate'] > upper_bound]
        if not outliers.empty:
            max_spike = outliers['heart_rate'].max()
            anomalies.append(f"Heart rate spike detected: {max_spike} bpm")
        
        # Time-based pattern detection
        if 'timestamp' in hr_data.columns:
            hr_data['hour'] = pd.to_datetime(hr_data['timestamp']).dt.hour
            
            # Check for unusual nighttime spikes (10 PM - 6 AM)
            night_hours = hr_data[hr_data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5, 6])]
            if not night_hours.empty:
                night_avg = night_hours['heart_rate'].mean()
                day_avg = hr_data[~hr_data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5, 6])]['heart_rate'].mean()
                
                if night_avg > day_avg * 1.1:
                    anomalies.append("Elevated nighttime heart rate detected - possible sleep issues")
        
        # Risk assessment
        avg_hr = np.mean(hr_values)
        max_hr = np.max(hr_values)
        
        if avg_hr > 100 or max_hr > 150:
            risk_level = 'high'
        elif avg_hr > 85 or max_hr > 120:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'anomalies': anomalies,
            'risk_level': risk_level,
            'avg_heart_rate': avg_hr,
            'max_heart_rate': max_hr
        }
    
    def analyze_correlations(self, health_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Detect correlations between different health metrics"""
        correlations = []
        
        # Check sleep-heart rate correlation
        if 'sleep' in health_data and 'heart_rate' in health_data:
            sleep_df = health_data['sleep']
            hr_df = health_data['heart_rate']
            
            if not sleep_df.empty and not hr_df.empty:
                # Simple correlation check
                avg_sleep = sleep_df['duration_hours'].mean()
                avg_hr = hr_df['heart_rate'].mean()
                
                if avg_sleep < 7 and avg_hr > 80:
                    correlations.append("Poor sleep may be contributing to elevated heart rate")
        
        return correlations