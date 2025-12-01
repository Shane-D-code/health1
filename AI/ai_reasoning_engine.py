import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict
import numpy as np

from pattern_analyzer import AdvancedPatternAnalyzer


class Insight(TypedDict, total=False):
    """
    Standard shape for insights returned by the AI engines.

    This is the schema your frontend / report generator can rely on.
    """

    id: str
    type: str  # e.g. 'sleep', 'heart_rate', 'hydration', 'activity'
    severity: str  # 'info' | 'good' | 'caution' | 'warning' | 'critical'
    title: str
    message: str
    suggested_actions: List[str]
    # Optional raw data to help debugging / deeper views
    metadata: Dict[str, Any]

class HealthInsightGenerator:
    def __init__(self):
        self.insight_templates = {
            'sleep': {
                'irregular': "Your sleep is irregular this week.",
                'insufficient': "You're getting less than 7 hours of sleep on average.",
                'good': "Your sleep pattern looks healthy this week."
            },
            'heart_rate': {
                'spike': "Heart rate peaked unusually at {time} — possible stress?",
                'elevated': "Your resting heart rate is elevated compared to your baseline.",
                'normal': "Your heart rate patterns look normal."
            },
            'hydration': {
                'low': "Hydration levels below recommended for {days} days.",
                'good': "You're maintaining good hydration levels."
            },
            'activity': {
                'low': "Physical activity has decreased by {percent}% this week.",
                'high': "Great job! You've increased activity by {percent}% this week."
            }
        }
    
    def analyze_sleep_patterns(self, sleep_data: pd.DataFrame) -> Insight:
        """Analyze sleep duration and regularity"""
        if sleep_data.empty:
            return {
                'id': 'sleep:no_data',
                'type': 'sleep',
                'severity': 'info',
                'title': 'No sleep data',
                'message': 'No sleep data available for the selected period.',
                'suggested_actions': [],
                'metadata': {},
            }
        
        avg_sleep = sleep_data['duration_hours'].mean()
        sleep_variance = sleep_data['duration_hours'].var()
        
        if avg_sleep < 7:
            return {
                'id': 'sleep:insufficient',
                'type': 'sleep',
                'severity': 'warning',
                'title': 'Sleep is below recommended levels',
                'message': self.insight_templates['sleep']['insufficient'],
                'suggested_actions': [],
                'metadata': {
                    'avg_hours': float(avg_sleep),
                    'variance': float(sleep_variance),
                },
            }
        elif sleep_variance > 2:
            return {
                'id': 'sleep:irregular',
                'type': 'sleep',
                'severity': 'caution',
                'title': 'Sleep pattern is irregular',
                'message': self.insight_templates['sleep']['irregular'],
                'suggested_actions': [],
                'metadata': {
                    'avg_hours': float(avg_sleep),
                    'variance': float(sleep_variance),
                },
            }
        else:
            return {
                'id': 'sleep:good',
                'type': 'sleep',
                'severity': 'good',
                'title': 'Sleep looks healthy',
                'message': self.insight_templates['sleep']['good'],
                'suggested_actions': [],
                'metadata': {
                    'avg_hours': float(avg_sleep),
                    'variance': float(sleep_variance),
                },
            }
    
    def analyze_heart_rate(self, hr_data: pd.DataFrame) -> Insight:
        """Detect heart rate anomalies"""
        if hr_data.empty:
            return {
                'id': 'heart_rate:no_data',
                'type': 'heart_rate',
                'severity': 'info',
                'title': 'No heart rate data',
                'message': 'No heart rate data available for the selected period.',
                'suggested_actions': [],
                'metadata': {},
            }
        
        baseline = hr_data['heart_rate'].quantile(0.5)
        spikes = hr_data[hr_data['heart_rate'] > baseline * 1.3]
        
        if not spikes.empty:
            spike_time = spikes.iloc[0]['timestamp'].strftime('%I%p')
            message = self.insight_templates['heart_rate']['spike'].format(time=spike_time)
            return {
                'id': 'heart_rate:spike',
                'type': 'heart_rate',
                'severity': 'warning',
                'title': 'Heart rate spike detected',
                'message': message,
                'suggested_actions': [],
                'metadata': {
                    'baseline': float(baseline),
                    'spike_count': int(len(spikes)),
                },
            }
        
        return {
            'id': 'heart_rate:normal',
            'type': 'heart_rate',
            'severity': 'good',
            'title': 'Heart rate looks normal',
            'message': self.insight_templates['heart_rate']['normal'],
            'suggested_actions': [],
            'metadata': {
                'baseline': float(baseline),
                'spike_count': int(len(spikes)),
            },
        }
    
    def analyze_hydration(self, hydration_data: pd.DataFrame) -> Insight:
        """Check hydration levels"""
        if hydration_data.empty:
            return {
                'id': 'hydration:no_data',
                'type': 'hydration',
                'severity': 'info',
                'title': 'No hydration data',
                'message': 'No hydration data available for the selected period.',
                'suggested_actions': [],
                'metadata': {},
            }
        
        recommended_daily = 2000  # ml
        low_days = (hydration_data['water_ml'] < recommended_daily).sum()
        
        if low_days >= 4:
            message = self.insight_templates['hydration']['low'].format(days=low_days)
            return {
                'id': 'hydration:low',
                'type': 'hydration',
                'severity': 'warning',
                'title': 'Hydration below recommended levels',
                'message': message,
                'suggested_actions': [],
                'metadata': {
                    'low_days': int(low_days),
                    'recommended_daily_ml': int(recommended_daily),
                },
            }
        
        return {
            'id': 'hydration:good',
            'type': 'hydration',
            'severity': 'good',
            'title': 'Hydration looks good',
            'message': self.insight_templates['hydration']['good'],
            'suggested_actions': [],
            'metadata': {
                'low_days': int(low_days),
                'recommended_daily_ml': int(recommended_daily),
            },
        }

class HealthRecommendationEngine:
    def __init__(self):
        self.recommendations = {
            'sleep': [
                "Try to maintain a consistent bedtime routine",
                "Avoid screens 1 hour before bed",
                "Keep your bedroom cool and dark"
            ],
            'heart_rate': [
                "Practice deep breathing exercises",
                "Consider meditation or yoga",
                "Monitor stress levels throughout the day"
            ],
            'hydration': [
                "Set hourly water reminders",
                "Keep a water bottle nearby",
                "Eat water-rich foods like fruits"
            ],
            'activity': [
                "Take short walks every 2 hours",
                "Use stairs instead of elevators",
                "Try 10-minute morning stretches"
            ]
        }
    
    def get_recommendations(self, insight_type: str, severity: str) -> List[str]:
        """Generate actionable recommendations based on insights"""
        if severity in ['warning', 'caution']:
            return self.recommendations.get(insight_type, [])
        return []

class ContextualReasoning:
    def __init__(self):
        self.context_window = 7  # days
    
    def get_weekly_context(self, data: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Filter data to last 7 days for contextual analysis"""
        if data.empty:
            return data
        
        # Check if date column exists, if not use timestamp or return all data
        if date_col not in data.columns:
            if 'timestamp' in data.columns:
                date_col = 'timestamp'
            else:
                return data  # Return all data if no date column found
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.context_window)
        
        data_copy = data.copy()
        data_copy[date_col] = pd.to_datetime(data_copy[date_col])
        return data_copy[data_copy[date_col] >= start_date]
    
    def compare_with_baseline(self, current_data: pd.DataFrame, historical_data: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Compare current week with historical baseline"""
        if current_data.empty or historical_data.empty:
            return {'change_percent': 0, 'trend': 'stable'}
        
        current_avg = current_data[metric].mean()
        historical_avg = historical_data[metric].mean()
        
        change_percent = ((current_avg - historical_avg) / historical_avg) * 100
        trend = 'improving' if change_percent > 5 else 'declining' if change_percent < -5 else 'stable'
        
        return {'change_percent': change_percent, 'trend': trend}

class AIReasoningEngine:
    def __init__(self):
        self.insight_generator = HealthInsightGenerator()
        self.recommendation_engine = HealthRecommendationEngine()
        self.contextual_reasoning = ContextualReasoning()
        # Advanced anomaly and pattern detection module
        self.pattern_analyzer = AdvancedPatternAnalyzer()
    
    def analyze_health_data(self, health_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Main analysis function that processes all health data"""
        insights: List[Insight] = []
        recommendations: List[str] = []
        
        # Get weekly context for all data
        weekly_data = {}
        for data_type, df in health_data.items():
            weekly_data[data_type] = self.contextual_reasoning.get_weekly_context(df)
        
        # Generate core insights for each data type
        if 'sleep' in weekly_data:
            sleep_insight = self.insight_generator.analyze_sleep_patterns(
                weekly_data['sleep']
            )
            insights.append(sleep_insight)
            recommendations.extend(self.recommendation_engine.get_recommendations('sleep', sleep_insight['severity']))
        
        if 'heart_rate' in weekly_data:
            hr_insight = self.insight_generator.analyze_heart_rate(
                weekly_data['heart_rate']
            )
            insights.append(hr_insight)
            recommendations.extend(self.recommendation_engine.get_recommendations('heart_rate', hr_insight['severity']))
        
        if 'hydration' in weekly_data:
            hydration_insight = self.insight_generator.analyze_hydration(weekly_data['hydration'])
            insights.append(hydration_insight)
            recommendations.extend(self.recommendation_engine.get_recommendations('hydration', hydration_insight['severity']))

        # --- Advanced anomaly & pattern insights (from AdvancedPatternAnalyzer) ---
        # These provide richer, more detailed anomalies that the UI can surface separately.
        if 'sleep' in weekly_data and not weekly_data['sleep'].empty:
            sleep_patterns = self.pattern_analyzer.detect_sleep_patterns(
                weekly_data['sleep']
            )
            if sleep_patterns.get('patterns'):
                insights.append(
                    Insight(
                        id='sleep:advanced_patterns',
                        type='sleep',
                        severity='caution',
                        title='Detailed sleep patterns detected',
                        message='; '.join(sleep_patterns['patterns']),
                        suggested_actions=[],
                        metadata={
                            'avg_duration': float(sleep_patterns['avg_duration']),
                            'consistency_score': float(
                                sleep_patterns['consistency_score']
                            ),
                        },
                    )
                )
        
        if 'heart_rate' in weekly_data and not weekly_data['heart_rate'].empty:
            hr_anomalies = self.pattern_analyzer.detect_heart_rate_anomalies(
                weekly_data['heart_rate']
            )
            if hr_anomalies.get('anomalies'):
                severity_map = {
                    'high': 'warning',
                    'moderate': 'caution',
                    'low': 'info',
                    'unknown': 'info',
                }
                severity = severity_map.get(hr_anomalies.get('risk_level', 'low'), 'info')
                insights.append(
                    Insight(
                        id='heart_rate:advanced_anomalies',
                        type='heart_rate',
                        severity=severity,
                        title='Detailed heart rate anomalies detected',
                        message='; '.join(hr_anomalies['anomalies']),
                        suggested_actions=[],
                        metadata={
                            'risk_level': hr_anomalies['risk_level'],
                            'avg_heart_rate': float(hr_anomalies['avg_heart_rate']),
                            'max_heart_rate': float(hr_anomalies['max_heart_rate']),
                        },
                    )
                )
        
        return {
            'insights': insights,
            'recommendations': list(set(recommendations)),  # Remove duplicates
            'analysis_date': datetime.now().isoformat(),
            'context_period': f"Last {self.contextual_reasoning.context_window} days"
        }