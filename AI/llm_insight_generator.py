import json
from typing import Dict, List, Any, Tuple
import pandas as pd


class PublicHealthGuidelineRetriever:
    """
    Very lightweight RAG-style retriever for public health guidelines.

    In a real deployment this would query a vector store over guideline texts.
    Here we simulate it with an in-memory knowledge base to demonstrate the pattern.
    """

    def __init__(self) -> None:
        self.knowledge_base: Dict[str, Dict[str, str]] = {
            'sleep': {
                'summary': (
                    "Most healthy adults are recommended to get 7–9 hours of sleep "
                    "per night to support cognitive function, mood, and metabolic health."
                ),
                'source': 'Based on general guidance from organizations like the CDC and NIH.',
            },
            'heart_rate': {
                'summary': (
                    "Resting heart rate for most adults typically ranges from about "
                    "60–100 beats per minute; consistently higher rates can be a sign "
                    "of stress, low fitness, or underlying health issues."
                ),
                'source': 'Summarized from common cardiology and public health references.',
            },
            'hydration': {
                'summary': (
                    "Daily fluid needs vary, but many guidelines suggest around 2–3 liters "
                    "of total fluids per day for adults, including water and other drinks."
                ),
                'source': 'Derived from general public health hydration recommendations.',
            },
        }

    def retrieve_guideline(self, topic: str) -> Tuple[str, str]:
        """Return a (summary, source_note) tuple for a given health topic."""
        data = self.knowledge_base.get(topic, {})
        return data.get('summary', ''), data.get('source', '')


class LLMInsightGenerator:
    def __init__(self):
        self.system_prompt = """You are a health data analyst AI. Analyze the provided health metrics and generate:
1. Clear, human-friendly explanations of patterns
2. Health warnings when needed
3. Lifestyle insights and suggestions

Important constraints:
- You are not a doctor and must not provide a diagnosis.
- Encourage users to consult a healthcare professional for serious concerns.
- Keep responses concise, empathetic, and actionable, using casual, friendly language."""
        self.guideline_retriever = PublicHealthGuidelineRetriever()

    def generate_sleep_insight(self, sleep_data: pd.DataFrame) -> str:
        """Generate LLM-powered sleep insights with guideline context (RAG-style)."""
        if sleep_data.empty:
            return "No sleep data available for analysis."
        
        stats = {
            'avg_hours': round(float(sleep_data['duration_hours'].mean()), 1),
            'min_hours': float(sleep_data['duration_hours'].min()),
            'max_hours': float(sleep_data['duration_hours'].max()),
            'variance': round(float(sleep_data['duration_hours'].var()), 2),
            'days_analyzed': len(sleep_data)
        }
        
        guideline_text, _ = self.guideline_retriever.retrieve_guideline('sleep')

        prompt = f"""System: {self.system_prompt}

User:
Here are 7-day sleep statistics for a user: {json.dumps(stats)}.

Public health guideline (retrieved via RAG):
\"\"\"{guideline_text}\"\"\"

Task:
- In 1–2 clear sentences, explain the user's sleep pattern in plain language.
- Reference the guideline only conceptually (do not sound overly clinical).
- Avoid medical diagnosis; gently suggest speaking to a professional if they are worried."""
        
        # Simulate LLM response (replace with actual LLM API call)
        return self._simulate_llm_response(prompt, 'sleep', stats)
    
    def generate_heart_rate_insight(self, hr_data: pd.DataFrame) -> str:
        """Generate LLM-powered heart rate insights with guideline context (RAG-style)."""
        if hr_data.empty:
            return "No heart rate data available for analysis."
        
        stats = {
            'avg_hr': round(float(hr_data['heart_rate'].mean()), 1),
            'max_hr': int(hr_data['heart_rate'].max()),
            'min_hr': int(hr_data['heart_rate'].min()),
            'spikes': len(hr_data[hr_data['heart_rate'] > hr_data['heart_rate'].quantile(0.8)]),
            'readings': len(hr_data)
        }
        
        guideline_text, _ = self.guideline_retriever.retrieve_guideline('heart_rate')

        prompt = f"""System: {self.system_prompt}

User:
Here are heart rate statistics for a user: {json.dumps(stats)}.

Public health guideline (retrieved via RAG):
\"\"\"{guideline_text}\"\"\"

Task:
- In 1–2 friendly sentences, describe any notable patterns (spikes, elevated average).
- Relate the pattern to the typical healthy range in simple language.
- Avoid diagnosis; instead, suggest general healthy habits and note that persistent issues
  should be discussed with a healthcare professional."""
        
        return self._simulate_llm_response(prompt, 'heart_rate', stats)
    
    def generate_lifestyle_recommendations(self, all_insights: List[str]) -> List[str]:
        """Generate comprehensive lifestyle recommendations"""
        prompt = f"""Based on these health insights: {all_insights}
        
Generate 3-5 specific, actionable lifestyle recommendations that address the identified issues.
Focus on practical daily habits and behavioral changes.

Response format: List of bullet points."""
        
        # Simulate LLM response
        return self._simulate_lifestyle_recommendations(all_insights)
    
    def _simulate_llm_response(self, prompt: str, data_type: str, stats: Dict) -> str:
        """Simulate LLM response with intelligent logic"""
        if data_type == 'sleep':
            avg_hours = stats['avg_hours']
            variance = stats['variance']
            
            if avg_hours < 6.5:
                return f"You're averaging only {avg_hours} hours of sleep - significantly below the recommended 7-9 hours."
            elif avg_hours < 7:
                return f"Your sleep duration of {avg_hours} hours is on the lower side - aim for 7-9 hours nightly."
            elif variance > 2:
                return f"Your sleep schedule varies widely ({stats['min_hours']}-{stats['max_hours']} hours) - consistency helps sleep quality."
            else:
                return f"Great job maintaining {avg_hours} hours of sleep consistently!"
        
        elif data_type == 'heart_rate':
            max_hr = stats['max_hr']
            avg_hr = stats['avg_hr']
            spikes = stats['spikes']
            
            if max_hr > 120 and spikes > 2:
                return f"Heart rate spiked to {max_hr} bpm with {spikes} elevated readings - monitor stress levels."
            elif avg_hr > 80:
                return f"Your average heart rate of {avg_hr} bpm is elevated - consider relaxation techniques."
            else:
                return f"Heart rate patterns look normal with average of {avg_hr} bpm."
        
        return "Data analyzed successfully."
    
    def _simulate_lifestyle_recommendations(self, insights: List[str]) -> List[str]:
        """Generate contextual lifestyle recommendations"""
        recommendations = []
        insight_text = ' '.join(insights).lower()
        
        if 'sleep' in insight_text and ('low' in insight_text or 'below' in insight_text):
            recommendations.extend([
                "Set a consistent bedtime routine 30 minutes before sleep",
                "Avoid caffeine after 2 PM to improve sleep quality"
            ])
        
        if 'heart rate' in insight_text and ('spike' in insight_text or 'elevated' in insight_text):
            recommendations.extend([
                "Practice 5-minute breathing exercises during stressful moments",
                "Take short walks to naturally regulate heart rate"
            ])
        
        if 'hydration' in insight_text and 'below' in insight_text:
            recommendations.extend([
                "Set hourly water reminders on your phone",
                "Keep a water bottle visible at your workspace"
            ])
        
        # Add general wellness recommendations
        recommendations.extend([
            "Take 2-minute movement breaks every hour",
            "Practice mindful eating by avoiding distractions during meals"
        ])
        
        return recommendations[:5]  # Return max 5 recommendations

class EnhancedAIReasoningEngine:
    def __init__(self):
        from ai_reasoning_engine import HealthInsightGenerator, ContextualReasoning
        self.traditional_generator = HealthInsightGenerator()
        self.llm_generator = LLMInsightGenerator()
        self.contextual_reasoning = ContextualReasoning()
    
    def analyze_with_llm(self, health_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced analysis using LLM-powered insights"""
        insights = []
        
        # Get weekly context
        weekly_data = {}
        for data_type, df in health_data.items():
            weekly_data[data_type] = self.contextual_reasoning.get_weekly_context(df)
        
        # Generate LLM-powered insights
        if 'sleep' in weekly_data and not weekly_data['sleep'].empty:
            sleep_insight = self.llm_generator.generate_sleep_insight(weekly_data['sleep'])
            insights.append(sleep_insight)
        
        if 'heart_rate' in weekly_data and not weekly_data['heart_rate'].empty:
            hr_insight = self.llm_generator.generate_heart_rate_insight(weekly_data['heart_rate'])
            insights.append(hr_insight)
        
        if 'hydration' in weekly_data and not weekly_data['hydration'].empty:
            # Use traditional analysis for hydration
            hydration_result = self.traditional_generator.analyze_hydration(weekly_data['hydration'])
            insights.append(hydration_result['message'])
        
        # Generate lifestyle recommendations
        recommendations = self.llm_generator.generate_lifestyle_recommendations(insights)
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'analysis_type': 'LLM-Enhanced',
            'context_period': "Last 7 days"
        }