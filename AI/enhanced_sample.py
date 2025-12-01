from llm_insight_generator import EnhancedAIReasoningEngine
from sample_usage import create_sample_data

def main():
    # Initialize enhanced AI engine with LLM capabilities
    ai_engine = EnhancedAIReasoningEngine()
    
    # Create sample data
    health_data = create_sample_data()
    
    # Analyze with LLM-powered insights
    results = ai_engine.analyze_with_llm(health_data)
    
    # Display enhanced results
    print("🤖 Enhanced AI Health Analysis Report")
    print("=" * 45)
    print(f"Analysis Type: {results['analysis_type']}")
    print(f"Context Period: {results['context_period']}")
    print()
    
    print("🧠 LLM-Generated Health Insights:")
    for i, insight in enumerate(results['insights'], 1):
        print(f"{i}. {insight}")
    
    print("\n💡 Personalized Lifestyle Recommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()