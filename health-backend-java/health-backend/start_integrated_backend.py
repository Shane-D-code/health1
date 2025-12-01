#!/usr/bin/env python3
"""
Startup script for the Integrated Health Backend
This script starts the FastAPI server with AI integration
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def main():
    print("🏥 Starting Integrated Health Backend...")
    print("📊 Features: AI Analysis, Trend Detection, Anomaly Detection")
    print("🤖 AI Engine: Active")
    print("🌐 CORS: Enabled for frontend integration")
    print("📡 Server starting on http://localhost:8001")
    print("-" * 50)
    
    try:
        # Import and run the integrated backend
        uvicorn.run(
            "integrated_main:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()