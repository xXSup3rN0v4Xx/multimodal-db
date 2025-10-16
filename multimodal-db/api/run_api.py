"""
API Launcher
Simple launcher for the FastAPI unified API.
"""
import sys
from pathlib import Path

# Add the parent multimodal-db directory to path so we can import 'core' and 'api'
# This file is in: multimodal-db/multimodal-db/api/run_api.py
# We need to add: multimodal-db/multimodal-db/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the API
if __name__ == "__main__":
    try:
        import uvicorn
        from api.main import app
        
        print("🗾 Starting Multimodal-DB Unified API...")
        print("📍 API Documentation: http://localhost:8001/docs")
        print("🚀 Ready for chatbot-python-core and chatbot-nextjs-webui integration!")
        
        uvicorn.run(
            "api.main:app",  # Use string import path for reload to work
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have activated the virtual environment and installed dependencies:")
        print("   venv\\Scripts\\Activate.ps1")
        print("   pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ Error starting API: {e}")