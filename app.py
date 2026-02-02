# Vercel FastAPI application entrypoint for Poetry deployment
import os
import sys

# Add the project root to the path to import from backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.app import app
    # For Vercel deployment
    application = app
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback - create a minimal app
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"message": "Heart Disease Prediction API - Poetry Deployment", "status": "running"}
    
    application = app

# Allow importing as module
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)