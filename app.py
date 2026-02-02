# Vercel FastAPI application entrypoint
import os
import sys

# Add the project root to the path to import from backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

# For Vercel deployment
application = app

# Allow importing as module
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)