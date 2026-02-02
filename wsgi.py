# WSGI/ASGI entrypoint for deployment platforms
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

# This is the standard ASGI entrypoint
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "wsgi:app", 
        host="0.0.0.0", 
        port=8002,
        reload=False,
        log_level="info"
    )