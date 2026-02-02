# Simple server entrypoint for deployment platforms
from backend.app import app

# This ensures compatibility with various deployment platforms
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8002)