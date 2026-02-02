# Main application entrypoint for Vercel deployment
from backend.app import app

# This ensures the app is available as the main application
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002)