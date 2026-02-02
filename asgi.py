# ASGI entrypoint for deployment platforms
from backend.app import app

# Standard ASGI application object
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "asgi:app", 
        host="0.0.0.0", 
        port=8002,
        reload=False
    )