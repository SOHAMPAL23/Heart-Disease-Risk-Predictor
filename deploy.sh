#!/bin/bash
# Deployment script for Heart Disease Prediction API

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI application using uvicorn
uvicorn backend.app:app --host 0.0.0.0 --port 8002 --reload=false