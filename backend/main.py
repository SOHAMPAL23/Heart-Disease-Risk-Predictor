from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
import sys

# Add the parent directory to the path to import from backend module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import PatientData, PredictionResult
from backend.predictor import HeartDiseasePredictor

app = FastAPI(title="Heart Disease Prediction API", 
              description="API for predicting heart disease risk based on patient data",
              version="1.0.0")

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = HeartDiseasePredictor()

@app.get("/")
async def root():
    return {"message": "Heart Disease Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResult)
async def predict_heart_disease(data: PatientData, model: str = "Logistic Regression"):
    """
    Predict heart disease risk based on patient data
    
    Args:
        data: Patient data including clinical measurements
        model: Model to use for prediction ("Logistic Regression" or "Random Forest")
    
    Returns:
        Prediction result including risk level, probability, and contributing factors
    """
    try:
        # Convert Pydantic model to dictionary
        patient_data = data.dict()
        
        # Validate model selection
        if model not in ["Logistic Regression", "Random Forest"]:
            raise HTTPException(status_code=400, detail="Invalid model selection")
        
        # Make prediction
        risk_level, probability, contributing_factors = predictor.predict(patient_data, model)
        
        # Generate clinical note based on risk level
        if risk_level == "High Risk":
            clinical_note = "Patient is at high risk for heart disease. Clinical follow-up recommended within 48 hours."
        else:
            clinical_note = "Patient is at low risk for heart disease. Routine checkup recommended annually."
        
        # Return result
        return PredictionResult(
            risk_level=risk_level,
            probability=float(probability),
            contributing_factors=contributing_factors,
            clinical_note=clinical_note,
            model_used=model
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)  # Changed port to 8002