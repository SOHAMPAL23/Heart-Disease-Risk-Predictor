from pydantic import BaseModel
from typing import List, Optional

class PatientData(BaseModel):
    age: int
    sex: int  # 0: female, 1: male
    cp: int   # chest pain type (0-3)
    trestbps: int  # resting blood pressure
    chol: int  # serum cholesterol
    fbs: int   # fasting blood sugar > 120 mg/dl (0/1)
    restecg: int  # resting electrocardiographic results (0-2)
    thalach: int  # maximum heart rate achieved
    exang: int   # exercise induced angina (0/1)
    oldpeak: float  # ST depression induced by exercise
    slope: int    # slope of peak exercise ST segment (0-2)
    ca: int       # number of major vessels colored (0-3)
    thal: int     # thalassemia (0-3)

class PredictionResult(BaseModel):
    risk_level: str  # "Low Risk" or "High Risk"
    probability: float
    contributing_factors: List[dict]
    clinical_note: str
    model_used: str  # "Logistic Regression" or "Random Forest"