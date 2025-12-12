import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor import HeartDiseasePredictor

# Create a test patient
test_patient = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

# Initialize predictor
predictor = HeartDiseasePredictor()

# Make prediction
risk_level, probability, contributing_factors = predictor.predict(test_patient)

print(f"Model: {predictor.model_name}")
print(f"Threshold: {predictor.threshold:.3f}")
print(f"Risk Level: {risk_level}")
print(f"Probability: {probability:.3f}")
print("\nTop Contributing Factors:")
for factor in contributing_factors:
    if 'coefficient' in factor:
        print(f"  - {factor['feature']}: coef={factor['coefficient']:.3f}")
    elif 'importance' in factor:
        print(f"  - {factor['feature']}: importance={factor['importance']:.3f}")