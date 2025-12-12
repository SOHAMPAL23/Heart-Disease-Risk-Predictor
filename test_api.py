import requests
import json

# Test data - sample patient information
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

# API endpoint
API_URL = "http://localhost:8002/predict"

def test_prediction():
    """Test the prediction API with sample data"""
    try:
        # Make POST request
        response = requests.post(
            API_URL,
            json=test_patient,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"Model: {result['model_used']}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Probability: {result['probability']:.3f}")
            print(f"Clinical Note: {result['clinical_note']}")
            print("\nTop Contributing Factors:")
            for factor in result['contributing_factors']:
                if 'coefficient' in factor:
                    print(f"  - {factor['feature']}: coef={factor['coefficient']:.3f}")
                elif 'importance' in factor:
                    print(f"  - {factor['feature']}: importance={factor['importance']:.3f}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error making prediction: {e}")
        return False

if __name__ == "__main__":
    print("Testing Heart Disease Prediction API")
    print("====================================")
    
    success = test_prediction()
    
    if success:
        print("\n✅ Test passed! The API is working correctly.")
    else:
        print("\n❌ Test failed. Please check the API.")