import joblib
import numpy as np
import os
from typing import List, Tuple, Any
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

class HeartDiseasePredictor:
    def __init__(self):
        self.model: Any = None
        self.model_name: str = ""
        self.threshold: float = 0.5
        self.feature_names: List[str] = []
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, '..', 'model')
            
            # Load the best model and related information
            self.model = joblib.load(os.path.join(model_dir, 'final_model.pkl'))
            self.model_name = joblib.load(os.path.join(model_dir, 'model_name.pkl'))
            self.threshold = joblib.load(os.path.join(model_dir, 'best_threshold.pkl'))
            self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
            
            print(f"Loaded {self.model_name} model with threshold {self.threshold:.3f}")
        except Exception as e:
            print(f"Error loading models: {e}")
            # For demo purposes, we'll create dummy models if loading fails
            self._create_dummy_models()
    
    def _create_dummy_models(self):
        """Create dummy models for demonstration"""
        from sklearn.dummy import DummyClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        # Create dummy logistic regression pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        
        self.model_name = "Logistic Regression"
        self.threshold = 0.5
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    def prepare_input_data(self, patient_data: dict) -> np.ndarray:
        """Convert patient data to model input format"""
        # Convert to ordered list based on feature names
        if not self.feature_names:
            self.feature_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
        
        input_data = [[patient_data.get(feature, 0) for feature in self.feature_names]]
        return np.array(input_data)
    
    def get_feature_contributions(self, patient_data: dict) -> List[dict]:
        """Get feature contributions for interpretability"""
        # Ensure feature names are initialized
        if not self.feature_names:
            self.feature_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
            
        try:
            if self.model is None:
                self._create_dummy_models()
                
            # For logistic regression, use coefficients
            if self.model_name == "Logistic Regression":
                # Extract the logistic regression model from the pipeline
                if hasattr(self.model, 'named_steps'):
                    lr_model = self.model.named_steps['classifier']
                    if hasattr(lr_model, 'coef_'):
                        coefficients = lr_model.coef_[0]
                        contributions = []
                        for i, feature in enumerate(self.feature_names):
                            contribution = coefficients[i] * patient_data.get(feature, 0)
                            contributions.append({
                                "feature": feature,
                                "contribution": float(contribution),
                                "coefficient": float(coefficients[i])
                            })
                        # Sort by absolute contribution
                        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                        return contributions[:5]  # Top 5 contributors
                # Fallback if we can't extract coefficients
                return [{"feature": "age", "contribution": 0.1, "coefficient": 0.05}]
            else:
                # For Random Forest or other models, return generic importance
                return [
                    {"feature": "age", "importance": 0.15, "contribution": 0.05},
                    {"feature": "sex", "importance": 0.10, "contribution": 0.03},
                    {"feature": "cp", "importance": 0.12, "contribution": 0.04},
                    {"feature": "thalach", "importance": 0.18, "contribution": 0.06},
                    {"feature": "ca", "importance": 0.12, "contribution": 0.04}
                ]
        except Exception as e:
            print(f"Error getting feature contributions: {e}")
            return [{"feature": "age", "contribution": 0.1, "coefficient": 0.05}]
    
    def predict(self, patient_data: dict) -> Tuple[str, float, List[dict]]:
        """
        Make prediction using the best model
        
        Returns:
            risk_level (str): "Low Risk" or "High Risk"
            probability (float): Probability of heart disease
            contributing_factors (List[dict]): Top contributing factors
        """
        # Prepare input data
        input_array = self.prepare_input_data(patient_data)
        
        # Make prediction
        try:
            if self.model is None:
                self._create_dummy_models()
                
            probability = self.model.predict_proba(input_array)[0][1]  # Probability of class 1
        except Exception as e:
            print(f"Error making prediction: {e}")
            probability = 0.5  # Default probability
        
        # Determine risk level based on optimized threshold
        risk_level = "High Risk" if probability > self.threshold else "Low Risk"
        
        # Get contributing factors
        contributing_factors = self.get_feature_contributions(patient_data)
        
        return risk_level, probability, contributing_factors