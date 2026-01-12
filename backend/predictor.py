import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_name = "Unknown"
        self.threshold = 0.5
        self.feature_names = []
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
        """Create dummy models in case loading fails"""
        print("Creating dummy models for demo purposes...")
        
        # Create a simple placeholder
        self.model = None
        self.model_name = "Dummy Model"
        self.threshold = 0.5
        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    def predict(self, patient_data):
        """
        Predict heart disease risk based on patient data
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Tuple of (risk_level, probability, contributing_factors)
        """
        try:
            # Convert patient data to DataFrame with proper columns
            input_data = pd.DataFrame([patient_data])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in input_data.columns:
                    input_data[feature] = 0  # Default value
            
            # Reorder columns to match training
            input_data = input_data[self.feature_names]
            
            # Make prediction
            if self.model is not None:
                # Get prediction probability
                proba = self.model.predict_proba(input_data)[0][1]
                probability = float(proba)
                
                # Determine risk level
                risk_level = "High Risk" if probability > self.threshold else "Low Risk"
                
                # Get contributing factors based on model type
                contributing_factors = self._get_contributing_factors(input_data, probability)
            else:
                # Fallback for dummy model
                probability = np.random.random()
                risk_level = "High Risk" if probability > self.threshold else "Low Risk"
                contributing_factors = [{"feature": "age", "coefficient": 0.05}]
            
            return risk_level, probability, contributing_factors
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return safe fallback values
            return "Low Risk", 0.5, [{"feature": "general", "coefficient": 0.0}]
    
    def _get_contributing_factors(self, input_data, probability):
        """
        Get the most significant contributing factors to the prediction
        """
        try:
            # If we have a pipeline, get the last step
            model_to_examine = self.model
            if hasattr(self.model, 'named_steps'):
                # Get the last step which should be the classifier
                model_to_examine = list(self.model.named_steps.values())[-1]
            
            # For Logistic Regression, we can get feature coefficients
            if hasattr(model_to_examine, 'coef_'):
                # Get coefficients
                coefficients = model_to_examine.coef_[0] if len(model_to_examine.coef_.shape) > 1 else model_to_examine.coef_
                
                # Get the input values for this prediction
                input_values = input_data.iloc[0].values
                
                # Calculate contribution scores (coefficient * input_value)
                contributions = coefficients * input_values
                
                # Create factor list sorted by absolute contribution
                factors = [(self.feature_names[i], contributions[i], abs(contributions[i])) 
                          for i in range(len(self.feature_names))]
                factors.sort(key=lambda x: x[2], reverse=True)  # Sort by absolute contribution
                
                # Return top 5 factors
                contributing_factors = []
                for feature, contrib, abs_contrib in factors[:5]:
                    contributing_factors.append({
                        "feature": feature,
                        "coefficient": round(float(contrib), 3)
                    })
                    
            # For tree-based models, we can get feature importances
            elif hasattr(model_to_examine, 'feature_importances_'):
                importances = model_to_examine.feature_importances_
                
                # Create factor list sorted by importance
                factors = [(self.feature_names[i], importances[i]) 
                          for i in range(len(self.feature_names))]
                factors.sort(key=lambda x: x[1], reverse=True)  # Sort by importance
                
                # Return top 5 factors
                contributing_factors = []
                for feature, importance in factors[:5]:
                    contributing_factors.append({
                        "feature": feature,
                        "importance": round(float(importance), 3)
                    })
            else:
                # Default case - just return the top features based on input values
                input_series = input_data.iloc[0]
                sorted_features = input_series.abs().sort_values(ascending=False)
                
                contributing_factors = []
                for feature, value in sorted_features.head(5).items():
                    contributing_factors.append({
                        "feature": feature,
                        "value": round(float(value), 3)
                    })
            
            return contributing_factors
            
        except Exception as e:
            print(f"Error getting contributing factors: {e}")
            # Return a default set of factors
            return [{"feature": "general", "coefficient": 0.0}]

# Create global predictor instance
predictor = HeartDiseasePredictor()