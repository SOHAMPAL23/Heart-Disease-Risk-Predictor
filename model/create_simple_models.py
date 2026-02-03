import pickle
import numpy as np

# Create dummy feature names
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Create a simple dummy model class
class SimpleDummyModel:
    def __init__(self):
        # Feature coefficients for logistic regression-like behavior
        self.coef_ = np.array([
            0.1,   # age
            0.5,   # sex
            0.8,   # cp
            0.3,   # trestbps
            0.2,   # chol
            0.1,   # fbs
            0.2,   # restecg
            -0.6,  # thalach
            0.7,   # exang
            0.4,   # oldpeak
            0.3,   # slope
            0.9,   # ca
            0.5    # thal
        ])
    
    def predict_proba(self, X):
        # Simple linear combination + sigmoid for probability
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Linear combination
        linear_combination = np.dot(X, self.coef_)
        
        # Sigmoid function to get probability
        prob_1 = 1 / (1 + np.exp(-linear_combination))
        prob_0 = 1 - prob_1
        
        return np.column_stack([prob_0, prob_1])

# Create a simple pipeline-like structure
class SimplePipeline:
    def __init__(self):
        self.named_steps = {
            'scaler': None,  # We'll handle scaling in the predictor
            'classifier': SimpleDummyModel()
        }
    
    def predict_proba(self, X):
        return self.named_steps['classifier'].predict_proba(X)

# Create the model
model = SimplePipeline()

# Save all required files
with open('final_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_name.pkl', 'wb') as f:
    pickle.dump("Logistic Regression", f)

with open('best_threshold.pkl', 'wb') as f:
    pickle.dump(0.5, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("Dummy models created successfully using pickle!")
print("Files saved:")
print("- final_model.pkl")
print("- model_name.pkl")
print("- best_threshold.pkl")
print("- feature_names.pkl")