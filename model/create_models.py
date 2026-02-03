import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create dummy feature names
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Create dummy logistic regression pipeline (mimicking the structure)
class DummyLogisticRegression:
    def __init__(self):
        self.coef_ = [np.array([
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
        ])]
    
    def predict_proba(self, X):
        # Generate random probabilities
        prob_1 = np.random.beta(2, 2, size=X.shape[0])  # Beta distribution for realistic probabilities
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

# Create dummy random forest model
class DummyRandomForest:
    def __init__(self):
        # Feature importances
        self.feature_importances_ = np.array([
            0.15,  # age
            0.10,  # sex
            0.12,  # cp
            0.08,  # trestbps
            0.07,  # chol
            0.05,  # fbs
            0.06,  # restecg
            0.18,  # thalach
            0.10,  # exang
            0.04,  # oldpeak
            0.03,  # slope
            0.12,  # ca
            0.10   # thal
        ])
    
    def predict_proba(self, X):
        # Generate random probabilities
        prob_1 = np.random.beta(2, 2, size=X.shape[0])  # Beta distribution for realistic probabilities
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

# Create dummy logistic regression pipeline
log_reg_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DummyLogisticRegression())
])

# Save the dummy models
joblib.dump(log_reg_model, 'final_model.pkl')
joblib.dump("Logistic Regression", 'model_name.pkl')
joblib.dump(0.5, 'best_threshold.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("Dummy models created successfully!")
print("Files saved:")
print("- final_model.pkl")
print("- model_name.pkl")
print("- best_threshold.pkl")
print("- feature_names.pkl")