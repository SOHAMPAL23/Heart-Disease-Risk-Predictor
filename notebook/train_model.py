import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Load the UCI Heart Disease dataset
# Since we don't have the actual dataset, we'll create a synthetic one that follows the same pattern
# But first, let's try to download the real dataset
try:
    # Try to load from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=column_names, na_values='?')
except:
    # If we can't download, create a synthetic dataset
    print("Could not download dataset, creating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Create more realistic synthetic data
    age = np.random.randint(29, 80, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)
    trestbps = np.random.randint(90, 200, n_samples)
    chol = np.random.randint(120, 400, n_samples)
    fbs = np.random.randint(0, 2, n_samples)
    restecg = np.random.randint(0, 3, n_samples)
    thalach = np.random.randint(70, 200, n_samples)
    exang = np.random.randint(0, 2, n_samples)
    oldpeak = np.round(np.random.uniform(0, 6, n_samples), 1)
    slope = np.random.randint(0, 3, n_samples)
    ca = np.random.randint(0, 4, n_samples)
    thal = np.random.randint(0, 4, n_samples)
    
    # Create target with some correlation to features
    target_prob = (
        0.1 * (age > 50) +
        0.15 * sex +
        0.2 * (cp > 1) +
        0.1 * (trestbps > 140) +
        0.05 * (chol > 240) +
        0.1 * fbs +
        0.05 * restecg +
        -0.15 * (thalach < 120) +
        0.2 * exang +
        0.1 * (oldpeak > 1) +
        0.05 * slope +
        0.25 * (ca > 0) +
        0.2 * (thal > 1)
    )
    
    target = np.random.binomial(1, np.clip(target_prob, 0, 1), n_samples)
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    }
    
    df = pd.DataFrame(data)

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df['target'].value_counts())

# Handle missing values
print("\nMissing values:")
print(df.isnull().sum())

# For the real dataset, we need to handle missing values
df = df.dropna()

# Prepare features and target
X = df.drop('target', axis=1)
y = (df['target'] > 0).astype(int)  # Convert to binary classification

print("\nDataset after cleaning:", X.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Create Logistic Regression pipeline with StandardScaler
print("\nTraining Logistic Regression...")
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train Logistic Regression
log_reg_pipeline.fit(X_train, y_train)

# Evaluate Logistic Regression
y_pred_lr = log_reg_pipeline.predict(X_test)
y_pred_proba_lr = log_reg_pipeline.predict_proba(X_test)[:, 1]

print("Logistic Regression Results:")
print(f"Accuracy: {(y_pred_lr == y_test).mean():.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# Create Random Forest with hyperparameter tuning
print("\nTraining Random Forest with hyperparameter tuning...")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search with Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Evaluate Random Forest
y_pred_rf = best_rf.predict(X_test)
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]

print("Random Forest Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy: {(y_pred_rf == y_test).mean():.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Compare models
print("\nModel Comparison:")
print("Metric\t\tLogistic Regression\tRandom Forest")
print(f"Accuracy\t{(y_pred_lr == y_test).mean():.4f}\t\t\t{(y_pred_rf == y_test).mean():.4f}")
print(f"F1 Score\t{f1_score(y_test, y_pred_lr):.4f}\t\t\t{f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC\t\t{roc_auc_score(y_test, y_pred_proba_lr):.4f}\t\t\t{roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Calibrate models
from sklearn.calibration import CalibratedClassifierCV

print("\nCalibrating models...")

# Calibrate Logistic Regression
calibrated_lr = CalibratedClassifierCV(log_reg_pipeline, method='sigmoid', cv=5)
calibrated_lr.fit(X_train, y_train)
y_pred_proba_lr_cal = calibrated_lr.predict_proba(X_test)[:, 1]

# Calibrate Random Forest
calibrated_rf = CalibratedClassifierCV(best_rf, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train)
y_pred_proba_rf_cal = calibrated_rf.predict_proba(X_test)[:, 1]

print("Calibrated Model Results:")
print(f"Logistic Regression (Calibrated) ROC AUC: {roc_auc_score(y_test, y_pred_proba_lr_cal):.4f}")
print(f"Random Forest (Calibrated) ROC AUC: {roc_auc_score(y_test, y_pred_proba_rf_cal):.4f}")

# Determine the best model based on ROC AUC
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)

if lr_auc >= rf_auc:
    print("\nSelecting Logistic Regression as the best model")
    best_model = log_reg_pipeline  # Use the original model for feature importance
    best_model_calibrated = calibrated_lr  # Use calibrated model for predictions
    best_model_name = "Logistic Regression"
    best_threshold = 0.5  # Default threshold
else:
    print("\nSelecting Random Forest as the best model")
    best_model = best_rf  # Use the original model for feature importance
    best_model_calibrated = calibrated_rf  # Use calibrated model for predictions
    best_model_name = "Random Forest"
    best_threshold = 0.5  # Default threshold

# Threshold tuning based on precision/recall trade-off
from sklearn.metrics import precision_recall_curve

print("\nPerforming threshold tuning...")
if best_model_name == "Logistic Regression":
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_lr)
else:
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_rf)

# Find threshold that maximizes F1 score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"Optimal threshold: {best_threshold:.3f}")
print(f"Best F1 score at this threshold: {f1_scores[best_threshold_idx]:.4f}")

# Save the best model
print("\nSaving the best model...")
joblib.dump(best_model_calibrated, '../model/final_model.pkl')
joblib.dump(best_model_name, '../model/model_name.pkl')
joblib.dump(best_threshold, '../model/best_threshold.pkl')
joblib.dump(list(X.columns), '../model/feature_names.pkl')

print("Model saved successfully!")

# Create visualization plots
print("\nCreating visualization plots...")

# ROC Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recalls, precisions, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.savefig('../model/evaluation_curves.png')
plt.close()

# Feature importance (for the selected model)
plt.figure(figsize=(10, 6))

if best_model_name == "Logistic Regression":
    # For Logistic Regression, use coefficients
    coefficients = best_model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(coefficients)
    }).sort_values('importance', ascending=False)
    plt.title('Logistic Regression: Feature Coefficients (Absolute Values)')
else:
    # For Random Forest, use feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.title('Random Forest: Feature Importances')

plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../model/feature_importance.png')
plt.close()

print("Plots saved successfully!")
print("\nTraining completed successfully!")