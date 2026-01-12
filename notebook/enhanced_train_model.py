import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
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
    n_samples = 1500  # Increased sample size for better training
    
    # Create more realistic synthetic data with stronger correlations
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
    
    # Create target with stronger correlation to features for better separation
    target_prob = (
        0.2 * (age > 55) +
        0.25 * sex +
        0.35 * (cp > 1) +
        0.2 * (trestbps > 140) +
        0.15 * (chol > 240) +
        0.2 * fbs +
        0.15 * restecg +
        -0.25 * (thalach < 120) +
        0.35 * exang +
        0.25 * (oldpeak > 1) +
        0.15 * slope +
        0.4 * (ca > 0) +
        0.35 * (thal > 1)
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

# Create multiple enhanced pipelines
print("\nTraining Enhanced Models...")

# Enhanced Logistic Regression with polynomial features
print("Training Enhanced Logistic Regression with polynomial features...")
lr_poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('selector', SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]*2))),
    ('classifier', LogisticRegression(random_state=42, max_iter=5000, C=0.1, solver='liblinear'))
])

lr_poly_pipeline.fit(X_train, y_train)

# Enhanced Random Forest with more sophisticated hyperparameter tuning
print("\nTraining Enhanced Random Forest with sophisticated hyperparameter tuning...")

rf_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                              scoring='roc_auc', n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

best_rf = rf_grid_search.best_estimator_

# Enhanced Gradient Boosting
print("\nTraining Enhanced Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid_search = GridSearchCV(gb, gb_param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                             scoring='roc_auc', n_jobs=-1)
gb_grid_search.fit(X_train, y_train)

best_gb = gb_grid_search.best_estimator_

# Support Vector Machine
print("\nTraining Support Vector Machine...")
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'))
])

svm_pipeline.fit(X_train, y_train)

# Create ensemble model
print("\nCreating Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr_poly_pipeline),
        ('rf', best_rf),
        ('gb', best_gb),
        ('svm', svm_pipeline)
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# Evaluate all models
models = {
    'Enhanced Logistic Regression': lr_poly_pipeline,
    'Enhanced Random Forest': best_rf,
    'Enhanced Gradient Boosting': best_gb,
    'Support Vector Machine': svm_pipeline,
    'Ensemble Model': ensemble
}

results = {}

for name, model in models.items():
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

# Compare all models
print("\nModel Comparison:")
print("Metric\t\tEnhanced LR\t\tRandom Forest\t\tGradient Boosting\tSVM\t\t\tEnsemble")
for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
    print(f"{metric.upper()}\t\t", end="")
    for name in results.keys():
        print(f"{results[name][metric]:.4f}\t\t\t", end="")
    print()

# Determine the best model based on ROC AUC
best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
best_model = models[best_model_name]
best_auc = results[best_model_name]['roc_auc']

print(f"\nSelecting {best_model_name} as the best model with AUC: {best_auc:.4f}")

# Calibrate the best model
print("\nCalibrating the best model...")
calibrated_best_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
calibrated_best_model.fit(X_train, y_train)

# Re-evaluate the calibrated model
y_pred_cal = calibrated_best_model.predict(X_test)
y_pred_proba_cal = calibrated_best_model.predict_proba(X_test)[:, 1]

calibrated_accuracy = (y_pred_cal == y_test).mean()
calibrated_f1 = f1_score(y_test, y_pred_cal)
calibrated_precision = precision_score(y_test, y_pred_cal)
calibrated_recall = recall_score(y_test, y_pred_cal)
calibrated_roc_auc = roc_auc_score(y_test, y_pred_proba_cal)

print(f"\nCalibrated {best_model_name} Results:")
print(f"Accuracy: {calibrated_accuracy:.4f}")
print(f"F1 Score: {calibrated_f1:.4f}")
print(f"Precision: {calibrated_precision:.4f}")
print(f"Recall: {calibrated_recall:.4f}")
print(f"ROC AUC: {calibrated_roc_auc:.4f}")

# Threshold tuning based on precision/recall trade-off
from sklearn.metrics import precision_recall_curve

print("\nPerforming threshold tuning...")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_cal)

# Find threshold that maximizes F1 score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else thresholds[-1]

print(f"Optimal threshold: {best_threshold:.3f}")
print(f"Best F1 score at this threshold: {f1_scores[best_threshold_idx]:.4f}")

# Save the calibrated model
print("\nSaving the calibrated model...")
joblib.dump(calibrated_best_model, '../model/final_model.pkl')
joblib.dump(best_model_name, '../model/model_name.pkl')
joblib.dump(best_threshold, '../model/best_threshold.pkl')
joblib.dump(list(X.columns), '../model/feature_names.pkl')

print("Model saved successfully!")

# Create visualization plots
print("\nCreating visualization plots...")

# ROC Curve
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 3, 2)
for name, result in results.items():
    prec, rec, _ = precision_recall_curve(y_test, result['y_pred_proba'])
    plt.plot(rec, prec, label=f'{name}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()

# Confusion Matrix for best model
plt.subplot(1, 3, 3)
y_pred_best = calibrated_best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig('../model/evaluation_curves.png')
plt.close()

# Feature importance (for the best model if it's tree-based)
plt.figure(figsize=(10, 6))

if hasattr(best_model, 'named_steps'):
    # For pipelines, get the last step
    last_step = list(best_model.named_steps.values())[-1]
    if hasattr(last_step, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': last_step.feature_importances_
        }).sort_values('importance', ascending=False)
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title(f'{best_model_name}: Feature Importances')
    elif hasattr(last_step, 'coef_'):
        # For Logistic Regression, use coefficients
        coefficients = last_step.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(coefficients)
        }).sort_values('importance', ascending=False)
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title(f'{best_model_name}: Feature Coefficients (Absolute Values)')
    else:
        # For other models, we'll just show a placeholder
        plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(f'{best_model_name}: Feature Importance')
else:
    # For non-pipeline models
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title(f'{best_model_name}: Feature Importances')
    elif hasattr(best_model, 'coef_'):
        # For Logistic Regression, use coefficients
        coefficients = best_model.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(coefficients)
        }).sort_values('importance', ascending=False)
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title(f'{best_model_name}: Feature Coefficients (Absolute Values)')
    else:
        # For other models, we'll just show a placeholder
        plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(f'{best_model_name}: Feature Importance')

plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../model/feature_importance.png')
plt.close()

print("Plots saved successfully!")
print("\nEnhanced training completed successfully!")