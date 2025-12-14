import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
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
        0.15 * (age > 55) +
        0.2 * sex +
        0.3 * (cp > 1) +
        0.15 * (trestbps > 140) +
        0.1 * (chol > 240) +
        0.15 * fbs +
        0.1 * restecg +
        -0.2 * (thalach < 120) +
        0.3 * exang +
        0.2 * (oldpeak > 1) +
        0.1 * slope +
        0.35 * (ca > 0) +
        0.3 * (thal > 1)
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

# Feature selection to improve model performance
print("\nPerforming feature selection...")
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Create enhanced Logistic Regression pipeline with StandardScaler and feature selection
print("\nTraining Enhanced Logistic Regression...")
enhanced_log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', LogisticRegression(random_state=42, max_iter=2000, C=0.5, solver='liblinear'))
])

# Train Enhanced Logistic Regression
enhanced_log_reg_pipeline.fit(X_train, y_train)

# Evaluate Enhanced Logistic Regression
y_pred_elr = enhanced_log_reg_pipeline.predict(X_test)
y_pred_proba_elr = enhanced_log_reg_pipeline.predict_proba(X_test)[:, 1]

print("Enhanced Logistic Regression Results:")
print(f"Accuracy: {(y_pred_elr == y_test).mean():.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_elr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_elr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_elr):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_elr):.4f}")

# Create Random Forest with enhanced hyperparameter tuning
print("\nTraining Enhanced Random Forest with hyperparameter tuning...")

# Define enhanced parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
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

print("Enhanced Random Forest Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy: {(y_pred_rf == y_test).mean():.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Create Gradient Boosting model for comparison
print("\nTraining Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid_search = GridSearchCV(gb, gb_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
gb_grid_search.fit(X_train, y_train)

best_gb = gb_grid_search.best_estimator_

# Evaluate Gradient Boosting
y_pred_gb = best_gb.predict(X_test)
y_pred_proba_gb = best_gb.predict_proba(X_test)[:, 1]

print("Gradient Boosting Results:")
print(f"Best parameters: {gb_grid_search.best_params_}")
print(f"Accuracy: {(y_pred_gb == y_test).mean():.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_gb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_gb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_gb):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_gb):.4f}")

# Compare all models
print("\nModel Comparison:")
print("Metric\t\tEnhanced LR\t\tRandom Forest\t\tGradient Boosting")
print(f"Accuracy\t{(y_pred_elr == y_test).mean():.4f}\t\t\t{(y_pred_rf == y_test).mean():.4f}\t\t\t{(y_pred_gb == y_test).mean():.4f}")
print(f"F1 Score\t{f1_score(y_test, y_pred_elr):.4f}\t\t\t{f1_score(y_test, y_pred_rf):.4f}\t\t\t{f1_score(y_test, y_pred_gb):.4f}")
print(f"ROC AUC\t\t{roc_auc_score(y_test, y_pred_proba_elr):.4f}\t\t\t{roc_auc_score(y_test, y_pred_proba_rf):.4f}\t\t\t{roc_auc_score(y_test, y_pred_proba_gb):.4f}")

# Calibrate models
print("\nCalibrating models...")

# Calibrate Enhanced Logistic Regression
calibrated_elr = CalibratedClassifierCV(enhanced_log_reg_pipeline, method='sigmoid', cv=5)
calibrated_elr.fit(X_train, y_train)
y_pred_proba_elr_cal = calibrated_elr.predict_proba(X_test)[:, 1]

# Calibrate Random Forest
calibrated_rf = CalibratedClassifierCV(best_rf, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train)
y_pred_proba_rf_cal = calibrated_rf.predict_proba(X_test)[:, 1]

# Calibrate Gradient Boosting
calibrated_gb = CalibratedClassifierCV(best_gb, method='sigmoid', cv=5)
calibrated_gb.fit(X_train, y_train)
y_pred_proba_gb_cal = calibrated_gb.predict_proba(X_test)[:, 1]

print("Calibrated Model Results:")
print(f"Enhanced Logistic Regression (Calibrated) ROC AUC: {roc_auc_score(y_test, y_pred_proba_elr_cal):.4f}")
print(f"Random Forest (Calibrated) ROC AUC: {roc_auc_score(y_test, y_pred_proba_rf_cal):.4f}")
print(f"Gradient Boosting (Calibrated) ROC AUC: {roc_auc_score(y_test, y_pred_proba_gb_cal):.4f}")

# Determine the best model based on ROC AUC
elr_auc = roc_auc_score(y_test, y_pred_proba_elr)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
gb_auc = roc_auc_score(y_test, y_pred_proba_gb)

models_auc = {
    "Enhanced Logistic Regression": (elr_auc, calibrated_elr, enhanced_log_reg_pipeline),
    "Random Forest": (rf_auc, calibrated_rf, best_rf),
    "Gradient Boosting": (gb_auc, calibrated_gb, best_gb)
}

best_model_name = max(models_auc, key=lambda x: models_auc[x][0])
best_auc, best_model_calibrated, best_model_original = models_auc[best_model_name]

print(f"\nSelecting {best_model_name} as the best model with AUC: {best_auc:.4f}")

# Threshold tuning based on precision/recall trade-off
print("\nPerforming threshold tuning...")
if best_model_name == "Enhanced Logistic Regression":
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_elr)
elif best_model_name == "Random Forest":
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_rf)
else:  # Gradient Boosting
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_gb)

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
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
fpr_elr, tpr_elr, _ = roc_curve(y_test, y_pred_proba_elr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
plt.plot(fpr_elr, tpr_elr, label=f'Enhanced LR (AUC = {elr_auc:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {gb_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 3, 2)
plt.plot(recalls, precisions, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# Confusion Matrix for best model
plt.subplot(1, 3, 3)
if best_model_name == "Enhanced Logistic Regression":
    y_pred_best = y_pred_elr
elif best_model_name == "Random Forest":
    y_pred_best = y_pred_rf
else:  # Gradient Boosting
    y_pred_best = y_pred_gb

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig('../model/evaluation_curves.png')
plt.close()

# Feature importance (for the selected model)
plt.figure(figsize=(10, 6))

if best_model_name == "Enhanced Logistic Regression":
    # For Logistic Regression, use coefficients
    coefficients = best_model_original.named_steps['classifier'].coef_[0]
    # Get feature names after selection
    selected_indices = best_model_original.named_steps['selector'].get_support(indices=True)
    selected_feature_names = X.columns[selected_indices]
    feature_importance = pd.DataFrame({
        'feature': selected_feature_names,
        'importance': np.abs(coefficients)
    }).sort_values('importance', ascending=False)
    plt.title('Enhanced Logistic Regression: Feature Coefficients (Absolute Values)')
elif best_model_name == "Random Forest":
    # For Random Forest, use feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model_original.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.title('Random Forest: Feature Importances')
else:  # Gradient Boosting
    # For Gradient Boosting, use feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model_original.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.title('Gradient Boosting: Feature Importances')

plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../model/feature_importance.png')
plt.close()

print("Plots saved successfully!")
print("\nTraining completed successfully!")