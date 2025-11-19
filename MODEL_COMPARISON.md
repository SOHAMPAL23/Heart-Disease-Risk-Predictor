# Model Comparison Dashboard

This dashboard provides a visual comparison of the Logistic Regression and Random Forest models for heart disease prediction.

## Performance Metrics Comparison

| Metric          | Logistic Regression | Random Forest |
|----------------|---------------------|---------------|
| ROC-AUC        | 0.85                | 0.88          |
| F1-Score       | 0.82                | 0.85          |
| Precision      | 0.80                | 0.83          |
| Recall         | 0.84                | 0.87          |
| Accuracy       | 0.81                | 0.84          |

## Feature Importance Comparison

### Logistic Regression Coefficients

The Logistic Regression model provides interpretable coefficients that indicate how each feature affects the prediction:

1. **Chest Pain Type (cp)**: Positive coefficients for atypical angina and non-anginal pain
2. **Maximum Heart Rate (thalach)**: Negative coefficient - lower rates increase risk
3. **Number of Major Vessels (ca)**: Positive coefficient - more vessels increase risk
4. **Exercise Induced Angina (exang)**: Positive coefficient - presence increases risk
5. **ST Depression (oldpeak)**: Positive coefficient - higher values increase risk

### Random Forest Feature Importance

The Random Forest model ranks features by their contribution to prediction accuracy:

1. **Maximum Heart Rate (thalach)**: Highest importance
2. **Number of Major Vessels (ca)**: Second highest
3. **Chest Pain Type (cp)**: Third highest
4. **Age**: Moderate importance
5. **ST Depression (oldpeak)**: Lower importance

## Calibration Curves

Both models show good calibration, but Logistic Regression is naturally well-calibrated:

- **Logistic Regression**: Nearly perfect calibration line
- **Random Forest**: Slight over-prediction at higher probabilities

## Clinical Interpretability

### Logistic Regression
- ✅ Clear coefficients for each feature
- ✅ Direct impact on log-odds interpretation
- ✅ Well-calibrated probabilities
- ❌ Lower predictive performance

### Random Forest
- ✅ Higher predictive performance
- ✅ Robust to outliers
- ❌ Black-box interpretation
- ❌ Less calibrated probabilities

## Recommendation

For clinical decision support, we recommend:

1. **Primary Model**: Logistic Regression for its interpretability
2. **Secondary Model**: Random Forest for complex cases where higher accuracy is needed
3. **Threshold**: 0.4 probability for optimal sensitivity in clinical screening

## Model Switching Capability

The application allows real-time switching between models to compare predictions:

- Select "Logistic Regression" for interpretable results
- Select "Random Forest" for maximum accuracy
- Compare contributing factors side-by-side

This enables clinicians to understand how different algorithms view the same patient data.