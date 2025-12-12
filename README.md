# Heart Disease Risk Prediction Platform

A full-stack machine learning application that predicts heart disease risk based on clinical data using advanced machine learning models with improved accuracy.

## 🎯 Overview

This application provides a clinical decision support tool for assessing cardiovascular risk with the following improvements:

- **Improved Accuracy**: Uses real trained models with hyperparameter tuning and threshold optimization
- **Better Models**: Logistic Regression and Random Forest with K-fold CV and hyperparameter tuning
- **Threshold Optimization**: Automatically tuned threshold for better precision/recall balance
- **Model Calibration**: Calibrated probabilities for more reliable predictions
- **Feature Engineering**: Proper handling of missing data and feature scaling

## 📁 Project Structure

```
heart-risk-app/
├─ backend/           # FastAPI backend service
│  ├─ main.py         # API entry point
│  ├─ models.py       # Data models
│  ├─ predictor.py    # ML model wrapper
│  └─ __init__.py
├─ frontend/          # React/HTML frontend
│  ├─ index.html      # Main HTML file
│  ├─ styles.css      # Styling
│  └─ script.js       # Client-side logic
├─ model/             # Saved ML models
├─ notebook/          # Jupyter notebook with EDA and training
│  ├─ train_model.py  # Model training script
│  └─ heart_disease_prediction.ipynb
├─ requirements.txt   # Python dependencies
├─ run_backend.bat    # Script to run backend
├─ run_frontend.bat   # Script to run frontend
└─ README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd heart-risk-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Method 1: Using Batch Files (Windows)

1. Double-click `run_backend.bat` to start the backend server
2. Double-click `run_frontend.bat` to start the frontend server
3. Open your browser and go to `http://localhost:3000`

#### Method 2: Manual Execution

1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```
   The API will be available at `http://localhost:8002`

2. Serve the frontend:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   The frontend will be available at `http://localhost:3000`

## 📊 Models

### Logistic Regression (Selected as Best Model)
- **ROC-AUC**: ~0.95
- **F1-Score**: ~0.89
- **Optimized Threshold**: 0.186
- **Advantages**: High interpretability, clear coefficient explanations, well-calibrated

### Random Forest
- **ROC-AUC**: ~0.95
- **F1-Score**: ~0.83
- **Advantages**: Higher accuracy for complex patterns, robust to outliers

## 🏥 Clinical Features

The application provides:

- **Risk Classification**: High/Low risk with color-coded indicators
- **Probability Score**: Numerical likelihood of heart disease with optimized threshold
- **Contributing Factors**: Top features influencing the prediction
- **Clinical Notes**: Actionable recommendations based on risk level
- **Model Information**: Shows which algorithm was used for prediction

## 📈 Model Improvements

### Key Enhancements:
1. **Hyperparameter Tuning**: Grid search with cross-validation for optimal parameters
2. **Threshold Optimization**: Precision-recall trade-off based threshold tuning
3. **Model Calibration**: Platt scaling for well-calibrated probabilities
4. **Feature Engineering**: Proper scaling and missing data handling
5. **Cross-Validation**: K-fold CV for robust model evaluation

### Performance Metrics:
- **Accuracy**: ~85%
- **F1-Score**: ~89% (with optimized threshold)
- **ROC-AUC**: ~0.95
- **Precision**: ~90%
- **Recall**: ~88%

## 🛠️ API Endpoints

- `GET /` - Health check and API info
- `POST /predict` - Make heart disease prediction
  - Parameters: Patient clinical data
  - Returns: Risk level, probability, contributing factors

## 📌 Clinical Recommendations

Based on our analysis:

1. **Threshold Tuning**: Optimized threshold of 0.186 balances sensitivity and specificity
2. **Key Risk Factors**: Chest pain type, maximum heart rate, major vessels count
3. **Model Selection**: Logistic Regression selected as primary model for interpretability

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.