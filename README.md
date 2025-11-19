# Heart Disease Risk Prediction Application

A full-stack machine learning application that predicts heart disease risk based on clinical data using Logistic Regression and Random Forest models.

## 🎯 Overview

This application provides a clinical decision support tool for assessing cardiovascular risk. It features:

- Two machine learning models: Logistic Regression and Random Forest
- Clean, medical-style user interface
- Real-time risk prediction with explainability insights
- Model comparison and performance metrics

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
│  └─ heart_disease_prediction.ipynb
├─ requirements.txt   # Python dependencies
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

1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```
   The API will be available at `http://localhost:8000`

2. Serve the frontend:
   You can use any static file server. For example, with Python:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   The frontend will be available at `http://localhost:3000`

## 📊 Models

### Logistic Regression
- Provides interpretable coefficients
- Good for understanding feature impact
- Well-calibrated probabilities

### Random Forest
- Higher predictive accuracy
- Handles non-linear relationships
- Feature importance ranking

## 🏥 Clinical Features

The application provides:

- **Risk Classification**: High/Low risk with color-coded indicators
- **Probability Score**: Numerical likelihood of heart disease
- **Contributing Factors**: Top features influencing the prediction
- **Clinical Notes**: Actionable recommendations based on risk level
- **Model Comparison**: Switch between algorithms to compare results

## 📈 Notebook Analysis

The Jupyter notebook includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Model training and hyperparameter tuning
- Performance comparison (ROC-AUC, F1-score, etc.)
- Feature importance visualization
- Calibration analysis
- Threshold tuning recommendations

## 🛠️ API Endpoints

- `GET /` - Health check and API info
- `POST /predict` - Make heart disease prediction
  - Parameters: Patient clinical data
  - Returns: Risk level, probability, contributing factors

## 📌 Clinical Recommendations

Based on our analysis:

1. **Threshold Tuning**: A threshold of 0.3-0.4 balances sensitivity and specificity
2. **Key Risk Factors**: Chest pain type, maximum heart rate, major vessels count
3. **Workload Impact**: Lower thresholds increase detection but also false positives

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.