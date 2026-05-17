# disease-prediction-project
Multi Disease Prediction System using Machine Learning and Streamlit
# Multi Disease Prediction System

This project is a Machine Learning based web application that predicts multiple diseases using medical data and user inputs.

## Diseases Included
- Diabetes Prediction
- Breast Cancer Prediction
- Heart Disease Prediction

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Models

| Disease | Model Used |
|----------|-------------|
| Diabetes | K-Nearest Neighbors (KNN) |
| Breast Cancer | Random Forest Classifier |
| Heart Disease | Random Forest Classifier |

## Features
- Interactive Streamlit Web App
- Multiple Disease Prediction
- Clean User Interface
- Fast Predictions
- Machine Learning Integration

## Project Structure

```plaintext
disease-prediction-project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── breast_cancer_model.pkl
│   └── heart_model.pkl
│
└── datasets/
    ├── diabetes.csv
    ├── breast_cancer.csv
    └── heart.csv
