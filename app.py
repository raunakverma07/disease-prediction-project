import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Multi Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# Load Models
# ==========================================

try:
    diabetes_model = joblib.load("models/diabetes_model.pkl")
    diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

    breast_cancer_model = joblib.load("models/breast_cancer_model.pkl")

    heart_model = joblib.load("models/heart_model.pkl")

except:
    st.error("Model files not found. Please check models folder.")

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("🩺 Disease Prediction System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🩸 Diabetes Prediction",
        "🎗️ Breast Cancer Prediction",
        "❤️ Heart Disease Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This project uses Machine Learning models to predict diseases based on medical inputs."
)

# ==========================================
# HOME PAGE
# ==========================================

if menu == "🏠 Home":

    st.title("🩺 Multi Disease Prediction System")

    st.markdown("## Machine Learning Based Healthcare Prediction Web Application")

    # Main Image
    st.image("medical_banner.jpg", use_container_width=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("### 🩸 Diabetes Prediction")
        st.write(
            "Predict diabetes risk using medical parameters like glucose level, BMI, insulin, age and blood pressure."
        )

    with col2:
        st.error("### 🎗️ Breast Cancer Prediction")
        st.write(
            "Detect breast cancer risk using machine learning and important tumor measurement features."
        )

    with col3:
        st.warning("### ❤️ Heart Disease Prediction")
        st.write(
            "Analyze heart disease risk using health indicators like blood pressure, cholesterol and lifestyle factors."
        )

    st.markdown("---")

    st.subheader("📌 About This Project")

    st.write(
        "This project is developed using Python, Machine Learning and Streamlit. "
        "The application predicts multiple diseases using trained ML models and user medical inputs."
    )

    st.subheader("🛠️ Technologies Used")

    st.write(
        "- Python\n"
        "- Pandas\n"
        "- Scikit-learn\n"
        "- Streamlit\n"
        "- Joblib"
    )

    st.subheader("📊 Machine Learning Models")

    model_df = pd.DataFrame({
        "Disease": [
            "Diabetes",
            "Breast Cancer",
            "Heart Disease"
        ],
        "Model Used": [
            "K-Nearest Neighbors (KNN)",
            "Random Forest",
            "Random Forest"
        ],
        "Accuracy": [
            "69.48%",
            "95.61%",
            "94.09%"
        ]
    })

    st.dataframe(model_df, use_container_width=True)

    st.markdown("---")

    st.success("Developed using Machine Learning and Streamlit")

# ==========================================
# DIABETES PREDICTION
# ==========================================

elif menu == "🩸 Diabetes Prediction":

    st.title("🩸 Diabetes Prediction")

    st.info(
        "Diabetes is a chronic health condition that affects how the body converts food into energy. "
        "This prediction system uses medical information to estimate diabetes risk."
    )

    st.markdown("### 📊 Model Accuracy : 69.48%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0.0)
        glucose = st.number_input("Glucose Level", min_value=0.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0)

    with col2:
        insulin = st.number_input("Insulin", min_value=0.0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_va
