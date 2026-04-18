import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

# Title
st.title("🌫️ Air Quality Index Predictor")
st.markdown("Predict AQI using real-time pollutant values")

# Load models
@st.cache_resource
def load_models():
    rf_reg = joblib.load('aqi_rf_regressor.pkl')
    xgb_clf = joblib.load('aqi_xgb_classifier.pkl')
    label_encoder = joblib.load('aqi_label_encoder.pkl')
    scaler = joblib.load('aqi_scaler.pkl')
    return rf_reg, xgb_clf, label_encoder, scaler

rf_reg, xgb_clf, label_encoder, scaler = load_models()

# Feature columns (must match training)
feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']

# Create two tabs
tab1, tab2 = st.tabs(["📈 Predict AQI (Regression)", "📊 Predict AQI Category (Classification)"])

# ============================================
# TAB 1: Regression - Predict AQI Value
# ============================================
with tab1:
    st.header("Predict AQI Value")
    st.markdown("Enter pollutant values below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0, max_value=800, value=100, step=5)
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0, max_value=800, value=150, step=5)
        no = st.number_input("NO (µg/m³)", min_value=0, max_value=300, value=50, step=5)
        no2 = st.number_input("NO2 (µg/m³)", min_value=0, max_value=300, value=50, step=5)
    
    with col2:
        nox = st.number_input("NOx (µg/m³)", min_value=0, max_value=500, value=100, step=5)
        co = st.number_input("CO (mg/m³)", min_value=0, max_value=30, value=5, step=1)
        so2 = st.number_input("SO2 (µg/m³)", min_value=0, max_value=200, value=40, step=5)
        o3 = st.number_input("O3 (µg/m³)", min_value=0, max_value=200, value=60, step=5)
    
    # Create input array
    input_data = np.array([[pm25, pm10, no, no2, nox, co, so2, o3]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    if st.button("Predict AQI", key="reg_btn"):
        prediction = rf_reg.predict(input_scaled)[0]
        
        # Display result
        st.success(f"### 🌟 Predicted AQI: **{prediction:.1f}**")
        
        # AQI category display
        if prediction <= 50:
            st.info("🟢 **Category: Good** - Air quality is satisfactory")
        elif prediction <= 100:
            st.info("🟢 **Category: Satisfactory** - Acceptable air quality")
        elif prediction <= 200:
            st.warning("🟡 **Category: Moderate** - Sensitive groups should reduce outdoor activity")
        elif prediction <= 300:
            st.warning("🟠 **Category: Poor** - Everyone may begin to experience health effects")
        elif prediction <= 400:
            st.error("🔴 **Category: Very Poor** - Health alert: everyone may experience serious effects")
        else:
            st.error("🟤 **Category: Severe** - Health warning of emergency conditions")
    
    # Show model info
    with st.expander("ℹ️ About this model"):
        st.write("**Model:** Random Forest Regressor")
        st.write("**R² Score:** 0.92")
        st.write("**RMSE:** 37.16")
        st.write("**Features used:** PM2.5, PM10, NO, NO2, NOx, CO, SO2, O3")

# ============================================
# TAB 2: Classification - Predict AQI Category
# ============================================
with tab2:
    st.header("Predict AQI Category")
    st.markdown("Enter pollutant values below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pm25_c = st.number_input("PM2.5 (µg/m³)", min_value=0, max_value=800, value=100, step=5, key="c_pm25")
        pm10_c = st.number_input("PM10 (µg/m³)", min_value=0, max_value=800, value=150, step=5, key="c_pm10")
        no_c = st.number_input("NO (µg/m³)", min_value=0, max_value=300, value=50, step=5, key="c_no")
        no2_c = st.number_input("NO2 (µg/m³)", min_value=0, max_value=300, value=50, step=5, key="c_no2")
    
    with col2:
        nox_c = st.number_input("NOx (µg/m³)", min_value=0, max_value=500, value=100, step=5, key="c_nox")
        co_c = st.number_input("CO (mg/m³)", min_value=0, max_value=30, value=5, step=1, key="c_co")
        so2_c = st.number_input("SO2 (µg/m³)", min_value=0, max_value=200, value=40, step=5, key="c_so2")
        o3_c = st.number_input("O3 (µg/m³)", min_value=0, max_value=200, value=60, step=5, key="c_o3")
    
    # Create input array
    input_data_c = np.array([[pm25_c, pm10_c, no_c, no2_c, nox_c, co_c, so2_c, o3_c]])
    
    # Scale input
    input_scaled_c = scaler.transform(input_data_c)
    
    # Predict
    if st.button("Predict Category", key="clf_btn"):
        pred_encoded = xgb_clf.predict(input_scaled_c)[0]
        prediction_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Display result
        st.success(f"### 🌟 Predicted AQI Category: **{prediction_label}**")
        
        # Color coding
        if prediction_label == "Good":
            st.info("🟢 Air quality is satisfactory with little to no risk")
        elif prediction_label == "Satisfactory":
            st.info("🟢 Acceptable air quality with moderate health concern for sensitive groups")
        elif prediction_label == "Moderate":
            st.warning("🟡 Sensitive groups should reduce outdoor activity")
        elif prediction_label == "Poor":
            st.warning("🟠 Everyone may begin to experience health effects")
        elif prediction_label == "Very Poor":
            st.error("🔴 Health alert: everyone may experience serious effects")
        else:
            st.error("🟤 Health warning of emergency conditions")
    
    # Show model info
    with st.expander("ℹ️ About this model"):
        st.write("**Model:** XGBoost Classifier")
        st.write("**Accuracy:** 79.06%")
        st.write("**F1-Score:** 0.79")
        st.write("**Features used:** PM2.5, PM10, NO, NO2, NOx, CO, SO2, O3")

# ============================================
# Sidebar with info
# ============================================
st.sidebar.header("📊 About")
st.sidebar.markdown("""
**AQI Prediction Tool**

This app uses Machine Learning to predict:
- **AQI value** (0-500+) using Random Forest
- **AQI Category** (Good → Severe) using XGBoost

**Data Source:** CPCB (Central Pollution Control Board), India

**Model Performance:**
- Regression R²: 0.92
- Classification Accuracy: 79%
""")

st.sidebar.header("📁 Feature Descriptions")
st.sidebar.markdown("""
- **PM2.5** - Particulate matter < 2.5μm
- **PM10** - Particulate matter < 10μm
- **NO, NO2, NOx** - Nitrogen oxides
- **CO** - Carbon monoxide
- **SO2** - Sulfur dioxide
- **O3** - Ozone
""")