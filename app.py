
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from PIL import Image
import io
import base64
from datetime import datetime
import shutil
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append('utils')
sys.path.append('.')

try:
    from utils.ocr_utils import OCRProcessor, validate_parameters, NORMAL_RANGES
    from utils.report_generator import ReportGenerator
    from model_training import load_model
except ImportError:
    from ocr_utils import OCRProcessor, validate_parameters, NORMAL_RANGES
    from report_generator import ReportGenerator
    from model_training import load_model

st.set_page_config(
    page_title="🏥 Cardiology Assessment Center - Heart Disease Prediction | By Shazim Javed",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/shazimjaved/heart-disease-prediction',
        'Report a bug': "https://github.com/shazimjaved/heart-disease-prediction/issues",
        'About': "# Heart Disease Prediction System\nDeveloped by SHAZIM JAVED\nAI Solutions Expert\nPowered by Advanced Machine Learning"
    }
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 0;
        min-height: 100vh;
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 25% 25%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(118, 75, 162, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    .header-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #a8d8ff;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .parameter-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        transition: all 0.3s ease;
    }
    
    .parameter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 0.9rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
        animation: pulse 2s infinite;
        margin: 0.5rem 0;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 0.9rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,167,38,0.3);
        margin: 0.5rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #66bb6a, #4caf50);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 0.9rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(76,175,80,0.3);
        margin: 0.5rem 0;
    }
    
    .risk-high h3, .risk-moderate h3, .risk-low h3 {
        font-size: 0.9rem !important;
        margin: 0 0 0.3rem 0 !important;
    }
    
    .risk-high h1, .risk-moderate h1, .risk-low h1 {
        font-size: 1.5rem !important;
        margin: 0.2rem 0 !important;
    }
    
    .risk-high h2, .risk-moderate h2, .risk-low h2 {
        font-size: 1rem !important;
        margin: 0.2rem 0 !important;
    }
    
    .risk-high p, .risk-moderate p, .risk-low p {
        font-size: 0.75rem !important;
        margin: 0.2rem 0 0 0 !important;
        opacity: 0.9;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .css-1d391kg .block-container {
        background: rgba(26, 26, 46, 0.8);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .download-btn {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        border-radius: 30px !important;
        padding: 1rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        box-shadow: 0 6px 20px rgba(40,167,69,0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-container {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stDataFrame {
        background-color: rgba(30, 30, 30, 0.95) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame > div {
        background-color: rgba(30, 30, 30, 0.95) !important;
        border-radius: 10px !important;
    }
    
    .stDataFrame table {
        background-color: rgba(30, 30, 30, 0.95) !important;
        color: white !important;
    }
    
    .stDataFrame thead th {
        background-color: rgba(20, 20, 20, 0.98) !important;
        color: #FFD700 !important;
        font-weight: bold !important;
        border-bottom: 2px solid #667eea !important;
    }
    
    .stDataFrame tbody td {
        background-color: rgba(30, 30, 30, 0.95) !important;
        color: white !important;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    .stDataFrame tbody tr:hover {
        background-color: rgba(102, 126, 234, 0.2) !important;
    }
    
    div[data-testid="stDataFrame"] {
        background: rgba(30, 30, 30, 0.95);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(30, 30, 30, 0.95) !important;
        color: white !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: rgba(30, 30, 30, 0.95) !important;
        color: white !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox > div > div > div:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(30, 30, 30, 0.95) !important;
        color: white !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stNumberInput > label,
    .stSelectbox > label,
    .stTextInput > label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(30, 30, 30, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
    }
    
    .stForm {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stSuccess {
        background-color: rgba(46, 204, 113, 0.2) !important;
        color: #2ecc71 !important;
        border: 1px solid #2ecc71 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background-color: rgba(231, 76, 60, 0.2) !important;
        color: #e74c3c !important;
        border: 1px solid #e74c3c !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background-color: rgba(255, 167, 38, 0.2) !important;
        color: #ffa726 !important;
        border: 1px solid #ffa726 !important;
        border-radius: 8px !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #e57373;
    }
    .risk-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #81c784;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_ml_model():
    model, scaler, feature_names = load_model()
    return model, scaler, feature_names

def main():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0f3460 0%, #16213e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center; border: 1px solid rgba(102, 126, 234, 0.3); box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
        <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🏥 CARDIOLOGY ASSESSMENT CENTER
        </h2>
        <p style="color: #a8d8ff; margin: 0.3rem 0; font-size: 0.9rem;">
            ❤️ AI-Powered Heart Disease Risk Analysis
        </p>
        <div style="margin-top: 0.5rem;">
            <span style="background: rgba(255,215,0,0.2); padding: 0.2rem 0.8rem; border-radius: 12px; color: #FFD700; font-size: 0.75rem; font-weight: bold; border: 1px solid rgba(255,215,0,0.3);">
                👨‍💻 By SHAZIM JAVED | AI Healthcare Expert
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.8rem; background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(102, 126, 234, 0.3);">
            <h3 style="color: white; margin: 0; font-size: 1.2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">🩺 Navigation</h3>
            <p style="color: #a8d8ff; margin: 0.2rem 0 0 0; font-size: 0.8rem;">Choose assessment method</p>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "📋 Select Input Method:",
            ["✍️ Manual Form Input", "📷 Image Upload", "ℹ️ About System"],
            index=0
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: white; margin-top: 0;">📊 System Stats</h4>
            <p style="color: #a8d8ff; font-size: 0.8rem; margin: 0.5rem 0;">
                🎯 Accuracy: 95%+<br>
                🏥 Models: 3 Advanced ML<br>
                📈 Predictions: Real-time<br>
                🔒 Privacy: 100% Secure
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,215,0,0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #FFD700;">
            <h4 style="color: #FFD700; margin-top: 0;">👨‍💻 Developer</h4>
            <p style="color: white; font-size: 0.8rem; margin: 0.5rem 0;">
                <strong>SHAZIM JAVED</strong><br>
                🏆 AI Solutions Expert<br>
                💻 Machine Learning Engineer<br>
                🩺 Medical AI Specialist
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: rgba(220,20,60,0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #dc143c;">
            <h4 style="color: #ff6b6b; margin-top: 0;">🚨 Emergency</h4>
            <p style="color: white; font-size: 0.8rem; margin: 0;">
                If experiencing chest pain or emergency symptoms, call 1122 immediately.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "ℹ️ About System":
        show_about_page()
    elif page == "✍️ Manual Form Input":
        show_manual_input_page()
    elif page == "📷 Image Upload":
        show_image_upload_page()

def show_about_page():
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; font-size: 1.6rem;">ℹ️ About This System</h3>
        <p style="color: #a8d8ff; margin: 0.3rem 0 0 0; font-size: 0.85rem;">AI-powered cardiac risk assessment platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
    ## 🏥 System Overview
    This Heart Disease Prediction System uses advanced machine learning algorithms to assess cardiac risk based on comprehensive health parameters.
    
    ### ✨ Key Features:
    - **📋 Manual Input**: Enter health parameters directly through an intuitive form
    - **📷 Image Upload**: Upload medical reports and extract parameters using OCR technology
    - **🎯 Risk Assessment**: Get accurate predictions using multiple trained ML models
    - **📄 PDF Reports**: Generate comprehensive medical reports with visualizations
    - **📊 Interactive Charts**: Real-time data visualization with Plotly
    
    ### 🚀 Technology Stack:
    - **🤖 Machine Learning**: Random Forest, Logistic Regression, XGBoost
    - **🌐 Web Framework**: Streamlit with custom CSS styling
    - **👁️ OCR Technology**: Tesseract for image text extraction
    - **📄 PDF Generation**: ReportLab for professional reports
    - **📊 Data Processing**: Pandas, NumPy, Plotly
    - **🎨 UI/UX**: Custom CSS with medical-grade styling
    
    ### 📈 Model Performance:
    - **🎯 Accuracy**: 95%+ on validation data
    - **📚 Training Data**: Cleveland Heart Disease Dataset (303 patients)
    - **🔢 Features**: 13 comprehensive clinical parameters
    - **✅ Validation**: 5-fold cross-validation for robust performance
    - **⚡ Speed**: Real-time predictions in milliseconds
    
    ### 🩺 Clinical Parameters Used:
    - **👤 Demographics**: Age, Gender
    - **💔 Symptoms**: Chest Pain Type, Exercise Induced Angina
    - **🩸 Vital Signs**: Blood Pressure, Heart Rate, Blood Sugar
    - **🔬 Lab Tests**: Cholesterol, Thalassemia
    - **📊 Cardiac Tests**: ECG Results, ST Depression, Slope, Major Vessels
    
    ### ⚠️ Medical Disclaimer:
    This system is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
    """)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="color: white; margin-top: 0;">📊 Clinical Normal Ranges</h3>
    </div>
    """, unsafe_allow_html=True)
    
    ranges_df = pd.DataFrame([
        {"Parameter": "Age", "Normal Range": "18-100 years", "Unit": "years"},
        {"Parameter": "Blood Pressure", "Normal Range": "90-120", "Unit": "mmHg"},
        {"Parameter": "Cholesterol", "Normal Range": "100-200", "Unit": "mg/dL"},
        {"Parameter": "Max Heart Rate", "Normal Range": "60-180", "Unit": "bpm"},
        {"Parameter": "Fasting Blood Sugar", "Normal Range": "70-100", "Unit": "mg/dL"}
    ])
    
    st.markdown("""
    <div style="background: rgba(30, 30, 30, 0.95); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    
    st.dataframe(ranges_df, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_manual_input_page():
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; font-size: 1.5rem;">📋 Patient Assessment Form</h3>
        <p style="color: #a8d8ff; margin: 0.3rem 0 0 0; font-size: 0.85rem;">Enter health parameters for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, scaler, feature_names = load_ml_model()
    
    if model is None:
        st.error("Model not found. Please run model training first.")
        return
    
    with st.form("health_form"):
        st.subheader("👤 Patient Information")
        col_patient = st.columns(2)
        
        with col_patient[0]:
            patient_name = st.text_input("Patient Name", placeholder="Enter patient's full name")
        
        with col_patient[1]:
            patient_id = st.text_input("Patient ID (Optional)", placeholder="Enter patient ID or leave blank")
        
        st.markdown("---")
        
        st.subheader("🏥 Health Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=50)
            sex = st.selectbox("Sex", ["Female", "Male"])
            cp = st.selectbox("Chest Pain Type", 
                            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG Results", 
                                 ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        
        with col2:
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                               ["Upsloping", "Flat", "Downsloping"])
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", 
                            ["0", "1", "2", "3"])
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        
        submitted = st.form_submit_button("Predict Heart Disease Risk")
    
    if submitted:
        if not patient_name.strip():
            st.error("⚠️ Please enter a patient name")
            return
        
        user_data = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,
            'cp': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs == "Yes" else 0,
            'restecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
            'thalach': thalach,
            'exang': 1 if exang == "Yes" else 0,
            'oldpeak': oldpeak,
            'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
            'ca': int(ca),
            'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        }
        
        patient_info = {
            'patient_name': patient_name.strip(),
            'patient_id': patient_id.strip() if patient_id.strip() else "N/A"
        }
        
        prediction_result, prediction_probability = make_prediction(user_data, model, scaler, feature_names)
        
        display_enhanced_prediction_results(user_data, prediction_result, prediction_probability, patient_info)

def show_image_upload_page():
    st.markdown('<h2 class="sub-header">Upload Medical Report Image</h2>', unsafe_allow_html=True)
    
    model, scaler, feature_names = load_ml_model()
    
    if model is None:
        st.error("Model not found. Please run model training first.")
        return
    st.subheader("👤 Patient Information")
    col_patient = st.columns(2)
    
    with col_patient[0]:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient's full name", key="upload_patient_name")
    
    with col_patient[1]:
        patient_id = st.text_input("Patient ID (Optional)", placeholder="Enter patient ID or leave blank", key="upload_patient_id")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Extract Parameters and Predict"):
            if not patient_name.strip():
                st.error("⚠️ Please enter a patient name")
                return
            
            with st.spinner("Processing image and extracting parameters..."):
                ocr_processor = OCRProcessor()
                
                if not ocr_processor.test_ocr():
                    st.error("❌ OCR not available. Please install Tesseract OCR.")
                    return
                
                result = ocr_processor.process_image_file(uploaded_file)
                
                if result['success']:
                    st.success(result['message'])
                    
                    with st.expander("Extracted Text"):
                        st.text(result['extracted_text'])
                    
                    if result['parameters']:
                        st.subheader("Extracted Parameters")
                        params_df = pd.DataFrame([
                            {"Parameter": k.replace('_', ' ').title(), "Value": v} 
                            for k, v in result['parameters'].items()
                        ])
                        st.dataframe(params_df, use_container_width=True)
                        
                        validated_params, warnings = validate_parameters(result['parameters'])
                        
                        if warnings:
                            st.warning("Parameter Warnings:")
                            for warning in warnings:
                                st.warning(warning)
                        
                        if len(validated_params) >= 5:  # Minimum required parameters
                            prediction_result, prediction_probability = make_prediction(
                                validated_params, model, scaler, feature_names
                            )
                            
                            patient_info = {
                                'patient_name': patient_name.strip(),
                                'patient_id': patient_id.strip() if patient_id.strip() else "N/A"
                            }
                            
                            display_prediction_results(validated_params, prediction_result, prediction_probability, patient_info)
                        else:
                            st.error("Insufficient parameters extracted. Please try with a clearer image or use manual input.")
                    else:
                        st.error("No parameters could be extracted from the image. Please try with a clearer image or use manual input.")
                else:
                    st.error(result['message'])
                    
                    if result['extracted_text']:
                        st.subheader("🔍 Debug Information")
                        st.info("Text was extracted but no parameters were found. Here's what was extracted:")
                        with st.expander("Extracted Text (Debug)"):
                            st.text(result['extracted_text'])
                        
                        st.warning("💡 **Tips for better OCR results:**")
                        st.markdown("""
                        - Use high-resolution images
                        - Ensure good lighting and contrast
                        - Use clear, printed text (not handwritten)
                        - Try different image formats (PNG, JPG)
                        - Make sure text is not blurry or rotated
                        """)
                    else:
                        st.error("No text could be extracted from the image. Please try with a clearer image.")

def make_prediction(user_data, model, scaler, feature_names):
    try:
        feature_vector = []
        for feature in feature_names:
            if feature in user_data:
                feature_vector.append(user_data[feature])
            else:
                feature_vector.append(0)
        
        X = np.array(feature_vector).reshape(1, -1)
        
        import pandas as pd
        X_df = pd.DataFrame(X, columns=feature_names)
        
        X_scaled = scaler.transform(X_df)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def display_enhanced_prediction_results(user_data, prediction_result, prediction_probability, patient_info=None):
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin: 1rem 0;">
        <h3 style="color: white; margin: 0; font-size: 1.8rem;">🎯 AI Analysis Results</h3>
        <p style="color: #a8d8ff; margin: 0.3rem 0 0 0; font-size: 0.85rem;">Machine learning assessment completed</p>
    </div>
    """, unsafe_allow_html=True)
    
    risk_level = "High" if prediction_probability > 0.7 else "Moderate" if prediction_probability > 0.4 else "Low"
    risk_color = "risk-high" if prediction_probability > 0.7 else "risk-moderate" if prediction_probability > 0.4 else "risk-low"
    
    st.markdown(f"""
    <div class="{risk_color}">
        <h3>🫀 CARDIAC RISK ASSESSMENT</h3>
        <h1>{prediction_probability:.1%}</h1>
        <h2>{risk_level.upper()} RISK</h2>
        <p>Probability of heart disease based on current health parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'High Risk'],
            values=[1-prediction_probability, prediction_probability],
            hole=.3,
            marker_colors=['#2ecc71', '#e74c3c'],
            textinfo='label+percent',
            textfont_size=14,
            marker_line=dict(color='#ffffff', width=3)
        )])
        
        fig_pie.update_layout(
            title={
                'text': '🥧 Risk Distribution',
                'x': 0.5,
                'font': {'size': 18, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "🎯 Risk Score", 'font': {'size': 18, 'color': 'white'}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(46, 204, 113, 0.3)"},
                    {'range': [40, 70], 'color': "rgba(255, 167, 38, 0.3)"},
                    {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"},
            height=300
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
        <h3 style="color: white; margin-top: 0;">📊 Health Parameters Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    param_names = []
    param_values = []
    param_normals = []
    param_colors = []
    
    params_info = {
        'age': {'name': 'Age', 'normal': 50, 'unit': 'years'},
        'trestbps': {'name': 'Blood Pressure', 'normal': 120, 'unit': 'mmHg'},
        'chol': {'name': 'Cholesterol', 'normal': 200, 'unit': 'mg/dL'},
        'thalach': {'name': 'Max Heart Rate', 'normal': 150, 'unit': 'bpm'},
        'oldpeak': {'name': 'ST Depression', 'normal': 1, 'unit': 'mm'}
    }
    
    for param, info in params_info.items():
        if param in user_data:
            param_names.append(info['name'])
            param_values.append(user_data[param])
            param_normals.append(info['normal'])
            
            if param == 'age':
                color = '#3498db'
            elif param == 'trestbps':
                color = '#e74c3c' if user_data[param] > 140 or user_data[param] < 90 else '#2ecc71'
            elif param == 'chol':
                color = '#e74c3c' if user_data[param] > 240 else '#f39c12' if user_data[param] > 200 else '#2ecc71'
            elif param == 'thalach':
                color = '#f39c12' if user_data[param] < 100 else '#2ecc71'
            else:
                color = '#e74c3c' if user_data[param] > 2 else '#2ecc71'
            
            param_colors.append(color)
    
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        name='Your Values',
        x=param_names,
        y=param_values,
        marker_color=param_colors,
        text=[f'{val}' for val in param_values],
        textposition='auto',
    ))
    
    fig_bar.add_trace(go.Scatter(
        name='Normal Reference',
        x=param_names,
        y=param_normals,
        mode='markers+lines',
        line=dict(color='white', width=2, dash='dash'),
        marker=dict(color='white', size=8, symbol='diamond')
    ))
    
    fig_bar.update_layout(
        title={
            'text': '📈 Parameter Comparison vs Normal Values',
            'x': 0.5,
            'font': {'size': 18, 'color': 'white'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    display_prediction_results(user_data, prediction_result, prediction_probability, patient_info)

def display_prediction_results(user_data, prediction_result, prediction_probability, patient_info=None):
    if prediction_result is None:
        return
    
    if patient_info:
        _ = st.subheader("👤 Patient Information")
        col_patient_display = st.columns(2)
        
        with col_patient_display[0]:
            patient_name = patient_info['patient_name'] if patient_info else 'N/A'
            _ = st.info(f"**Patient Name:** {patient_name}")
        
        with col_patient_display[1]:
            patient_id = patient_info['patient_id'] if patient_info else 'N/A'
            _ = st.info(f"**Patient ID:** {patient_id}")
        
        _ = st.markdown("---")
    
    risk_level = "HIGH RISK" if prediction_result == 1 else "LOW RISK"
    risk_class = "risk-high" if prediction_result == 1 else "risk-low"
    
    _ = st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
    _ = st.markdown(f"## 🎯 Risk Assessment: {risk_level}")
    _ = st.markdown(f"**Risk Probability: {prediction_probability:.2%}**")
    _ = st.markdown('</div>', unsafe_allow_html=True)
    
    
    _ = st.session_state.update({
        'user_data': user_data,
        'prediction_result': prediction_result,
        'prediction_probability': prediction_probability,
        'patient_info': patient_info
    })
    
    if 'user_data' in st.session_state and 'prediction_result' in st.session_state:
        user_data = st.session_state['user_data']
        prediction_result = st.session_state['prediction_result']
        prediction_probability = st.session_state['prediction_probability']
        patient_info = st.session_state.get('patient_info', None)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_text = f"""HEART DISEASE RISK ASSESSMENT REPORT
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

PATIENT INFORMATION:
Name: {patient_info['patient_name'] if patient_info else 'N/A'}
ID: {patient_info['patient_id'] if patient_info else 'N/A'}

RISK ASSESSMENT:
Risk Level: {'HIGH RISK' if prediction_result == 1 else 'LOW RISK'}
Risk Probability: {prediction_probability:.2%}

HEALTH PARAMETERS:"""
        
        for param, value in user_data.items():
            param_name = param.replace('_', ' ').title()
            report_text += f"\n{param_name}: {value}"
        
        report_text += """

RECOMMENDATIONS:
• Consult with a healthcare professional for detailed analysis
• Maintain a healthy diet and regular exercise routine
• Monitor blood pressure and cholesterol levels regularly
• Avoid smoking and excessive alcohol consumption
• Get regular health check-ups"""

        if prediction_result == 1:
            report_text = report_text.replace("RECOMMENDATIONS:", "RECOMMENDATIONS:\n• HIGH RISK: Immediate consultation with a cardiologist recommended")
        
        report_text += """

DISCLAIMER:
This report is for informational purposes only and should not replace 
professional medical advice, diagnosis, or treatment. Always consult with 
qualified healthcare professionals for medical decisions."""

        _ = st.markdown("### 📋 Report Preview")
        _ = st.text_area("Report Content", report_text, height=400, disabled=True)
        
        
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            import io
            
            buffer = io.BytesIO()
            pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            
            pdf_canvas.setFillColorRGB(1.0, 1.0, 1.0)
            pdf_canvas.rect(0, height - 80, width, 80, fill=1)
            
            y_position = height - 30
            
            pdf_canvas.setFont("Helvetica-Bold", 18)
            pdf_canvas.setFillColorRGB(0.2, 0.2, 0.2)
            pdf_canvas.drawString(50, y_position, "CARDIOLOGY ASSESSMENT CENTER")
            
            y_position -= 20
            pdf_canvas.setFont("Helvetica", 12)
            pdf_canvas.setFillColorRGB(0.4, 0.4, 0.4)
            pdf_canvas.drawString(50, y_position, "Heart Disease Risk Analysis Report")
            
            pdf_canvas.setFont("Helvetica", 9)
            pdf_canvas.setFillColorRGB(0.5, 0.5, 0.5)
            pdf_canvas.drawString(width - 200, height - 30, f"Date: {datetime.now().strftime('%B %d, %Y')}")
            pdf_canvas.drawString(width - 200, height - 40, f"Patient ID: HD-{hash(str(user_data)) % 10000:04d}")
            pdf_canvas.drawString(width - 200, height - 50, f"Risk Level: {'HIGH' if prediction_probability > 0.7 else 'MODERATE' if prediction_probability > 0.4 else 'LOW'}")
            
            y_position -= 15
            pdf_canvas.setStrokeColorRGB(0.8, 0.8, 0.8)
            pdf_canvas.setLineWidth(1)
            pdf_canvas.line(50, y_position, width - 50, y_position)
            
            y_position -= 30
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.setFillColorRGB(0.2, 0.2, 0.2)
            pdf_canvas.drawString(50, y_position, "PATIENT INFORMATION")
            
            y_position -= 20
            pdf_canvas.setFont("Helvetica", 10)
            pdf_canvas.setFillColorRGB(0.3, 0.3, 0.3)
            pdf_canvas.drawString(50, y_position, f"Name: {patient_info['patient_name'] if patient_info else 'N/A'}")
            
            y_position -= 15
            pdf_canvas.drawString(50, y_position, f"ID: {patient_info['patient_id'] if patient_info else 'N/A'}")
            
            y_position -= 30
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.setFillColorRGB(0.2, 0.2, 0.2)
            pdf_canvas.drawString(50, y_position, "RISK ASSESSMENT")
            
            y_position -= 20
            pdf_canvas.setFont("Helvetica", 10)
            pdf_canvas.setFillColorRGB(0.3, 0.3, 0.3)
            risk_text = f"Risk Level: {'HIGH RISK' if prediction_result == 1 else 'LOW RISK'}"
            pdf_canvas.drawString(50, y_position, risk_text)
            
            y_position -= 15
            pdf_canvas.drawString(50, y_position, f"Risk Probability: {prediction_probability:.2%}")
            
            y_position -= 30
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.setFillColorRGB(0.2, 0.2, 0.2)
            pdf_canvas.drawString(50, y_position, "RISK VISUALIZATION")
            
            y_position -= 80
           
            center_x, center_y = 150, y_position
            radius = 50
            
            risk_percentage = prediction_probability
            safe_percentage = 1 - risk_percentage
            
            import math
            
            
            pdf_canvas.setStrokeColorRGB(0.7, 0.7, 0.7)
            pdf_canvas.setLineWidth(1)
            pdf_canvas.circle(center_x, center_y, radius, fill=0)
            
            
            start_angle = 90
            
            if safe_percentage > 0:
                green_angle = 360 * safe_percentage
                
            
                path = pdf_canvas.beginPath()
                path.moveTo(center_x, center_y)
                
                
                num_segments = max(int(green_angle / 5), 8)
                for i in range(num_segments + 1):
                    angle_rad = math.radians(start_angle - (green_angle * i / num_segments))
                    x = center_x + radius * math.cos(angle_rad)
                    y = center_y + radius * math.sin(angle_rad)
                    path.lineTo(x, y)
                
                path.lineTo(center_x, center_y)
                pdf_canvas.setFillColorRGB(0.2, 0.7, 0.2)  # Clean green
                pdf_canvas.drawPath(path, fill=1)
            
            # Draw RED slice (High Risk)
            if risk_percentage > 0:
                red_start = start_angle - (360 * safe_percentage)
                red_angle = 360 * risk_percentage
                
                # Create path for red slice
                path = pdf_canvas.beginPath()
                path.moveTo(center_x, center_y)
                
                # Draw arc using multiple line segments
                num_segments = max(int(red_angle / 5), 8)
                for i in range(num_segments + 1):
                    angle_rad = math.radians(red_start - (red_angle * i / num_segments))
                    x = center_x + radius * math.cos(angle_rad)
                    y = center_y + radius * math.sin(angle_rad)
                    path.lineTo(x, y)
                
                path.lineTo(center_x, center_y)
                pdf_canvas.setFillColorRGB(0.8, 0.2, 0.2)  # Clean red
                pdf_canvas.drawPath(path, fill=1)
            
            # Add center dot
            pdf_canvas.setFillColorRGB(1, 1, 1)
            pdf_canvas.circle(center_x, center_y, 5, fill=1)
            pdf_canvas.setStrokeColorRGB(0.3, 0.3, 0.3)
            pdf_canvas.setLineWidth(1)
            pdf_canvas.circle(center_x, center_y, 5, fill=0)
            
            # Simple legend
            pdf_canvas.setFillColorRGB(0, 0, 0)
            pdf_canvas.setFont("Helvetica", 9)
            
            legend_x = center_x + radius + 20
            
            # Green legend
            pdf_canvas.setFillColorRGB(0.2, 0.7, 0.2)
            pdf_canvas.rect(legend_x - 10, center_y + 10, 8, 8, fill=1)
            pdf_canvas.setFillColorRGB(0, 0, 0)
            pdf_canvas.drawString(legend_x, center_y + 12, f"Low Risk: {(1-prediction_probability):.1%}")
            
            # Red legend
            pdf_canvas.setFillColorRGB(0.8, 0.2, 0.2)
            pdf_canvas.rect(legend_x - 10, center_y - 5, 8, 8, fill=1)
            pdf_canvas.setFillColorRGB(0, 0, 0)
            pdf_canvas.drawString(legend_x, center_y - 3, f"High Risk: {prediction_probability:.1%}")
            
            # Clean border
            pdf_canvas.setStrokeColorRGB(0.3, 0.3, 0.3)
            pdf_canvas.setLineWidth(1)
            pdf_canvas.circle(center_x, center_y, radius, fill=0)
            
            y_position -= 80
            
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.setFillColorRGB(0.2, 0.2, 0.2)
            pdf_canvas.drawString(50, y_position, "HEALTH PARAMETERS")
            
            y_position -= 20
            pdf_canvas.setFont("Helvetica", 10)
            pdf_canvas.setFillColorRGB(0.3, 0.3, 0.3)
            
            normal_ranges = {
                'age': (18, 65),
                'trestbps': (90, 120),
                'chol': (100, 200),
                'thalach': (60, 100)
            }
            
            for param, value in user_data.items():
                param_name = param.replace('_', ' ').title()
                status = ""
                
                if param == 'sex':
                    display_value = "Male" if value == 1 else "Female"
                    pdf_canvas.drawString(50, y_position, f"{param_name}: {display_value}")
                else:
                    if param in normal_ranges:
                        min_val, max_val = normal_ranges[param]
                        if min_val <= value <= max_val:
                            status = " ✓ Normal"
                        else:
                            status = " ⚠ Outside Range"
                    
                    pdf_canvas.drawString(50, y_position, f"{param_name}: {value}{status}")
                
                y_position -= 15
                
                if y_position < 100:
                    pdf_canvas.showPage()
                    y_position = height - 50
                    pdf_canvas.setFont("Helvetica", 10)
            
            if y_position < 150:
                pdf_canvas.showPage()
                y_position = height - 50
            
            y_position -= 30
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.drawString(50, y_position, "DISCLAIMER:")
            
            y_position -= 20
            pdf_canvas.setFont("Helvetica", 10)
            disclaimer_lines = [
                "This report is for informational purposes only and should not replace",
                "professional medical advice, diagnosis, or treatment. Always consult",
                "with qualified healthcare professionals for medical decisions."
            ]
            for line in disclaimer_lines:
                pdf_canvas.drawString(50, y_position, line)
                y_position -= 15
            
            if y_position < 200:
                pdf_canvas.showPage()
                y_position = height - 50
            
            y_position -= 40
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.setFillColorRGB(0.2, 0.2, 0.2)
            pdf_canvas.drawString(50, y_position, "RECOMMENDATIONS")
            
            y_position -= 30
            pdf_canvas.setFont("Helvetica-Bold", 12)
            pdf_canvas.setFillColorRGB(0, 0, 0)
            
            recommendations = []
            
            if prediction_probability > 0.7:
                recommendations.extend([
                    "HIGH RISK - IMMEDIATE ACTION REQUIRED:",
                    "• Schedule urgent cardiology consultation within 1-2 weeks",
                    "• Consider stress test and cardiac catheterization",
                    "• Start aggressive lifestyle modifications immediately",
                    "• Monitor blood pressure daily"
                ])
            elif prediction_probability > 0.4:
                recommendations.extend([
                    "MODERATE RISK - PROACTIVE MEASURES NEEDED:",
                    "• Schedule cardiology consultation within 1 month",
                    "• Consider cardiac stress test",
                    "• Implement structured exercise program",
                    "• Regular monitoring of vital parameters"
                ])
            else:
                recommendations.extend([
                    "LOW RISK - MAINTAIN HEALTHY LIFESTYLE:",
                    "• Continue current healthy habits",
                    "• Annual cardiac screening recommended",
                    "• Maintain regular exercise routine",
                    "• Monitor parameters quarterly"
                ])
            
            recommendations.append("")
            recommendations.append("PARAMETER-SPECIFIC RECOMMENDATIONS:")
            
            if user_data.get('age', 0) > 60:
                recommendations.append("• Age 60+: Increase cardiac screening frequency")
            elif user_data.get('age', 0) < 30:
                recommendations.append("• Young age: Focus on prevention and healthy habits")
            
            if user_data.get('trestbps', 0) > 140:
                recommendations.extend([
                    "• HIGH BP: Reduce sodium intake (<2300mg/day)",
                    "• Consider antihypertensive medication consultation",
                    "• Daily BP monitoring recommended"
                ])
            elif user_data.get('trestbps', 0) < 90:
                recommendations.append("• LOW BP: Monitor for symptoms, increase fluid intake")
            
            if user_data.get('chol', 0) > 240:
                recommendations.extend([
                    "• HIGH CHOLESTEROL: Start cholesterol-lowering diet",
                    "• Consider statin therapy consultation",
                    "• Reduce saturated fat intake"
                ])
            elif user_data.get('chol', 0) > 200:
                recommendations.append("• BORDERLINE CHOLESTEROL: Dietary modifications needed")
            
            if user_data.get('thalach', 0) < 60:
                recommendations.append("• LOW HEART RATE: Monitor for symptoms, consider pacemaker evaluation")
            elif user_data.get('thalach', 0) > 180:
                recommendations.append("• HIGH HEART RATE: Stress management, avoid stimulants")
            
            if user_data.get('oldpeak', 0) > 2:
                recommendations.append("• ABNORMAL ST: Urgent cardiac evaluation needed")
            
            recommendations.extend([
                "",
                "🏃‍♂️ LIFESTYLE RECOMMENDATIONS:",
                "• Exercise: 150 minutes moderate activity per week",
                "• Diet: Mediterranean diet with omega-3 fatty acids",
                "• Sleep: 7-9 hours quality sleep nightly",
                "• Stress: Practice meditation or yoga",
                "• Smoking: Complete cessation if applicable",
                "• Alcohol: Limit to 1-2 drinks per day maximum"
            ])
            
            recommendations.extend([
                "",
                "📅 FOLLOW-UP SCHEDULE:",
                f"• Next screening: {3 if prediction_probability < 0.3 else 1 if prediction_probability > 0.7 else 2} months",
                "• Blood work: Every 6 months",
                "• ECG: Annually or as recommended by physician",
                "• Echocardiogram: Every 2-3 years if indicated"
            ])
            
            recommendations.extend([
                "",
                "🚨 SEEK IMMEDIATE MEDICAL ATTENTION IF:",
                "• Chest pain or pressure lasting >5 minutes",
                "• Shortness of breath with minimal exertion",
                "• Irregular heartbeat or palpitations",
                "• Dizziness, fainting, or severe fatigue",
                "• Swelling in legs, ankles, or feet"
            ])
            
            for recommendation in recommendations:
                if y_position < 80:
                    pdf_canvas.showPage()
                    y_position = height - 50
                    pdf_canvas.setFont("Helvetica", 10)
                
                if recommendation.startswith("🚨") or recommendation.startswith("⚠️") or recommendation.startswith("✅"):
                    pdf_canvas.setFont("Helvetica-Bold", 11)
                    pdf_canvas.setFillColorRGB(0.8, 0.1, 0.1) if recommendation.startswith("🚨") else \
                    pdf_canvas.setFillColorRGB(0.8, 0.5, 0.1) if recommendation.startswith("⚠️") else \
                    pdf_canvas.setFillColorRGB(0.1, 0.6, 0.1)
                elif recommendation.startswith("📋") or recommendation.startswith("🏃‍♂️") or recommendation.startswith("📅"):
                    pdf_canvas.setFont("Helvetica-Bold", 10)
                    pdf_canvas.setFillColorRGB(0.2, 0.4, 0.8)
                elif recommendation.startswith("•"):
                    pdf_canvas.setFont("Helvetica", 9)
                    pdf_canvas.setFillColorRGB(0.2, 0.2, 0.2)
                else:
                    pdf_canvas.setFont("Helvetica", 10)
                    pdf_canvas.setFillColorRGB(0, 0, 0)
                
                if recommendation.strip():
                    pdf_canvas.drawString(50, y_position, recommendation)
                y_position -= 15
            
            if y_position < 100:
                pdf_canvas.showPage()
                y_position = height - 50
            
            pdf_canvas.setStrokeColorRGB(0.8, 0.8, 0.8)
            pdf_canvas.setLineWidth(1)
            pdf_canvas.line(50, 80, width - 50, 80)
            
            pdf_canvas.setFont("Helvetica", 8)
            pdf_canvas.setFillColorRGB(0.5, 0.5, 0.5)
            pdf_canvas.drawString(50, 60, "Generated by Heart Disease Prediction System")
            pdf_canvas.drawString(50, 50, f"Report Date: {datetime.now().strftime('%B %d, %Y')}")
            
            pdf_canvas.drawString(width - 200, 60, f"© {datetime.now().year}")
            pdf_canvas.drawString(width - 200, 50, "All Rights Reserved")
            
            pdf_canvas.save()
            buffer.seek(0)
            pdf_data = buffer.getvalue()
            
            pdf_filename = f"heart_disease_report_{timestamp}.pdf"
            _ = st.download_button(
                label="📄 Download Report",
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf",
                help="Download as PDF file",
                type="primary",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"PDF generation failed: {str(e)}")
            st.info("Please try again or contact support.")
            txt_filename = f"heart_disease_report_{timestamp}.txt"
            _ = st.download_button(
                label="📄 Download as Text",
                data=report_text,
                file_name=txt_filename,
                mime="text/plain",
                help="Download as text file (PDF failed)",
                type="primary",
                use_container_width=True
            )
            
    else:
        st.info("Make a prediction to see your medical report here.")

if __name__ == "__main__":
    main()
