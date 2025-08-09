import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time as dt_time, timedelta
import pytz
import base64
import os

st.set_page_config(
    page_title="SunNet | Solar Energy Predictor",
    page_icon="☀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    * {
        box-sizing: border-box;
    }

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #f8fafc;
        background-color: #0f172a;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        background-color: #0f172a;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid #475569;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
        color: #f1f5f9;
        font-size: 0.95rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.75rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.2s ease;
        cursor: pointer;
        background-color: transparent;
        border: 1px solid transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: rgba(129, 230, 217, 0.1);
        border-color: rgba(129, 230, 217, 0.3);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #81e6d9;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9;
        font-weight: 600;
        letter-spacing: -0.025em;
        margin-bottom: 1rem;
    }

    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
    }

    h2 {
        font-size: 1.875rem;
        line-height: 1.3;
    }

    h3 {
        font-size: 1.5rem;
        line-height: 1.4;
    }

    .main-title {
        background: linear-gradient(135deg, #81e6d9, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem;
        line-height: 1.1;
    }

    .subtitle {
        text-align: center;
        font-size: 1.25rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 1rem;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #81e6d9, #fbbf24);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
        border-color: #64748b;
    }
    
    .metric-card h4 {
        color: #cbd5e1;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #81e6d9;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .solar-noon-info {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 2px solid #81e6d9;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .solar-noon-info::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(129, 230, 217, 0.05), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%); }
        100% { transform: translateX(100%) translateY(100%); }
    }
    
    .solar-noon-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #81e6d9;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .stForm {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.8));
        border: 1px solid #475569;
        border-radius: 1rem;
        padding: 2rem;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #81e6d9, #6ed0c0);
        color: #0f172a;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(129, 230, 217, 0.39);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px 0 rgba(129, 230, 217, 0.5);
        background: linear-gradient(135deg, #6ed0c0, #5bc5b5);
    }

    .stSlider > div > div > div > div {
        background-color: #81e6d9;
    }
    
    .stSlider > div > div > div {
        background-color: #334155;
    }

    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #334155, #475569);
        color: #f1f5f9;
        font-weight: 500;
        border-radius: 0.75rem;
        padding: 1rem 1.5rem;
        border: 1px solid #64748b;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .streamlit-expanderHeader::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(180deg, #81e6d9, #fbbf24);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #475569, #64748b);
        border-color: #81e6d9;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(51, 65, 85, 0.95));
        color: #cbd5e1;
        border-radius: 0 0 0.75rem 0.75rem;
        border: 1px solid #64748b;
        border-top: none;
        padding: 1.5rem;
        backdrop-filter: blur(5px);
    }

    .stAlert {
        border-radius: 0.75rem;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }
    
    .stAlert > div {
        padding: 1rem 1.5rem;
    }
    
    div[data-baseweb="notification"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(51, 65, 85, 0.95));
        border: 1px solid #475569;
        border-radius: 0.75rem;
        backdrop-filter: blur(10px);
    }

    .team-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 1rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .team-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #81e6d9, #fbbf24, #f59e0b);
    }
    
    .team-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
        border-color: #64748b;
    }
    
    .team-card h4 {
        background: linear-gradient(135deg, #81e6d9, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.75rem;
        font-size: 1.25rem;
        font-weight: 600;
    }

    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #475569, transparent);
    }

    .form-section {
        background: linear-gradient(135deg, rgba(51, 65, 85, 0.3), rgba(71, 85, 105, 0.3));
        border: 1px solid #475569;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(5px);
    }

    .stSelectbox label, .stSlider label, .stDateInput label, .stTimeInput label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }

    .stSpinner > div {
        border-color: #81e6d9 transparent #81e6d9 transparent;
    }

    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem !important;
        }
        
        .metric-card, .team-card {
            margin-bottom: 1rem;
        }
        
        .solar-noon-value {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    
    try:
        if not os.path.exists("weighted_ensemble_model.pkl"):
            st.warning("🚫 Model file not found. Using demo mode.")
            return {
                'columns': ["temperature", "humidity", "wind-speed", "distance-to-solar-noon",
                           "sky-cover", "visibility", "average-pressure-(period)",
                           "wind-direction", "average-wind-speed-(period)",
                           "temp_squared", "wind_speed_humidity", "daylight_factor",
                           "rain_or_fog_likelihood", "pollution_proxy", "overheat_flag", "dew_morning_risk"],
                'classifier': None,
                'rf': None,
                'xg': None,
                'lgb': None,
                'weights': {'rf': 0.2, 'xg': 0.4, 'lgb': 0.4}
            }
        model = joblib.load("weighted_ensemble_model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

def calculate_distance_to_solar_noon(time_input, longitude=None, timezone_str=None):
    """
    Calculates the 'distance_to_solar_noon' based on time input only
    
    Returns a value representing the distance from solar noon (0.0504 to 1.1414 range).
    """
    try:
        
        solar_noon_time = dt_time(12, 30, 0)  
        
        input_minutes = time_input.hour * 60 + time_input.minute
        solar_noon_minutes = solar_noon_time.hour * 60 + solar_noon_time.minute
        
        diff_minutes = abs(input_minutes - solar_noon_minutes)
        
        if diff_minutes > 720:  
            diff_minutes = 1440 - diff_minutes  
        
        diff_hours = diff_minutes / 60.0
        
        min_val = 0.0504
        max_val = 1.1414
       
        normalized_distance = min_val + (diff_hours / 12.0) * (max_val - min_val)
        
        return round(normalized_distance, 4)
        
    except Exception as e:
        st.error(f"Error calculating solar noon distance: {e}")
        return None

def feature_engineering(df):
    """Generates new features from the raw input data."""
    required_cols = [
        "temperature", "humidity", "wind-speed", "distance-to-solar-noon",
        "sky-cover", "visibility", "average-pressure-(period)",
        "wind-direction", "average-wind-speed-(period)"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"🚨 Oops! Looks like we're missing some key details: {missing}. Can you help us fill them in?")
        st.stop()
        
    df["temp_squared"] = df["temperature"] ** 2
    df["wind_speed_humidity"] = df["wind-speed"] * df["humidity"]
    df["daylight_factor"] = np.maximum(0, np.cos(2 * np.pi * (df["distance-to-solar-noon"] - 0.5)))
    df["rain_or_fog_likelihood"] = ((df["sky-cover"] >= 3) & (df["humidity"] >= 80) & (df["visibility"] <= 6)).astype(int)
    df["pollution_proxy"] = ((df["visibility"] < 7) & (df["average-pressure-(period)"] < 29.9)).astype(int)
    df["overheat_flag"] = (df["temperature"] > 30).astype(int)
    df["dew_morning_risk"] = ((df["temperature"] < 10) & (df["humidity"] > 90) & (df["distance-to-solar-noon"] < 0.2)).astype(int)
    
    return df[model['columns']]

def predict_power(features_df):
    """
    Predicts solar power output using a two-stage weighted ensemble model.
    """
    try:
        engineered_features = feature_engineering(features_df)
 
        if model['classifier'] is None:
            
            temp = features_df["temperature"].iloc[0]
            humidity = features_df["humidity"].iloc[0]
            sky_cover = features_df["sky-cover"].iloc[0]
            distance_to_noon = features_df["distance-to-solar-noon"].iloc[0]
            
            if distance_to_noon > 0.8:  
                return 0.0
            
            base_power = max(0, (temp - 40) * 100)  
            base_power *= max(0, (100 - humidity) / 100)  
            base_power *= max(0, (4 - sky_cover) / 4)  
            base_power *= max(0, 1 - distance_to_noon)  
            
            return max(0, base_power * 200)
        
        binary = model['classifier'].predict(engineered_features)
        
        if binary[0] == 0:
            return 0.0
            
        pred_rf = model['rf'].predict(engineered_features)
        pred_xg = model['xg'].predict(engineered_features)
        pred_lgb = model['lgb'].predict(engineered_features)
        
        final_pred = (
            model['weights']['rf'] * pred_rf +
            model['weights']['xg'] * pred_xg +
            model['weights']['lgb'] * pred_lgb
        )
        
        return np.clip(final_pred[0], 0, None)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

with st.sidebar:
    st.markdown("<h2>☀ SunNet</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #cbd5e1; font-size: 0.9rem; margin-bottom: 2rem;'>Your guide to a brighter energy future</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Data Visualisation", "⚙ Predictor", "👨‍💻 Our Team"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='margin: 2rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, #475569, transparent);'>", unsafe_allow_html=True)

def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    except:
        pass
    return None

base64_image = None
if page == "🏠 Home":
    base64_image = get_base64_image("harness.jpg")
else:
    base64_image = get_base64_image("Other_bg.jpg")

if base64_image:
    st.markdown(f"""
    <style>
        .stApp {{
            background: url("data:image/png;base64,{base64_image}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(1px);
            z-index: -1;
        }}
    </style>
    """, unsafe_allow_html=True)

if page == "🏠 Home":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">Solar Energy Predictor using Machine Learning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Illuminating the Future of Energy ☀</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.8)); 
                   border-radius: 1rem; border: 1px solid #475569; backdrop-filter: blur(10px); margin: 2rem 0;">
            <p style="font-size: 1.125rem; color: #cbd5e1; margin: 0;">
                Welcome to <strong style="color: #81e6d9;">SunNet</strong>, where we're using Machine Learning to help you predict solar energy generation with confidence. 
                Our powerful model makes it easy to understand and predict how much power your solar panels can produce based on local weather conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🚀 Our Secret Sauce: A Two-Stage Model</h2>", unsafe_allow_html=True)
    with st.expander("🔍 Peek behind the curtain to see how it works", expanded=False):
        st.markdown("""
        <div style="padding: 1rem 0;">
            <p style="color: #cbd5e1; font-size: 1.05rem; line-height: 1.7;">
                Our system is super smart! It uses a special two-part process to give you the best possible prediction:
            </p>
            <div style="display: grid; gap: 1.5rem; margin: 2rem 0;">
                <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(129, 230, 217, 0.1), rgba(251, 191, 36, 0.1)); 
                           border-left: 4px solid #81e6d9; border-radius: 0.5rem;">
                    <h4 style="color: #81e6d9; margin-bottom: 0.5rem;">1. Classification Stage</h4>
                    <p style="color: #cbd5e1; margin: 0;">First, a model quickly checks if there's any chance of solar power being generated at all. 
                    It's like asking, "Is it day or night? Is it super cloudy or clear?" This helps us avoid making a guess when it's not needed.</p>
                </div>
                <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1)); 
                           border-left: 4px solid #fbbf24; border-radius: 0.5rem;">
                    <h4 style="color: #fbbf24; margin-bottom: 0.5rem;">2. Ensemble Regression</h4>
                    <p style="color: #cbd5e1; margin: 0;">If the sun is out, three of our best machine learning models—Random Forest, 
                    <strong>XGBoost</strong>, and <strong>LightGBM</strong>-get to work. They all make a prediction, and we combine their answers 
                    to get a final, super-reliable number. It's like having three experts agree on the best prediction!</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
  
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🛠 The Tools We Used</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 The Brains</h4>
            <p style="font-size: 1.1rem; color: #cbd5e1;">Python • Scikit-learn • XGBoost • LightGBM</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎨 The Interface</h4>
            <p style="font-size: 1.1rem; color: #cbd5e1;">Streamlit • Modern UI/UX Design</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Data Visualisation":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>📈 Our Data Visualisation in Pictures</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #cbd5e1; margin-bottom: 3rem;'>We believe in being transparent! These charts show you exactly how well our model performs and how it makes its decisions.</p>", unsafe_allow_html=True)

    
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>📊 Key Performance Metrics</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h4>R² Score</h4><p>0.9124</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>MAE (kW)</h4><p>1428.91</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>RMSE (kW)</h4><p>3103.88</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    

    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("<h3>📚 Metric Definitions</h3>", unsafe_allow_html=True)
    
    definitions = [
        ("R² (R-squared) Score", "Also known as the Coefficient of Determination, this metric measures how well our model's predictions align with the actual data. A score of 1.0 indicates a perfect fit."),
        ("MAE (Mean Absolute Error)", "This represents the Mean Absolute Error, which is the average of the absolute differences between the predicted values and the actual values. It gives a straightforward measure of the model's average prediction error."),
        ("RMSE (Root Mean Square Error)", "This stands for the Root Mean Square Error. It measures the standard deviation of the prediction errors. Because it squares the errors before averaging them, it gives more weight to large errors, making it a good indicator of how sensitive the model is to significant mistakes.")
    ]
    
    for title, description in definitions:
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem; padding: 1rem; background: rgba(51, 65, 85, 0.3); border-radius: 0.5rem; border-left: 3px solid #81e6d9;">
            <h4 style="color: #81e6d9; margin-bottom: 0.5rem;">{title}</h4>
            <p style="color: #cbd5e1; margin: 0; line-height: 1.6;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🔍 Zero-Power Prediction Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="form-section">
            <p style="color: #cbd5e1; font-size: 1.05rem; line-height: 1.7;">
                Our two-stage approach is designed to be highly accurate, especially for distinguishing between "on" and "off" states. Here's how it performed:
            </p>
            <div style="display: grid; gap: 1rem; margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: rgba(129, 230, 217, 0.1); border-radius: 0.5rem;">
                    <span style="color: #cbd5e1;">Total actual zero power instances:</span>
                    <span style="color: #81e6d9; font-weight: 600;">251</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.5rem;">
                    <span style="color: #cbd5e1;">Correctly predicted zero power:</span>
                    <span style="color: #10b981; font-weight: 600;">238</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: rgba(245, 158, 11, 0.1); border-radius: 0.5rem;">
                    <span style="color: #cbd5e1;">False Positives:</span>
                    <span style="color: #f59e0b; font-weight: 600;">13</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-radius: 0.5rem;">
                    <span style="color: #cbd5e1;">False Negatives:</span>
                    <span style="color: #ef4444; font-weight: 600;">14</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accuracy = (238 / 251) * 100
        st.markdown(f"""
        <div class="metric-card" style="height: 100%;">
            <h4>Accuracy Rate</h4>
            <p style="font-size: 3rem;">{accuracy:.1f}%</p>
            <p style="font-size: 0.9rem; color: #cbd5e1; margin-top: 1rem;">
                Exceptional performance in zero-power detection
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
  
    image_sections = [
        {
            "title": "📌 1. What Influences the Prediction?",
            "file": "correlation_matrix.jpeg",
            "caption": "A visual guide to how weather features are connected",
            "description": "This matrix shows how different weather conditions affect solar output. You can see how things like temperature, humidity, and cloud cover are all linked to how much energy is produced!"
        },
        {
            "title": "📌 2. How Good Are Our Predictions?",
            "file": "residual_vs_actual.jpeg",
            "caption": "Comparing our predictions to what really happened",
            "description": "This plot shows our predictions against the actual energy generated. The dashed line represents perfect prediction. The closer the dots are to the line, the better the model is doing!"
        },
        {
            "title": "📌 3. Finding the Small Errors",
            "file": "actual_vs_predicted.jpeg",
            "caption": "Spotting where we might over- or under-predict",
            "description": "This graph helps us find any small, hidden biases in our model. If all the dots are scattered randomly around the middle line, it means our predictions are consistent!"
        }
    ]
    
    for section in image_sections:
        st.markdown(f"<h2>{section['title']}</h2>", unsafe_allow_html=True)
        
        if os.path.exists(section['file']):
            st.image(section['file'], caption=section['caption'], use_container_width=True)
        else:
            st.markdown(f"""
            <div style="padding: 3rem; text-align: center; background: linear-gradient(135deg, rgba(51, 65, 85, 0.3), rgba(71, 85, 105, 0.3)); 
                       border: 2px dashed #475569; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="color: #81e6d9; margin-bottom: 1rem;">📊 Visualization Coming Soon</h4>
                <p style="color: #cbd5e1; font-style: italic;">{section['caption']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="form-section">
            <p style="color: #cbd5e1; font-size: 1.05rem; line-height: 1.7; margin: 0;">
                {section['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(129, 230, 217, 0.1), rgba(16, 185, 129, 0.1)); 
               border-radius: 1rem; border: 1px solid rgba(129, 230, 217, 0.3);">
        <p style="color: #81e6d9; font-size: 1.1rem; font-weight: 600; margin: 0;">
            💡 All of these graphs are based on data our model hasn't seen before!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "⚙ Predictor":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>⚙ See Our Model in Action</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #cbd5e1; margin-bottom: 2rem;'>Ready to try it yourself? Just enter some information below, and we'll give you a Solar Power Forecast!</p>", unsafe_allow_html=True)
    

    if model['classifier'] is None:
        st.markdown("""
        <div style="padding: 1rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(249, 115, 22, 0.1)); 
                   border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 0.75rem; margin-bottom: 2rem;">
            <p style="color: #f59e0b; margin: 0; text-align: center; font-weight: 500;">
                🚨 Running in demo mode. Please upload the model file for accurate predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.form("input_form"):
        st.markdown("<h3>☀ Solar Position Analysis</h3>", unsafe_allow_html=True)
        
        time_input = st.time_input("Time", dt_time(12, 0))
        st.markdown("<p style='color: #94a3b8; font-size: 0.875rem; margin-top: 0.5rem;'>⏰ Enter any time of the day</p>", unsafe_allow_html=True)
        
        distance_to_solar_noon = calculate_distance_to_solar_noon(time_input)
        
        if distance_to_solar_noon is not None:
            time_from_noon_hours = abs((time_input.hour + time_input.minute/60) - 12.5)
            time_from_noon_minutes = int(time_from_noon_hours * 60)
            
            col_solar1, col_solar2 = st.columns(2)

            box_style = """
                border: 2px solid {border_color};
                border-radius: 0.75rem;
                padding: 1.5rem;
                margin: 0.5rem 0;
                background: {bg_color};
                height: 100%;
            """
            with col_solar1:
                st.markdown(f"""
                <div style="{box_style.format(border_color='#10b981', bg_color='rgba(16, 185, 129, 0.05)')}">
                    <h4>🌞 Distance to Solar Noon</h4>
                    <div style="font-size: 2rem; font-weight: bold;">{distance_to_solar_noon:.4f}</div>
                    <p><strong>Time difference:</strong> {time_from_noon_minutes} minutes from 12:30 PM</p>
                </div>
                """, unsafe_allow_html=True)

            with col_solar2:
                if distance_to_solar_noon < 0.3:
                    solar_status = "☀ Peak Solar Hours"
                    solar_color = "#10b981"
                    solar_bg = "rgba(16, 185, 129, 0.05)"
                elif distance_to_solar_noon < 0.6:
                    solar_status = "🌤 Good Solar Hours"
                    solar_color = "#f59e0b"
                    solar_bg = "rgba(245, 158, 11, 0.05)"
                elif distance_to_solar_noon < 0.9:
                    solar_status = "🌅 Low Solar Hours"
                    solar_color = "#f97316"
                    solar_bg = "rgba(249, 115, 22, 0.05)"
                else:
                    solar_status = "🌙 Nighttime/No Solar"
                    solar_color = "#6b7280"
                    solar_bg = "rgba(107, 114, 128, 0.05)"

                st.markdown(f"""
                <div style="{box_style.format(border_color=solar_color, bg_color=solar_bg)}">
                    <h4 style="color:{solar_color};">Solar Condition</h4>
                    <p style="font-size:1.2rem; font-weight:bold; color:{solar_color};">{solar_status}</p>
                    <p>Based on time of day</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h3>🌤 Weather Conditions</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color: #81e6d9; font-size: 1rem; margin-bottom: 1rem;'>🌡 Temperature & Wind</h4>", unsafe_allow_html=True)
            temperature = st.slider("Temperature (°F)", min_value=42, max_value=78, value=65)
            wind_speed = st.slider("Wind Speed (mph)", min_value=1.1, max_value=26.6, value=5.0, step=0.1)
            avg_wind = st.slider("Avg Wind Speed (mph)", min_value=0.0, max_value=40.0, value=5.0, step=0.1)
            wind_dir = st.slider("Wind Direction (°)", min_value=1, max_value=36, value=18)

        with col2:
            st.markdown("<h4 style='color: #fbbf24; font-size: 1rem; margin-bottom: 1rem;'>💧 Atmospheric Conditions</h4>", unsafe_allow_html=True)
            humidity = st.slider("Humidity (%)", min_value=14, max_value=100, value=50)
            sky_cover = st.slider("Sky Cover (0–8, 0=clear, 8=overcast)", min_value=0, max_value=8, value=2)
            visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
            avg_pressure = st.slider("Avg Pressure (inHg)", min_value=29.48, max_value=30.53, value=29.9, step=0.01)

        submitted = st.form_submit_button("⚡ Predict Solar Power")

        if submitted:
            if distance_to_solar_noon is not None:
                input_df = pd.DataFrame([{
                    "distance-to-solar-noon": distance_to_solar_noon,
                    "temperature": temperature,
                    "wind-direction": wind_dir,
                    "wind-speed": wind_speed,
                    "sky-cover": sky_cover,
                    "visibility": visibility,
                    "humidity": humidity,
                    "average-wind-speed-(period)": avg_wind,
                    "average-pressure-(period)": avg_pressure
                }])
                
                with st.spinner("⏳ Our Model is working its magic..."):
                    prediction = predict_power(input_df)
                    if prediction is not None:
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 2rem; margin: 2rem 0; background: linear-gradient(135deg, rgba(129, 230, 217, 0.2), rgba(16, 185, 129, 0.2)); 
                                   border: 2px solid #10b981; border-radius: 1rem; backdrop-filter: blur(10px);">
                            <h2 style="color: #10b981; margin-bottom: 1rem;">☀ Prediction Result</h2>
                            <div style="font-size: 3rem; font-weight: 700; color: #81e6d9; margin: 1rem 0; font-family: 'JetBrains Mono', monospace;">
                                {prediction:.2f} W
                            </div>
                            <p style="color: #cbd5e1; font-size: 1.1rem; margin: 0;">Your predicted solar power output</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        
                        if prediction > 20000:
                            st.balloons()
                            st.markdown("""
                            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1)); 
                                       border: 1px solid #10b981; border-radius: 0.75rem; margin: 1rem 0;">
                                <p style="color: #10b981; font-size: 1.1rem; font-weight: 600; margin: 0; text-align: center;">
                                    🌟 Excellent solar conditions! Peak power generation expected.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif prediction > 10000:
                            st.markdown("""
                            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(132, 204, 22, 0.1)); 
                                       border: 1px solid #22c55e; border-radius: 0.75rem; margin: 1rem 0;">
                                <p style="color: #22c55e; font-size: 1.1rem; font-weight: 600; margin: 0; text-align: center;">
                                    ☀ Good solar conditions! Solid power generation expected.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif prediction > 1000:
                            st.markdown("""
                            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(249, 115, 22, 0.1)); 
                                       border: 1px solid #f59e0b; border-radius: 0.75rem; margin: 1rem 0;">
                                <p style="color: #f59e0b; font-size: 1.1rem; font-weight: 600; margin: 0; text-align: center;">
                                    ⛅ Moderate solar conditions. Some power generation expected.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif prediction > 0:
                            st.markdown("""
                            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(249, 115, 22, 0.1), rgba(239, 68, 68, 0.1)); 
                                       border: 1px solid #f97316; border-radius: 0.75rem; margin: 1rem 0;">
                                <p style="color: #f97316; font-size: 1.1rem; font-weight: 600; margin: 0; text-align: center;">
                                    🌥 Limited solar conditions. Minimal power generation expected.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(107, 114, 128, 0.1), rgba(75, 85, 99, 0.1)); 
                                       border: 1px solid #6b7280; border-radius: 0.75rem; margin: 1rem 0;">
                                <p style="color: #6b7280; font-size: 1.1rem; font-weight: 600; margin: 0; text-align: center;">
                                    🌙 No solar power generation expected (nighttime or very poor conditions).
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "👨‍💻 Our Team":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>👨‍🔬 Meet the Brilliant Minds Behind SunNet</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #cbd5e1; margin-bottom: 3rem;'>We're a small, dedicated team with a big goal: to make renewable energy smarter and more accessible for everyone.</p>", unsafe_allow_html=True)
    
    
    col1, col2 = st.columns(2)
    team_members = [
        {
            "name": "Husain Ghulam",
            "role": "ML Engineer",
            "description": "Created machine learning models for solar power prediction and documented all code and results.",
            "icon": "🧠"
        },
        {
            "name": "Shreesh Jugade",
            "role": "Frontend Developer & ML Contributor", 
            "description": "Built the complete UI/UX interface and assisted in machine learning development and testing.",
            "icon": "🎯"
        },
        {
            "name": "Ayush Shevde", 
            "role": "ML Engineer",
            "description": "Assisted in creating and testing machine learning models and contributed to documentation.",
            "icon": "📊"
        },
        {
            "name": "Tanishk Deshpande",
            "role": "Frontend Assistant & GitHub Organizer",
            "description": "Assisted in frontend development and organized the GitHub repository structure.",
            "icon": "🚀"
        }
    ]
    
    for i, member in enumerate(team_members):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{member['icon']}</div>
                <h4>{member['name']}</h4>
                <p style="color: #fbbf24; font-weight: 600; margin-bottom: 1rem;">{member['role']}</p>
                <p style="color: #cbd5e1; line-height: 1.6; margin: 0;">{member['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
   
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.8)); 
            border-radius: 1rem; border: 1px solid #475569; backdrop-filter: blur(10px);">
        <p style="color: #81e6d9; font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">
            Connect With Our Project
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <a href="https://github.com/husain34/SunNet-TechRush.git" target="_blank" 
            style="color: #cbd5e1; text-decoration: none;">
            🔗 GitHub Repo
            </a>
            <span style="color: #cbd5e1;">🏆 Proudly built during Hackathon 2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown('</div>', unsafe_allow_html=True)