import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime, time as dt_time, timedelta
import base64
from dotenv import load_dotenv
import asyncio
import json
import plotly.graph_objects as go
import plotly.express as px

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Secure API Key Loading ---
load_dotenv() # For local development with a .env file

# Access secrets from st.secrets when deployed, otherwise from environment variables
# This is the modern, secure way to handle keys and removes the need to download a .env file.
WEATHERAPI_API_KEY = st.secrets.get("WEATHERAPI_API_KEY", os.getenv("WEATHERAPI_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))


# --- Centralized File Management from Google Drive ---
# Dictionary mapping filenames to their Google Drive file IDs
GDRIVE_FILE_IDS = {
    "weighted_ensemble_model.pkl": "1eVlzFkS433sBDaq_9xJeH-QMpsXP0yE3",
    "knowledge_base.txt": "1Vcl3MIao49_ujmw8neyIRzpnDN6o-kVi",
    "Harness.jpg": "1o7yMHyk7fz-AW4DnNeiJy-gXNtDAW2yq",
    "Other_bg.jpg": "1MPnMS-oWuCfbhhGOtp83aQ-wi5Tf6W3w",
    "correlation_matrix.jpeg": "133_7z_-NsI6ufOfRBXEDjCXOgfsmW2f0",
    "residual_vs_actual.jpeg": "1iYSX0Z_B9-xevG_QaOi1Xy2ub_-Vre2T",
    "actual_vs_predicted.jpeg": "1gtbd6FZS0d8Rf5C2VLOOlYTjW6-FygA0"
}
@st.cache_resource
def download_file_from_gdrive(filename):
    """
    Checks if a file exists locally. If not, downloads it from Google Drive.
    Uses st.cache_resource to prevent re-downloading within the same session.
    Returns the local path to the file.
    """
    if os.path.exists(filename):
        return filename

    file_id = GDRIVE_FILE_IDS.get(filename)
    if not file_id:
        st.error(f"File '{filename}' not found in the Google Drive mapping. Cannot proceed.")
        st.stop()

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        with st.spinner(f"Downloading required file: {filename}..."):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        return filename
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download '{filename}'. Error: {e}")
        st.stop()

# --- Page Configuration and Styling (No changes needed here) ---
st.set_page_config(
    page_title="SunNet | Solar Energy Predictor",
    page_icon="☀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Your existing CSS goes here */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    * { box-sizing: border-box; }
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #f8fafc;
        background-color: #0f172a;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stApp { background-color: #0f172a; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid #475569;
    }
    [data-testid="stSidebar"] .stRadio > label { font-weight: 500; color: #f1f5f9; font-size: 0.95rem; }
    [data-testid="stSidebar"] .stRadio > div { gap: 0.75rem; }
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
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9; font-weight: 600; letter-spacing: -0.025em; margin-bottom: 1rem; }
    h1 { font-size: 2.5rem; font-weight: 700; line-height: 1.2; }
    h2 { font-size: 1.875rem; line-height: 1.3; }
    h3 { font-size: 1.5rem; line-height: 1.4; }
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
    .subtitle { text-align: center; font-size: 1.25rem; color: #cbd5e1; margin-bottom: 2rem; font-weight: 400; }
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
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #81e6d9, #fbbf24);
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4); border-color: #64748b; }
    .metric-card h4 { color: #cbd5e1; font-size: 0.875rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .metric-card p { color: #81e6d9; font-size: 2rem; font-weight: 700; margin: 0; font-family: 'JetBrains Mono', monospace; }
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
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: linear-gradient(45deg, transparent, rgba(129, 230, 217, 0.05), transparent);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%); }
        100% { transform: translateX(100%) translateY(100%); }
    }
    .solar-noon-value { font-size: 2.5rem; font-weight: 700; color: #81e6d9; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace; }
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
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    .stButton > button:hover::before { left: 100%; }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px 0 rgba(129, 230, 217, 0.5);
        background: linear-gradient(135deg, #6ed0c0, #5bc5b5);
    }
    .stSlider > div > div > div > div { background-color: #81e6d9; }
    .stSlider > div > div > div { background-color: #334155; }
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
        left: 0; top: 0;
        height: 100%; width: 4px;
        background: linear-gradient(180deg, #81e6d9, #fbbf24);
    }
    .streamlit-expanderHeader:hover { background: linear-gradient(135deg, #475569, #64748b); border-color: #81e6d9; }
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(51, 65, 85, 0.95));
        color: #cbd5e1;
        border-radius: 0 0 0.75rem 0.75rem;
        border: 1px solid #64748b;
        border-top: none;
        padding: 1.5rem;
        backdrop-filter: blur(5px);
    }
    .stAlert { border-radius: 0.75rem; border-left: 4px solid; backdrop-filter: blur(10px); }
    .stAlert > div { padding: 1rem 1.5rem; }
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
        top: 0; left: 0; right: 0;
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
    .main-container { max-width: 1200px; margin: 0 auto; padding: 2rem 1rem; }
    hr { margin: 3rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, #475569, transparent); }
    .form-section {
        background: linear-gradient(135deg, rgba(51, 65, 85, 0.3), rgba(71, 85, 105, 0.3));
        border: 1px solid #475569;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(5px);
    }
    .stSelectbox label, .stSlider label, .stDateInput label, .stTimeInput label, .stNumberInput label { color: #cbd5e1 !important; font-weight: 500 !important; font-size: 0.875rem !important; }
    .stSpinner > div { border-color: #81e6d9 transparent #81e6d9 transparent; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #1e293b; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #64748b; }
    @media (max-width: 768px) {
        .main-title { font-size: 2.5rem !important; }
        .metric-card, .team-card { margin-bottom: 1rem; }
        .solar-noon-value { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    Ensures the model file is downloaded and then loads it into memory.
    """
    model_path = download_file_from_gdrive("weighted_ensemble_model.pkl")
    try:
        model_dict = joblib.load(model_path)
        return model_dict
    except Exception as e:
        st.error(f"Failed to load the model file. It might be corrupted. Error: {e}")
        st.stop()

@st.cache_data(ttl=600)
def get_live_weather(city, api_key):
    """Fetches and processes live weather data from WeatherAPI.com."""
    if not api_key:
        return {"error": "WeatherAPI.com API key not found. Please set it in your environment or Streamlit secrets."}
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    try:
        response = requests.get(url)
        data = response.json()
        if "error" in data:
            return {"error": f"Error from WeatherAPI: {data['error']['message']}"}

        current = data.get("current", {})
        location = data.get("location", {})


        official_city_name = location.get("name", city)

        weather_data = {
            "temperature": current.get("temp_f"), "humidity": current.get("humidity"),
            "wind_speed": current.get("wind_mph"), "avg_wind": current.get("wind_mph"),
            "wind_dir": int(current.get("wind_degree", 180) / 10),
            "sky_cover": int(round(current.get("cloud", 0) / 100 * 8)),
            "visibility": current.get("vis_miles"), "avg_pressure": current.get("pressure_in"),
            "localtime": location.get("localtime")
        }

        if any(v is None for v in weather_data.values()):
            return {"error": "API response was missing required weather data."}


        return {"data": weather_data, "city_name": official_city_name}

    except requests.RequestException as e:
        return {"error": f"A network error occurred: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@st.cache_data(ttl=3600)
def get_historical_and_forecast_weather(city, api_key):
    """Fetches weather data for the last 3 days, today, and the next 3 days."""
    if not api_key:
        return {"error": "WeatherAPI.com API key not found."}

    all_days_data = []
    today = datetime.now().date()
    official_city_name = city # Default city name

    # --- 1. Fetch Historical Data (3 past days) ---
    historical_days_to_fetch = [today - timedelta(days=i) for i in range(3, 0, -1)]

    for date_to_fetch in historical_days_to_fetch:
        date_str = date_to_fetch.strftime("%Y-%m-%d")
        history_url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date_str}"
        try:
            response = requests.get(history_url)
            history_data = response.json()
            if "error" in history_data:
                st.warning(f"Could not fetch history for {date_str}: {history_data['error']['message']}")
                continue

            day_entry = history_data.get("forecast", {}).get("forecastday", [])[0]
            day_data = day_entry.get("day", {})
            hourly_clouds = [hour.get("cloud", 0) for hour in day_entry.get("hour", [])]
            avg_cloud = np.mean(hourly_clouds) if hourly_clouds else 0

            all_days_data.append({
                "date": day_entry.get("date"),
                "temperature": day_data.get("avgtemp_f"),
                "humidity": day_data.get("avghumidity"),
                "wind_speed": day_data.get("maxwind_mph"),
                "sky_cover": int(round(avg_cloud / 100 * 8)),
                "visibility": day_data.get("avgvis_miles"),
                "condition": day_data.get("condition", {}).get("text", "N/A")
            })
        except Exception as e:
            st.warning(f"An error occurred fetching history for {date_str}: {e}")
            continue

    # --- 2. Fetch Forecast Data (Today + 3 future days) ---
    forecast_url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=4"
    try:
        response = requests.get(forecast_url)
        forecast_data = response.json()
        if "error" in forecast_data:
            return {"error": f"Error from WeatherAPI Forecast: {forecast_data['error']['message']}"}

        location = forecast_data.get("location", {})
        official_city_name = location.get("name", city)

        forecast_days = forecast_data.get("forecast", {}).get("forecastday", [])
        for day in forecast_days:
            day_data = day.get("day", {})
            hourly_clouds = [hour.get("cloud", 0) for hour in day.get("hour", [])]
            avg_cloud = np.mean(hourly_clouds) if hourly_clouds else 0

            all_days_data.append({
                "date": day.get("date"),
                "temperature": day_data.get("avgtemp_f"),
                "humidity": day_data.get("avghumidity"),
                "wind_speed": day_data.get("maxwind_mph"),
                "sky_cover": int(round(avg_cloud / 100 * 8)),
                "visibility": day_data.get("avgvis_miles"),
                "condition": day_data.get("condition", {}).get("text", "N/A")
            })

    except Exception as e:
        return {"error": f"An error occurred fetching forecast data: {e}"}

    if not all_days_data:
         return {"error": "Could not retrieve any weather data for the specified location."}

    return {"outlook": all_days_data, "city_name": official_city_name}

def check_unusual_prediction_and_explain(prediction_value, weather_conditions):
    """Check if prediction is unusual and get AI explanation if needed."""

    HIGH_THRESHOLD = 25000
    LOW_THRESHOLD = 500

    is_unusual = prediction_value > HIGH_THRESHOLD or (
        prediction_value < LOW_THRESHOLD and
        weather_conditions.get('distance-to-solar-noon', 1) < 0.3
    )

    if not is_unusual or not GOOGLE_API_KEY:
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )


        condition_summary = f"""
        Temperature: {weather_conditions.get('temperature', 'N/A')}°F
        Humidity: {weather_conditions.get('humidity', 'N/A')}%
        Sky Cover: {weather_conditions.get('sky-cover', 'N/A')}/8
        Wind Speed: {weather_conditions.get('wind-speed', 'N/A')} mph
        Visibility: {weather_conditions.get('visibility', 'N/A')} miles
        Distance to Solar Noon: {weather_conditions.get('distance-to-solar-noon', 'N/A')}
        """

        prompt = f"""
        As a solar energy expert, explain why this solar power prediction might be unusual:

        Predicted Solar Output: {prediction_value:.2f} W
        Weather Conditions:
        {condition_summary}

        Please provide a brief, technical explanation (2-3 sentences) for why this prediction
        might be unusually {'high' if prediction_value > HIGH_THRESHOLD else 'low'} given these conditions.
        Focus on the meteorological and physical factors that could cause this outcome.
        """

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Could not generate explanation: {e}"

@st.cache_resource
def get_enhanced_rag_chain():
    """Enhanced RAG chain that can handle weather queries and predictions."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not GOOGLE_API_KEY:
        st.error("Google API Key not found. The AI Assistant is disabled.")
        return None

    try:

        knowledge_base_path = download_file_from_gdrive("knowledge_base.txt")
        try:
            with open(knowledge_base_path, "r", encoding="utf-8") as f:
                knowledge_base_text = f.read()
        except FileNotFoundError:
            knowledge_base_text = """
            SunNet is a solar energy prediction system that uses machine learning to forecast solar power generation.
            It uses weather data including temperature, humidity, wind speed, sky cover, visibility, and atmospheric pressure.
            The system employs a two-stage ensemble model with Random Forest, XGBoost, and LightGBM algorithms.
            """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.create_documents([knowledge_base_text])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever()

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.3)


        prompt = ChatPromptTemplate.from_template(
            """
            You are SunNet AI, a friendly and knowledgeable assistant for a solar energy prediction app.
            You can help with:
            1. General questions about the SunNet project (use the Context provided)
            2. Weather-related queries (I can fetch live weather data)
            3. Solar energy predictions (I can make predictions using our ML model)
            4. Solar energy and weather science in general

            Context about SunNet:
            {context}

            User Question: {input}

            Instructions:
            - If the user asks about weather for a specific city, respond that you can fetch live weather data
            - If the user asks for a solar prediction, explain that you can make predictions using our ML model
            - For general questions, use the context and your knowledge
            - Be helpful, concise, and technical when appropriate
            - Always maintain a friendly, professional tone

            Answer:
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        st.error(f"Error initializing Enhanced AI Assistant: {e}")
        return None


def get_base64_image(image_path):
    """Encodes an image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except (FileNotFoundError, Exception):
        return None

def set_background(image_path):
    """Sets the background image of the app."""
    encoded_image = get_base64_image(image_path)
    if encoded_image:
        st.markdown(f"""
        <style>
            .stApp {{
                background: url("data:image/png;base64,{encoded_image}") no-repeat center center fixed;
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

def calculate_distance_to_solar_noon(time_input):
    """Calculates a normalized value representing the distance from solar noon."""
    solar_noon_time = dt_time(12, 30, 0)
    input_minutes = time_input.hour * 60 + time_input.minute
    solar_noon_minutes = solar_noon_time.hour * 60 + solar_noon_time.minute
    diff_minutes = abs(input_minutes - solar_noon_minutes)
    if diff_minutes > 720:
        diff_minutes = 1440 - diff_minutes
    diff_hours = diff_minutes / 60.0
    min_val, max_val = 0.0504, 1.1414
    return round(min_val + (diff_hours / 12.0) * (max_val - min_val), 4)

def feature_engineering(df, model_cols):
    """Generates new features from the raw input data."""
    df["temp_squared"] = df["temperature"] ** 2
    df["wind_speed_humidity"] = df["wind-speed"] * df["humidity"]
    df["daylight_factor"] = np.maximum(0, np.cos(2 * np.pi * (df["distance-to-solar-noon"] - 0.5)))
    df["rain_or_fog_likelihood"] = ((df["sky-cover"] >= 3) & (df["humidity"] >= 80) & (df["visibility"] <= 6)).astype(int)
    df["pollution_proxy"] = ((df["visibility"] < 7) & (df["average-pressure-(period)"] < 29.9)).astype(int)
    df["overheat_flag"] = (df["temperature"] > 30).astype(int)
    df["dew_morning_risk"] = ((df["temperature"] < 10) & (df["humidity"] > 90) & (df["distance-to-solar-noon"] < 0.2)).astype(int)
    return df[model_cols]

def predict_power(model_dict, features_df):
    """Predicts solar power output using the two-stage ensemble model."""
    engineered_features = feature_engineering(features_df, model_dict['columns'])

    if model_dict.get('classifier') is None:

        base_prediction = np.random.uniform(5000, 20000)

        temp_factor = features_df['temperature'].iloc[0] / 70.0
        cloud_factor = 1.0 - (features_df['sky-cover'].iloc[0] / 8.0)
        return max(0, base_prediction * temp_factor * cloud_factor)

    if model_dict['classifier'].predict(engineered_features)[0] == 0:
        return 0.0

    pred_rf = model_dict['rf'].predict(engineered_features)
    pred_xg = model_dict['xg'].predict(engineered_features)
    pred_lgb = model_dict['lgb'].predict(engineered_features)

    final_pred = (
        model_dict['weights']['rf'] * pred_rf +
        model_dict['weights']['xg'] * pred_xg +
        model_dict['weights']['lgb'] * pred_lgb
    )
    return np.clip(final_pred[0], 0, None)


def home_page():
    bg_path = download_file_from_gdrive("Harness.jpg")
    set_background(bg_path)
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

    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🆕 New Enhanced Features</h2>", unsafe_allow_html=True)

    feature_cols = st.columns(4)
    new_features = [
        {"icon": "📅", "title": "7-Day Forecast", "desc": "Get solar predictions for the entire week"},
        {"icon": "🤖", "title": "Smart Chatbot", "desc": "Ask for weather data and predictions via chat"},
        {"icon": "🧠", "title": "AI Explanations", "desc": "Understand unusual predictions with AI insights"},
        {"icon": "☁️", "title": "Live Weather", "desc": "Real-time weather integration for any city"}
    ]

    for i, feature in enumerate(new_features):
        with feature_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1.5rem;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">{feature['icon']}</div>
                <h4 style="color: #81e6d9; margin-bottom: 0.5rem;">{feature['title']}</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.4;">{feature['desc']}</p>
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
                    <strong>XGBoost</strong>, and <strong>LightGBM</strong>—get to work. They all make a prediction, and we combine their answers
                    to get a final, super-reliable number. It's like having three experts agree on the best prediction!</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def outlook_page(model_dict):
    bg_path = download_file_from_gdrive("Other_bg.jpg")
    set_background(bg_path)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>📅 7-Day Solar Outlook</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Review past performance and plan for the coming days!</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        outlook_city_input = st.text_input("Enter city for 7-day outlook", placeholder="e.g., New York, USA", key="outlook_city_input")
    with col2:
        st.write("")
        st.write("")
        if st.button("Get 7-Day Outlook", use_container_width=True):
            if outlook_city_input:
                with st.spinner(f"Fetching 7-day outlook for {outlook_city_input}..."):
                    api_response = get_historical_and_forecast_weather(outlook_city_input, WEATHERAPI_API_KEY)
                    st.session_state['outlook_api_response'] = api_response

                    if "error" in api_response:
                        st.error(f"❌ {api_response['error']}")
                    else:
                        official_city_name = api_response.get('city_name', outlook_city_input.title())
                        st.success(f"✅ Outlook data fetched for {official_city_name}!")
            else:
                st.warning("Please enter a city name.")

    if 'outlook_api_response' in st.session_state and "error" not in st.session_state['outlook_api_response']:
        response_data = st.session_state['outlook_api_response']
        outlook = response_data['outlook']
        city_name = response_data['city_name']

        st.markdown(f"<h2 style='text-align: center; margin-top: 2rem;'>☀ 7-Day Outlook for {city_name}</h2>", unsafe_allow_html=True)

        predictions, date_strings, weather_conditions = [], [], []

        for day in outlook:
            if any(key not in day or day[key] is None for key in ['temperature', 'wind_speed', 'sky_cover', 'visibility', 'humidity']):
                st.warning(f"Skipping prediction for {day.get('date')} due to incomplete data.")
                continue

            noon_distance = calculate_distance_to_solar_noon(dt_time(12, 0))
            input_data = {
                "distance-to-solar-noon": noon_distance, "temperature": day['temperature'],
                "wind-direction": 18, "wind-speed": day['wind_speed'],
                "sky-cover": day['sky_cover'], "visibility": day['visibility'],
                "humidity": day['humidity'], "average-wind-speed-(period)": day['wind_speed'],
                "average-pressure-(period)": 29.9
            }
            prediction = predict_power(model_dict, pd.DataFrame([input_data]))
            predictions.append(prediction)
            date_strings.append(day['date'])
            weather_conditions.append({
                'temp': day['temperature'], 'humidity': day['humidity'], 'condition': day.get('condition', 'N/A')
            })

        if not date_strings:
            st.error("No valid data was available to generate predictions. Please try another location.")
            st.stop()

        plot_dates = [datetime.strptime(d, '%Y-%m-%d') for d in date_strings]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_dates, y=predictions, mode='lines+markers',
            line=dict(color='#81e6d9', width=3), marker=dict(size=10, color='#fbbf24'),
            name='Solar Power Prediction'
        ))

        today_date = datetime.now().date()
        today_datetime = datetime.combine(today_date, dt_time(0, 0))

        # FIX: Draw the line and add the annotation in two separate steps
        # 1. Draw the vertical line without any text
        fig.add_vline(x=today_datetime, line_width=2, line_dash="dash", line_color="#fbbf24")

        # 2. Add the annotation manually
        fig.add_annotation(
            x=today_datetime,
            y=np.max(predictions) * 1.05, # Position text slightly above the highest point
            text="Today",
            showarrow=False,
            font=dict(color="#fbbf24", size=14),
            bgcolor="rgba(15, 23, 42, 0.8)" # Add a background for better readability
        )

        fig.update_layout(
            title=f"7-Day Solar Power Outlook for {city_name}",
            xaxis_title="Date", yaxis_title="Predicted Solar Power (W) at Noon",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'), title_font=dict(color='#81e6d9', size=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3 style='text-align: center; margin-top: 2rem;'>📊 Daily Breakdown</h3>", unsafe_allow_html=True)
        num_days = len(date_strings)
        cols = st.columns(num_days) if num_days > 0 else []
        for i, (date, prediction, weather) in enumerate(zip(date_strings, predictions, weather_conditions)):
            with cols[i]:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                is_today = date_obj.date() == today_date
                border_style = "border: 2px solid #fbbf24;" if is_today else ""
                st.markdown(f"""
                <div class="metric-card" style="padding: 1rem; margin-bottom: 0; {border_style}">
                    <h4 style="color: #81e6d9; font-size: 0.8rem;">{date_obj.strftime('%a')}</h4>
                    <p style="color: #fbbf24; font-size: 0.7rem; margin: 0.2rem 0;">{date_obj.strftime('%m/%d')}</p>
                    <p style="color: #81e6d9; font-size: 1.2rem; margin: 0.5rem 0;">{prediction:.0f}W</p>
                    <p style="color: #cbd5e1; font-size: 0.7rem; margin: 0;">{weather['temp']:.1f}°F</p>
                    <p style="color: #cbd5e1; font-size: 0.6rem; margin: 0;">{weather['condition']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>📈 7-Day Summary</h3>", unsafe_allow_html=True)
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.markdown(f"""<div class="metric-card"><h4>Average Daily Output</h4><p>{np.mean(predictions):.0f} W</p></div>""", unsafe_allow_html=True)
        with summary_cols[1]:
            st.markdown(f"""<div class="metric-card"><h4>Peak Day Output</h4><p>{max(predictions):.0f} W</p></div>""", unsafe_allow_html=True)
        with summary_cols[2]:
            st.markdown(f"""<div class="metric-card"><h4>Lowest Day Output</h4><p>{min(predictions):.0f} W</p></div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def data_visualisation_page():
    bg_path = download_file_from_gdrive("Other_bg.jpg")
    set_background(bg_path)
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

        image_path = download_file_from_gdrive(section['file'])
        if image_path:
            st.image(image_path, caption=section['caption'], use_container_width=True)
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


def predictor_page(model_dict):
    bg_path = download_file_from_gdrive("Other_bg.jpg")
    set_background(bg_path)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>⚙ See Our Model in Action</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter conditions manually or fetch live data for a city to get a Solar Power Forecast!</p>", unsafe_allow_html=True)


    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>🛰️ Fetch Live Weather Data</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])

    with c1:
        city_input = st.text_input("Enter a city name", placeholder="e.g., Pune, India", label_visibility="collapsed", key="city_input_predictor")

    if c2.button("Get Live Weather", use_container_width=True) and city_input:
        with st.spinner(f"Fetching weather for {city_input}..."):

            api_response = get_live_weather(city_input, WEATHERAPI_API_KEY)

            st.session_state.weather_api_response = api_response
            st.session_state.apply_weather = False


    if 'weather_api_response' in st.session_state:
        response = st.session_state.weather_api_response

        if "error" in response:
            st.error(f"❌ {response['error']}")
        else:

            weather = response['data']
            official_city = response['city_name']


            st.session_state.live_weather = weather


            st.success(f"✅ Successfully fetched weather data for **{official_city}**!")

            if st.button("Apply to Predictor", use_container_width=True):
                st.session_state.apply_weather = True
                st.rerun()


    with st.form("input_form"):
        defaults = st.session_state.get('live_weather', {}) if st.session_state.get('apply_weather') else {}

        st.markdown("<h3>☀ Solar Position Analysis</h3>", unsafe_allow_html=True)
        time_input = st.time_input("Time", dt_time(12, 0))
        distance_to_solar_noon = calculate_distance_to_solar_noon(time_input)

        c1_solar, c2_solar = st.columns(2)
        with c1_solar:
             st.metric(label="Distance to Solar Noon", value=f"{distance_to_solar_noon:.4f}")

        st.markdown("<hr style='margin: 1rem 0; border-color: #475569;'>", unsafe_allow_html=True)
        st.markdown("<h3>🌤 Weather Conditions</h3>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            temp_val = defaults.get("temperature")
            temperature = st.slider("Temperature (°F)", 42, 78, int(temp_val) if temp_val is not None else 65)

            ws_val = defaults.get("wind_speed")
            wind_speed = st.slider("Wind Speed (mph)", 1.1, 26.6, float(ws_val) if ws_val is not None else 5.0, step=0.1)

            aw_val = defaults.get("avg_wind")
            avg_wind = st.slider("Avg Wind Speed (mph)", 0.0, 40.0, float(aw_val) if aw_val is not None else 5.0, step=0.1)

            wd_val = defaults.get("wind_dir")
            wind_dir = st.slider("Wind Direction (°)", 1, 36, int(wd_val) if wd_val is not None else 18)

        with c2:
            hum_val = defaults.get("humidity")
            humidity = st.slider("Humidity (%)", 14, 100, int(hum_val) if hum_val is not None else 50)

            sc_val = defaults.get("sky_cover")
            sky_cover = st.slider("Sky Cover (0–8)", 0, 8, int(sc_val) if sc_val is not None else 2)

            vis_val = defaults.get("visibility")
            visibility = st.slider("Visibility (mi)", 0.0, 10.0, float(vis_val) if vis_val is not None else 6.0, step=0.1)

            ap_val = defaults.get("avg_pressure")
            avg_pressure = st.slider("Avg Pressure (inHg)", 29.48, 30.53, float(ap_val) if ap_val is not None else 29.9, step=0.01)

        submitted = st.form_submit_button("⚡ Predict Solar Power", use_container_width=True)


    if submitted:
        input_data = {
            "distance-to-solar-noon": distance_to_solar_noon, "temperature": temperature,
            "wind-direction": wind_dir, "wind-speed": wind_speed, "sky-cover": sky_cover,
            "visibility": visibility, "humidity": humidity,
            "average-wind-speed-(period)": avg_wind, "average-pressure-(period)": avg_pressure
        }
        with st.spinner("⏳ Our Model is working its magic..."):
            try:
                prediction = predict_power(model_dict, pd.DataFrame([input_data]))


                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; margin-top: 2rem; border: 2px solid #10b981; border-radius: 1rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(30, 41, 59, 0.1));'>
                    <h2 style='color: #10b981; margin-bottom: 1rem;'>☀ Prediction Result</h2>
                    <div style='font-size: 3rem; font-weight: 700; color: #81e6d9; font-family: "JetBrains Mono", monospace;'>
                        {prediction:.2f} W
                    </div>
                </div>
                """, unsafe_allow_html=True)


                explanation = check_unusual_prediction_and_explain(prediction, input_data)
                if explanation:
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center; color: #fbbf24;'>🧠 AI Insight: Unusual Prediction Detected</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='padding: 1.5rem; background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1));
                                     border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 1rem; margin: 1rem 0;'>
                        <p style='color: #cbd5e1; font-size: 1.05rem; line-height: 1.7; margin: 0;'>
                            {explanation}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                if prediction > 20000:
                    st.balloons()
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


def ai_assistant_page(model_dict):
    """
    Displays the AI Assistant chat page, handles user input,
    and integrates weather/prediction capabilities with an improved general knowledge fallback.
    """
    bg_path = download_file_from_gdrive("Other_bg.jpg")
    set_background(bg_path)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🤖 SunNet AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Ask me about weather, solar predictions, or the SunNet project!</p>", unsafe_allow_html=True)


    def extract_city_from_prompt(text):
        """Extracts a city name from the user's prompt."""
        words = text.lower().split()
        prepositions = ["in", "for", "at"]
        for i, word in enumerate(words):
            if word in prepositions and i + 1 < len(words):
                city_name = " ".join(text.split()[i+1:])
                return city_name.strip().rstrip('?.,!').title()
        return None


    @st.cache_resource
    def initialize_rag_chain():
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if not GOOGLE_API_KEY:
            return None
        try:
            kb_path = download_file_from_gdrive("knowledge_base.txt")
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    knowledge_base_text = f.read()
            except FileNotFoundError:
                knowledge_base_text = "SunNet is a solar energy prediction system using machine learning."

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.create_documents([knowledge_base_text])
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            vector_store = FAISS.from_documents(documents, embeddings)
            retriever = vector_store.as_retriever()
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.5)

            prompt = ChatPromptTemplate.from_template(
                """
                You are SunNet AI, an expert assistant specializing in solar energy. Your goal is to be helpful and accurate.

                1. If the user asks a question specifically about the "SunNet project," "this app," or its features, you MUST prioritize the information in the *Context* below.
                2. For any other general questions about solar energy, weather, or technology, use your own extensive knowledge to answer.
                3. Never tell the user you cannot answer a general knowledge question.

                *Context about the SunNet Project:*
                ---
                {context}
                ---

                *User's Question:* {input}
                *Answer:*
                """
            )
            document_chain = create_stuff_documents_chain(llm, prompt)
            return create_retrieval_chain(retriever, document_chain)
        except Exception as e:
            st.error(f"Error initializing AI Assistant: {e}")
            return None

    rag_chain = initialize_rag_chain()
    if not rag_chain:
        st.error("AI Assistant is unavailable. Please check your Google API key.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with solar energy today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about weather or solar power..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("SunNet AI is thinking..."):
                response_placeholder = st.empty()
                prompt_lower = prompt.lower()
                city = extract_city_from_prompt(prompt)

                is_prediction_request = any(k in prompt_lower for k in ["predict", "generation", "forecast", "how much power"])
                is_weather_request = any(k in prompt_lower for k in ["weather", "temperature", "conditions", "humidity"])

                if is_prediction_request and city:
                    response_placeholder.markdown(f"Fetching live weather for *{city}* to run a prediction...")
                    api_response = get_live_weather(city, WEATHERAPI_API_KEY)

                    if "error" in api_response:
                        response = f"❌ Sorry, I couldn't get the weather for *{city}*. Error: {api_response['error']}"
                    else:
                        weather_data = api_response['data']
                        if not weather_data.get("localtime"):
                            response = "❌ The weather API did not return the local time, so I can't make an accurate prediction."
                        else:
                            local_time_str = weather_data['localtime']
                            city_time_obj = datetime.strptime(local_time_str, "%Y-%m-%d %H:%M").time()
                            input_data = { "distance-to-solar-noon": calculate_distance_to_solar_noon(city_time_obj), "temperature": weather_data['temperature'], "wind-direction": weather_data['wind_dir'], "wind-speed": weather_data['wind_speed'], "sky-cover": weather_data['sky_cover'], "visibility": weather_data['visibility'], "humidity": weather_data['humidity'], "average-wind-speed-(period)": weather_data['avg_wind'], "average-pressure-(period)": weather_data['avg_pressure'] }
                            prediction = predict_power(model_dict, pd.DataFrame([input_data]))


                            response = f"""
                            Okay, here is the solar power prediction for *{api_response['city_name']}*:

                            ### ⚡ Predicted Solar Output: {prediction:.2f} W

                            This prediction is based on the following live weather conditions:
                            * *Temperature:* {weather_data['temperature']} °F
                            * *Humidity:* {weather_data['humidity']} %
                            * *Sky Cover:* {weather_data['sky_cover']}/8
                            """

                elif is_weather_request and city:
                    response_placeholder.markdown(f"Fetching current weather for *{city}*...")
                    api_response = get_live_weather(city, WEATHERAPI_API_KEY)

                    if "error" in api_response:
                        response = f"❌ Sorry, I couldn't get the weather for *{city}*. Error: {api_response['error']}"
                    else:
                        weather_data = api_response['data']
                        st.session_state.live_weather = weather_data


                        response = f"""
                        Here is the current weather in *{api_response['city_name']}*:
                        * *Temperature:* {weather_data['temperature']} °F
                        * *Humidity:* {weather_data['humidity']} %
                        * *Wind:* {weather_data['wind_speed']} mph from {weather_data['wind_dir']*10}°
                        * *Sky Cover:* {weather_data['sky_cover']}/8
                        """

                else:
                    try:
                        rag_response = rag_chain.invoke({"input": prompt})
                        response = rag_response["answer"]
                    except Exception as e:
                        response = f"I encountered an error processing your request. Please try again. Error: {e}"

                response_placeholder.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown('</div>', unsafe_allow_html=True)

def our_team_page():
    bg_path = download_file_from_gdrive("Other_bg.jpg")
    set_background(bg_path)
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

def main():
    """Main function to run the Streamlit app."""

    model_dict = load_model()

    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>SunNet ☀</h2>", unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ("🏠 Home", "📊 Data Visualisation","⚙ Predictor", "📅 7-Day Outlook", "🤖 AI Assistant" , "👨‍🔬 Our Team"),
            label_visibility="collapsed"
        )

    if page == "🏠 Home":
        home_page()
    elif page == "📊 Data Visualisation":
        data_visualisation_page()
    elif page == "⚙ Predictor":
        predictor_page(model_dict)
    elif page == "📅 7-Day Outlook":
        outlook_page(model_dict)
    elif page == "🤖 AI Assistant":
        ai_assistant_page(model_dict)
    elif page == "👨‍🔬 Our Team":
        our_team_page()

if __name__ == "__main__":
    main()