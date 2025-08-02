# Import the libraries we need
import streamlit as st
import pandas as pd
import numpy as np

# This sets up the page title and icon that appears in the browser tab
st.set_page_config(page_title="Solar Energy Predictor", page_icon="☀️")

# Create the main title for our webpage
st.title("☀️ Solar Energy Prediction Tool")

# Add some space and a description
st.write("")
st.write("This tool helps you predict solar energy generation based on weather conditions.")

# Create a divider line
st.write("---")


# Create two columns to organize the inputs 
left_column, right_column = st.columns(2)

#Taking inputs

st.title("Solar Energy Prediction - Input Parameters")

# Temperature (°C)
temperature = st.slider("🌡️ Temperature (°C)", 0, 100, 25)
step=1

#distance to solar noon
dist_to_noon=st.slider("Distance to solar noon: ",0.0,1.0,0.4322)
step=0.1

# Humidity (%)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
step=1

# Wind Speed (km/h)
wind_speed = st.slider("💨 Wind Speed (km/h)", 0, 50, 10)
step=0.25

# Wind Direction (degrees)
wind_direction = st.slider("🧭 Wind Direction (°)", 0, 360, 180)
step=1

# Sky Cover (%)
sky_cover = st.slider("☁️ Sky Cover (%)", 0, 100, 40)
step=1

# Visibility (km)
visibility = st.slider("👀 Visibility (km)", 0, 50, 10)
step=1

# Average Wave Power (kW/m)
avg_wave_power = st.slider("🌊 Average Wave Power (kW/m)", 0.0, 100.0, 10.0)
step=2

# Show input summary
st.subheader("Your Input Summary:")
st.write(f"🌡️ Temperature: {temperature} °C")
st.write(f"💧 Humidity: {humidity} %")
st.write(f"💨 Wind Speed: {wind_speed} km/h")
st.write(f"🧭 Wind Direction: {wind_direction} °")
st.write(f"☁️ Sky Cover: {sky_cover} %")
st.write(f"👀 Visibility: {visibility} km")
st.write(f"🌊 Avg Wave Power: {avg_wave_power} kW/m")
st.write(f"Distance_to_solar noon: {dist_to_noon}")


# Add some space
st.write("")

# Create a big prediction button
# When someone clicks this button, it will run the code inside the if statement
if st.button("🔮 Predict Solar Energy Generation", type="primary"):
    
    # This is where we would normally use a trained machine learning model
    # For now, creating a simple formula that uses the input values
    # In a real project, load a pre-trained model here
    
    # Simple prediction formula (this is just for demonstration)
    
    predicted_energy = (
        (100 - humidity * 0.5) +           # Less humidity = more energy
        (temperature * 0.3) +              # Warmer = more energy
        ((10 - sky_cover) * 15) +          # Less clouds = more energy  
        (visibility * 2) +                 # Better visibility = more energy
        (avg_wave_power * 3) +         # More wave power = more energy
        (wind_speed * 1.5) -               # Some wind helps cooling
        (dist_to_noon * 20)      # Closer to noon = more energy
    )
    
    # Make sure the prediction isn't negative
    if predicted_energy < 0:
        predicted_energy = 0
    
    # Show the results
    st.write("---")
    st.header("📊 Prediction Results")
    
    # Create three columns to show different metrics
    result_col1, result_col2= st.columns(2)
    
    # Show the main prediction
    with result_col1:
        st.metric(
            label="⚡ Predicted Energy Generation", 
            value=f"{predicted_energy:.2f} kWh",
            help="Estimated solar energy generation"
        )
    
    # Show some additional info
    with result_col2:
        # Calculate efficiency based on conditions
        efficiency = min(100, max(10, predicted_energy * 2))
        st.metric(
            label="🎯 System Efficiency", 
            value=f"{efficiency:.1f}%",
            help="How efficiently the solar panels are working"
        )
    
   
    
    # Some explanatory text
    st.write("")
    st.info("💡 **How this works:** The prediction is based on weather conditions that affect solar panel performance. Sunny, clear days with moderate temperatures produce the most energy!")
    
    #Factors are helping or hurting the prediction
    st.write("")
    st.subheader("📈 Factors Affecting Your Prediction:")
    
    #Simple analysis of the input conditions
    factors = []
    
    if sky_cover <= 2:
        factors.append("✅ Clear skies are great for solar energy!")
    elif sky_cover >= 7:
        factors.append("⛅ Heavy cloud cover will reduce energy generation")
    
    if temperature >= 20 and temperature <= 30:
        factors.append("✅ Temperature is in the optimal range for solar panels")
    elif temperature > 35:
        factors.append("🌡️ Very hot weather can reduce panel efficiency")
    
    if humidity <= 50:
        factors.append("✅ Low humidity provides excellent conditions")
    elif humidity >= 80:
        factors.append("💧 High humidity may reduce solar output")
    
    if visibility >= 15:
        factors.append("✅ Excellent visibility means clear atmospheric conditions")
    
    if dist_to_noon <= 1:
        factors.append("☀️ Close to solar noon - peak sun time!")
    
    # Display all the factors
    for factor in factors:
        st.write(factor)
    
    # If no specific factors were identified, show a general message
    if not factors:
        st.write("🌤️ Conditions are moderate - expect normal solar generation")

# Add some space at the bottom
st.write("")
st.write("---")

# Add footer information
st.write("")
st.write("**About this tool:** This solar energy predictor uses weather data to estimate how much electricity your solar panels might generate. The prediction is based on factors like temperature, cloud cover, and time of day.")

