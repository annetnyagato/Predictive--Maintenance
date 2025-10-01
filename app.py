import streamlit as st
import pandas as pd
import joblib

st.title("📡 Smart Predictive Maintenance - Base Station")

# Load model
model = joblib.load("rf_model.joblib")

st.sidebar.header("Input Base Station Parameters")

# User inputs (match dataset features)
traffic_load = st.sidebar.number_input("Traffic Load (Erlangs)", min_value=0.0, max_value=200.0, value=50.0)
ambient_temp = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0)
battery_voltage = st.sidebar.number_input("Battery Voltage (V)", min_value=40.0, max_value=55.0, value=48.0)
psu_temp = st.sidebar.number_input("PSU Temperature (°C)", min_value=20.0, max_value=80.0, value=45.0)
fan_speed = st.sidebar.number_input("Fan Speed (RPM)", min_value=1000.0, max_value=6000.0, value=3000.0)
power_consumption = st.sidebar.number_input("Power Consumption (kWh)", min_value=0.0, max_value=50.0, value=10.0)
site_id = st.sidebar.selectbox("Site ID", [1,2,3,4,5])

# Create dataframe for prediction
input_df = pd.DataFrame({
    "site_id": [site_id],
    "traffic_load": [traffic_load],
    "ambient_temp": [ambient_temp],
    "battery_voltage": [battery_voltage],
    "psu_temp": [psu_temp],
    "fan_speed": [fan_speed],
    "power_consumption": [power_consumption]
})

st.write("### Input Data")
st.write(input_df)

# Predict
prob = model.predict_proba(input_df)[0,1]
pred = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.write("Failure Risk Probability:", round(prob, 2))
if pred == 1:
    st.error("⚠️ High Risk of Failure - Maintenance Recommended")
else:
    st.success("✅ Normal Operation - No Immediate Risk")
