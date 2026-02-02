import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Motor Fault Detection",
    layout="centered"
)

st.title("⚙️ Intelligent Motor Fault Detection System")
st.write("Machine Learning–based motor health and fault prediction")

# ---------------- DEBUG (SAFE) ----------------
st.write("📂 Project files:")
st.write(os.listdir())

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("project_model.pkl")
    scaler = joblib.load("project_scaler.pkl")
    st.success("✅ Model and scaler loaded successfully")
except Exception as e:
    st.error("❌ Failed to load model or scaler")
    st.exception(e)
    st.stop()

# ---------------- FAULT MAP ----------------
fault_map = {
    0: "Healthy",
    1: "Bearing Fault",
    2: "Rotor Imbalance",
    3: "Stator Winding Fault",
    4: "Overheating"
}

# ---------------- HEALTH INDEX ----------------
def health_index(vibration, temperature, current):
    score = 100
    score -= vibration * 5
    score -= (temperature - 50) * 0.7
    score -= abs(current - 10) * 3
    return max(0, min(100, score))

# ---------------- USER INPUT ----------------
st.subheader("🔧 Enter Motor Sensor Values")

vibration = st.slider("Vibration (mm/s)", 0.0, 10.0, 3.0)
temperature = st.slider("Temperature (°C)", 20, 120, 70)
current = st.slider("Current (A)", 5.0, 20.0, 12.0)
voltage = st.slider("Voltage (V)", 350, 450, 400)
speed = st.slider("Speed (RPM)", 1000, 1800, 1450)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Motor Condition"):
    input_data = pd.DataFrame([{
        "Vibration_mm_s": vibration,
        "Temperature_C": temperature,
        "Current_A": current,
        "Voltage_V": voltage,
        "Speed_RPM": speed
    }])

    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        chi = health_index(vibration, temperature, current)

        st.subheader("📊 Results")
        st.write("**Predicted Fault:**", fault_map[prediction])
        st.write(f"**Health Index:** {chi:.2f}%")

        if prediction != 0:
            st.error("⚠ Motor Fault Detected")
        elif chi < 60:
            st.warning("⚠ Motor Health Degrading")
        else:
            st.success("✅ Motor Operating Normally")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
