import streamlit as st
import os
import joblib
import pandas as pd

st.set_page_config(page_title="Debug Mode")

st.title("🛠 Streamlit Debug Mode")

st.write("### Files in current directory:")
st.write(os.listdir())

try:
    st.write("Loading model...")
    model = joblib.load("project_model.pkl")
    st.success("✅ Model loaded")

    st.write("Loading scaler...")
    scaler = joblib.load("project_scaler.pkl")
    st.success("✅ Scaler loaded")

except Exception as e:
    st.error("❌ Error while loading files")
    st.exception(e)
    st.stop()

st.success("🎉 App started successfully!")
