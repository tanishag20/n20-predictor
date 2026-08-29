import streamlit as st
import numpy as np
import joblib
import os
import urllib.request

MODEL_URL = "https://github.com/tanishag20/n20-predictor/releases/download/v1/rf_model.pkl"
MODEL_PATH = "rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (first run only)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="N2O Predictor", layout="wide")

st.title("🌱 N₂O Emission Prediction System")
st.markdown("Predict nitrous oxide emissions based on process parameters.")

st.divider()

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 Process Variables")
    NH4 = st.number_input("NH4", value=0.0)
    NO3 = st.number_input("NO3", value=0.0)
    O2 = st.number_input("O2 Tank 1", value=0.0)
    O2_set = st.number_input("O2 Setpoint", value=0.0)
    temp = st.number_input("Temperature", value=25.0)

with col2:
    st.subheader("⚙️ Operational Variables")
    airflow = st.number_input("Airflow", value=0.0)
    inlet_flow = st.number_input("Inlet Flow", value=0.0)
    N2O_lag1 = st.number_input("N2O Lag 1", value=0.0)
    N2O_roll6 = st.number_input("N2O Rolling Mean (6)", value=0.0)
    N2O_roll12 = st.number_input("N2O Rolling Mean (12)", value=0.0)

st.divider()

if st.button("🚀 Predict N₂O Emission"):
    
    input_dict = {f: 0 for f in features}

    input_dict["NH4"] = NH4
    input_dict["NO3"] = NO3
    input_dict["O2_tank1"] = O2
    input_dict["O2_setpoint"] = O2_set
    input_dict["temperature"] = temp
    input_dict["airflow"] = airflow
    input_dict["inlet_flow"] = inlet_flow

    input_dict["N2O_lag1"] = N2O_lag1
    input_dict["N2O_roll6_mean"] = N2O_roll6
    input_dict["N2O_roll12_mean"] = N2O_roll12

    X = np.array([list(input_dict.values())])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)

    st.success(f"🌿 Predicted N₂O Emission: {pred[0]:.2f}")

    if pred[0] > 30000:
        st.warning("⚠️ High emission detected — check process conditions")
    else:
        st.info("✅ Emission level within normal range")

st.divider()
st.caption("Built using Random Forest model with time-series feature engineering")