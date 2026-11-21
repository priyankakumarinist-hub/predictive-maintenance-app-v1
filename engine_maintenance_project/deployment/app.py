
import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_engine_maintenance_model_v1.joblib"
)

model = joblib.load(model_path)

# Application title
st.title("Engine Failure Prediction App")

st.write("""
This application predicts the likelihood of a engine failing based on its operational parameters.
Enter the sensor and configuration data below to get a prediction
""")

# Engine details

Engine_rpm = st.number_input(
    "Engine rpm",
    min_value=61,
    max_value=2239,
    value=746,
    step=1,
)

Lub_oil_pressure = st.number_input(
    "Lub oil pressure",
    min_value=0.0033,
    max_value=7.2656,
    value=3.16,
    step=0.01,
    format="%.4f",
)

Fuel_pressure = st.number_input(
    "Fuel pressure",
    min_value=0.0031,
    max_value=21.1384,
    value=6.20,
    step=0.01,
    format="%.4f",
)

Coolant_pressure = st.number_input(
    "Coolant pressure",
    min_value=0.0024,
    max_value=7.4786,
    value=2.17,
    step=0.01,
    format="%.4f",
)

Lub_oil_temp = st.number_input(
    "Lub oil temp (°C)",
    min_value=71.3219,
    max_value=89.5808,
    value=76.82,
    step=0.01,
    format="%.4f",
)

Coolant_temp = st.number_input(
    "Coolant temp (°C)",
    min_value=61.6733,
    max_value=195.5280,
    value=78.35,
    step=0.01,
    format="%.4f",
)

# Predict engine health
if st.button("Predict Engine Health"):

    # Column names and order must match the data the model was trained on
    input_data = pd.DataFrame([{
        "Engine rpm": Engine_rpm,
        "Lub oil pressure": Lub_oil_pressure,
        "Fuel pressure": Fuel_pressure,
        "Coolant pressure": Coolant_pressure,
        "lub oil temp": Lub_oil_temp,
        "Coolant temp": Coolant_temp,
    }])

    prediction = int(model.predict(input_data)[0])
    fault_probability = float(model.predict_proba(input_data)[0][1])

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("The model predicts: **Faulty** - the engine is likely to need maintenance.")
    else:
        st.success("The model predicts: **Healthy** - the engine appears to be operating normally.")

    st.metric("Estimated probability of a fault", f"{fault_probability:.2%}")
    st.caption(
        "The model is tuned to catch as many faulty engines as possible, so it also "
        "flags some healthy engines. Treat a 'Faulty' result as a prompt to inspect the engine."
    )


