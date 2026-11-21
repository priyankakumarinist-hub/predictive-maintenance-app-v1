import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Engine Failure Prediction App", page_icon="🔧")

# ---------------------------------------------------------------------------
# Colour theme: change these two values to restyle the whole app
# ---------------------------------------------------------------------------
PRIMARY = "#1F5F6B"   # button, headings, number values
ACCENT = "#2E86AB"    # banner gradient end, input highlight

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.banner {
    background: linear-gradient(120deg, __PRIMARY__ 0%, __ACCENT__ 100%);
    padding: 1.3rem 1.6rem; border-radius: 12px; margin-bottom: 1rem;
}
.banner h1 {margin: 0; padding: 0; font-size: 1.9rem; color: #FFFFFF;}
.banner p {margin: .4rem 0 0 0; color: #FFFFFF; opacity: .92;}
h2, h3 {color: __PRIMARY__ !important;}
div[data-baseweb="input"]:focus-within {border-color: __ACCENT__ !important;}
[data-testid="stMetricValue"] {color: __PRIMARY__;}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background-color: __PRIMARY__; border-color: __PRIMARY__; color: #FFFFFF; font-weight: 700;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background-color: __ACCENT__; border-color: __ACCENT__; color: #FFFFFF;
}
</style>
"""
st.markdown(CSS.replace("__PRIMARY__", PRIMARY).replace("__ACCENT__", ACCENT), unsafe_allow_html=True)

# Load the trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_engine_maintenance_model_v1.joblib"
)

model = joblib.load(model_path)

# Application title
st.markdown(
    """
    <div class="banner">
        <h1>🔧 Engine Failure Prediction App</h1>
        <p>This application predicts whether an engine is likely to need maintenance based on its
        sensor readings. Enter the values below and click Predict.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Engine details
st.subheader("Sensor readings")

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
if st.button("Predict Engine Health", type="primary"):

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
