import streamlit as st
import pandas as pd
import numpy as np
from utils import load_model, prepare_prediction_input

st.set_page_config(
    page_title="Felt Prediction",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0E1B20;
}
</style>
""", unsafe_allow_html=True)

model = load_model()

st.title("🤖 Will This Earthquake Be Felt?")

st.markdown(
    """
    Enter the characteristics of an earthquake and let the
    trained **Tuned XGBoost model** estimate the probability
    that it will be reported as felt.
    """
)

st.info(
    "The prediction follows the same feature structure and "
    "preprocessing pipeline used during model training."
)

st.header("🌋 Earthquake Characteristics")

st.subheader("Physical Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    mag = st.number_input(
        "Magnitude",
        min_value=-2.0,
        max_value=10.0,
        value=4.0,
        step=0.1
    )

with col2:
    depth = st.number_input(
        "Depth (km)",
        min_value=-10.0,
        max_value=800.0,
        value=10.0,
        step=1.0
    )

with col3:
    tsunami = st.selectbox(
        "Tsunami",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.subheader("📍 Location")

col1, col2, col3 = st.columns(3)

with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=0.0,
        step=0.01
    )

with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=0.0,
        step=0.01
    )

with col3:
    is_remote_label = st.selectbox(
        "Location Type",
        options=["Near Location", "Remote"]
    )

is_remote = 1 if is_remote_label == "Remote" else 0

st.subheader("⚙️ Event Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    status = st.selectbox(
        "Status",
        options=["automatic", "reviewed"]
    )

with col2:
    event_type = st.selectbox(
        "Event Type",
        options=["earthquake", "other"]
    )

with col3:
    mag_type = st.selectbox(
        "Magnitude Type",
        options=["mb", "ml", "md", "mww", "mw", "other"]
    )

with col4:
    net = st.text_input(
        "Network",
        value="us"
    )

st.subheader("🕐 Time Features")

col1, col2, col3 = st.columns(3)

with col1:
    hour = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=12
    )

with col2:
    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

with col3:
    day_of_week = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2,
        help="0 = Monday, 6 = Sunday"
    )

st.subheader("📡 Seismic Measurements")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sig = st.number_input(
        "Significance (sig)",
        min_value=0.0,
        max_value=5000.0,
        value=100.0,
        step=1.0
    )

with col2:
    nst = st.number_input(
        "Number of Stations (nst)",
        min_value=0.0,
        max_value=500.0,
        value=20.0,
        step=1.0
    )

with col3:
    dmin = st.number_input(
        "Minimum Distance (dmin)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.01
    )

with col4:
    rms = st.number_input(
        "RMS",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.01
    )

col1, col2 = st.columns(2)

with col1:
    gap = st.number_input(
        "Azimuthal Gap",
        min_value=0.0,
        max_value=360.0,
        value=100.0,
        step=1.0
    )

with col2:
    mag_depth_ratio = mag / (depth + 1)

st.caption(f"Calculated Magnitude / Depth Ratio: **{mag_depth_ratio:.4f}**")

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Felt Probability",
    type="primary",
    use_container_width=True
)

if predict_button:

    input_data = pd.DataFrame([{
        "mag": mag,
        "status": status,
        "tsunami": tsunami,
        "sig": sig,
        "net": net,
        "nst": nst,
        "dmin": dmin,
        "rms": rms,
        "gap": gap,
        "magType": mag_type,
        "type": event_type,
        "longitude": longitude,
        "latitude": latitude,
        "depth": depth,
        "hour": hour,
        "month": month,
        "day_of_week": day_of_week,
        "mag_depth_ratio": mag_depth_ratio,
        "is_remote": is_remote
    }])

    input_data = prepare_prediction_input(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.header("🔮 Prediction Result")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        if prediction == 1:
            st.error("🌋 Likely to be Felt")
        else:
            st.success("🌎 Likely Not to be Felt")

    with col_res2:
        st.metric(
            label="Felt Probability",
            value=f"{probability * 100:.1f}%"
        )

    st.progress(float(probability))

    st.caption(
        "The displayed probability is produced by the trained "
        "XGBoost classifier."
    )

    with st.expander("📋 View Input Features"):
        st.dataframe(
            input_data.T.rename(columns={0: "Value"}),
            use_container_width=True
        )