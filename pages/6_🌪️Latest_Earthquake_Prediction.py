from pathlib import Path

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

from utils import (
    load_model,
    load_cluster_model,
    prepare_prediction_input,
    prediction_label,
    cluster_profile_label,
)

st.set_page_config(
    page_title="FeltQuake | Latest Earthquake",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — same tokens as app.py so every page matches
# ---------------------------------------------------------------------------
st.markdown("""
<style>

    .stApp {
        background-color: #0E1B20;
        color: #F4F7F8;
    }

    section[data-testid="stSidebar"] {
        background-color: #101F25;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
        color: #F4F7F8;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #AFC0C5;
        line-height: 1.6;
        margin-bottom: 30px;
    }

    .metric-card {
        background-color: #172A31;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #29434B;
        text-align: center;
        transition: all 0.3s ease-in-out;
        cursor: pointer;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .metric-card:hover {
        transform: translateY(-6px);
        border-color: #E27D45;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        background-color: #1A313A;
    }

    .metric-title {
        color: #9FB1B7;
        font-size: 14px;
        margin-bottom: 8px;
    }

   .metric-value {
    color: #F4F7F8;
    font-size: 22px;
    font-weight: 700;
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .insight-box {
        background-color: #172A31;
        border: 1px solid #29434B;
        border-left: 5px solid #E27D45;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }

    .insight-box:hover {
        transform: translateY(-6px);
        border-color: #E27D45;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        background-color: #1A313A;
    }

    .result-label {
        color: #9FB1B7;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .result-value {
        color: #F4F7F8;
        font-size: 26px;
        font-weight: 700;
    }

    .result-sub {
        color: #AFC0C5;
        font-size: 13px;
        margin-top: 6px;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEED_URLS = {
    "Past Hour": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
    "Past Day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    "Past Week": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson",
}

# Exact training feature order (target `reported` excluded)
CLASSIFIER_FEATURES = [
    "mag", "status", "tsunami", "sig", "net", "nst", "dmin", "rms", "gap",
    "magType", "type", "longitude", "latitude", "depth",
    "hour", "month", "day_of_week", "mag_depth_ratio", "is_remote",
]
CLUSTER_FEATURES = ["mag", "longitude", "latitude", "depth", "mag_depth_ratio", "is_remote"]


# ---------------------------------------------------------------------------
# Data fetching — same logic as latest_earthquake.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner="Fetching latest earthquake data from USGS...")
def fetch_latest_earthquakes(feed_url: str) -> pd.DataFrame:

    response = requests.get(feed_url, timeout=10)
    response.raise_for_status()
    data = response.json()

    rows = []

    for feature in data["features"]:

        props = feature["properties"]
        lon, lat, depth = feature["geometry"]["coordinates"]

        rows.append({
            "place": props.get("place"),
            "mag": props.get("mag"),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
            "time": pd.to_datetime(props.get("time"), unit="ms", utc=True),
            "felt_reports": props.get("felt"),
            "status": props.get("status"),
            "tsunami": props.get("tsunami"),
            "sig": props.get("sig"),
            "net": props.get("net"),
            "nst": props.get("nst"),
            "dmin": props.get("dmin"),
            "rms": props.get("rms"),
            "gap": props.get("gap"),
            "magType": props.get("magType"),
            "type": props.get("type"),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("time", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering — same logic as latest_earthquake.py
# ---------------------------------------------------------------------------
def compute_is_remote(place: str) -> int:

    if not place:
        return 1

    return int("km" not in place.lower())


def engineer_features(row: pd.Series) -> pd.Series:

    row = row.copy()
    row["mag_depth_ratio"] = row["mag"] / (row["depth"] + 1)
    row["is_remote"] = compute_is_remote(row["place"])
    row["month"] = row["time"].month
    row["hour"] = row["time"].hour
    row["day_of_week"] = row["time"].dayofweek

    return row


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">🌍 Latest Earthquake</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-subtitle">
        Live from the USGS feed
        <br>
        The most recent earthquake on record, run through both FeltQuake models in real time.
    </div>
    """,
    unsafe_allow_html=True,
)

feed_choice = st.radio("Feed window", list(FEED_URLS.keys()), index=1, horizontal=True)
df = fetch_latest_earthquakes(FEED_URLS[feed_choice])

if df.empty:
    st.info("No earthquakes recorded in this window. Try a wider feed (e.g. Past Week).")
    st.stop()

latest = engineer_features(df.iloc[0])

# --- Summary cards -----------------------------------------------------------
cols = st.columns(4)
card_data = [
    ("MAGNITUDE", f"{latest['mag']:.1f}"),
    ("DEPTH", f"{latest['depth']:.1f} km"),
    ("LOCATION", latest["place"]),
    ("TIME (UTC)", latest["time"].strftime("%Y-%m-%d %H:%M")),
]

for col, (title, value) in zip(cols, card_data):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value" title="{value}">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if pd.notna(latest["felt_reports"]):
    st.markdown(
        f"""
        <div class="insight-box">
            📣 <strong>{int(latest['felt_reports'])}</strong> people already reported feeling this
            earthquake on USGS "Did You Feel It?".
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Map -----------------------------------------------------------------------
fig = px.scatter_geo(
    pd.DataFrame([latest]),
    lat="latitude", lon="longitude",
    hover_name="place",
)
fig.update_traces(marker=dict(size=max(latest["mag"], 1) * 6, color="#E27D45"))
fig.update_geos(
    projection_type="natural earth",
    center=dict(lat=latest["latitude"], lon=latest["longitude"]),
    projection_scale=4,
    resolution=50,
    showland=True, landcolor="#172A31",
    showocean=True, oceancolor="#0E1B20",
    showcountries=True, countrycolor="#29434B",
    showcoastlines=True, coastlinecolor="#29434B",
    bgcolor="#0E1B20",
)
fig.update_layout(
    height=380,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="#0E1B20",
)
st.plotly_chart(fig, use_container_width=True)

# --- Model analysis --------------------------------------------------------------
st.markdown('<div class="section-title">🔮 Model Predictions</div>', unsafe_allow_html=True)

# Build the row exactly as the training pipeline expects, then apply the
# SAME rare-category grouping + dtype casting used during training (utils.py)
X_live = pd.DataFrame([{col: latest[col] for col in CLASSIFIER_FEATURES}])
X_live = prepare_prediction_input(X_live)

model = load_model()
scaler, kmeans = load_cluster_model()

analysis_cols = st.columns(2)

with analysis_cols[0]:
    proba = model.predict_proba(X_live)[0][1]
    pred = model.predict(X_live)[0]

    st.markdown(
        f"""
        <div class="insight-box">
            <div class="result-label">WILL IT BE FELT?</div>
            <div class="result-value">{prediction_label(pred)} — {proba * 100:.1f}%</div>
            <div class="result-sub">Tuned XGBoost classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(min(max(proba, 0.0), 1.0)))

with analysis_cols[1]:
    cluster_input = X_live[CLUSTER_FEATURES]
    scaled = scaler.transform(cluster_input)
    cluster_id = kmeans.predict(scaled)[0]

    st.markdown(
        f"""
        <div class="insight-box">
            <div class="result-label">SEISMIC PROFILE</div>
            <div class="result-value">{cluster_profile_label(cluster_id)}</div>
            <div class="result-sub">K-Means clustering (k=3)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Recent earthquakes table -----------------------------------------------------
with st.expander("See all recent earthquakes in this window"):
    st.dataframe(
        df[["time", "place", "mag", "depth", "felt_reports"]],
        use_container_width=True, hide_index=True,
    )