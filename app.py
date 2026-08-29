import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="FeltQuake | Earthquake AI",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        font-size: 55px;
        font-weight: 800;
        line-height: 1.1;
        color: #F4F7F8;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #AFC0C5;
        line-height: 1.6;
        margin-bottom: 35px;
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
        transform: translateY(-8px);
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
        font-size: 28px;
        font-weight: 700;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .insight-box {
        background-color: #172A31;
        border-left: 5px solid #E27D45;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
    }

    .workflow {
        background-color: #172A31;
        padding: 25px 15px;
        border-radius: 15px;
        border: 1px solid #29434B;
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }

    .workflow:hover {
        transform: translateY(-10px);
        border-color: #E27D45;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        background-color: #1A313A;
    }

    .workflow-icon {
        font-size: 36px;
        margin-bottom: 10px;
        transition: transform 0.3s ease;
    }

    .workflow:hover .workflow-icon {
        transform: scale(1.2);
    }

    .workflow-title {
        font-weight: 700;
        font-size: 16px;
        color: #F4F7F8;
        margin-bottom: 5px;
    }

    .workflow-text {
        color: #AFC0C5;
        font-size: 13px;
        line-height: 1.4;
    }

</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "earthquakes_cleaned.csv"
df = pd.read_csv(data_path)

with st.sidebar:
    st.markdown("## 🌋 FeltQuake")
    st.markdown(
        """
        **Earthquake Impact Analysis & Prediction**

        Explore earthquake patterns and predict whether
        an earthquake is likely to be felt by people.
        """
    )
    st.markdown("---")
    st.markdown("### 📡 Data Source")
    st.markdown(
        """
        **USGS Earthquake Catalog**

        The project uses earthquake event data collected
        from the USGS earthquake API.
        """
    )
    st.markdown("---")
    st.caption("Machine Learning Project")
    st.caption("Tuned XGBoost Classifier")

st.markdown(
    '<div class="hero-title">🌋 FeltQuake</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
        Earthquake Impact Analysis & Prediction
        <br>
        Understanding what makes an earthquake felt —
        and using Machine Learning to predict it.
    </div>
    """,
    unsafe_allow_html=True
)

total_events = len(df)
felt_count = int(df["reported"].sum())
felt_percentage = felt_count / total_events * 100
avg_magnitude = df["mag"].mean()

cols = st.columns(4)

with cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">TOTAL EARTHQUAKES</div>
            <div class="metric-value">{total_events:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols[1]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">FELT EARTHQUAKES</div>
            <div class="metric-value">{felt_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols[2]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">FELT RATE</div>
            <div class="metric-value">{felt_percentage:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">AVG MAGNITUDE</div>
            <div class="metric-value">{avg_magnitude:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="section-title">🎯 Project Goal</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    This project investigates the factors associated with whether
    an earthquake is reported as felt by people.

    Instead of relying only on earthquake magnitude, the analysis
    considers several characteristics such as depth, location,
    event type, tsunami association, seismic network and
    time-based features.

    The final stage uses a tuned **XGBoost classifier** to estimate
    the probability that an earthquake will be felt.
    """
)

st.markdown(
    '<div class="section-title">🔬 Project Workflow</div>',
    unsafe_allow_html=True
)

workflow_cols = st.columns(5)

workflow = [
    ("📡", "USGS Data", "Collect earthquake events"),
    ("🧹", "Data Preparation", "Clean & engineer features"),
    ("🔎", "EDA", "Discover earthquake patterns"),
    ("🤖", "Machine Learning", "Train & tune models"),
    ("🔮", "Prediction", "Predict felt probability")
]

for col, item in zip(workflow_cols, workflow):
    icon, title, text = item
    with col:
        st.markdown(
            f"""
            <div class="workflow">
                <div class="workflow-icon">{icon}</div>
                <div class="workflow-title">{title}</div>
                <div class="workflow-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="section-title">💡 The Main Question</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="insight-box">

    <strong>What makes an earthquake more likely to be felt?</strong>

    <br><br>

    The analysis explores whether magnitude, depth, location,
    time and other earthquake characteristics can explain
    differences in human reporting.

    <br><br>

    The final Machine Learning model then turns these patterns
    into a prediction.

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    """
    ### 👈 Explore the project

    Use the navigation menu to:

    - Explore the earthquake data
    - Discover what makes earthquakes felt
    - Try the prediction model
    - Evaluate the final Machine Learning model
    - Discover seismic profiles through clustering
    - See the latest earthquake live
    """
)