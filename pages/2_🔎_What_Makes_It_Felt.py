import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_data


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

st.set_page_config(
    page_title="What Makes It Felt?",
    page_icon="🔎",
    layout="wide"
)


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1B20;
}

.insight {
    background-color: #172A31;
    border-left: 5px solid #E27D45;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 25px;
}

.question {
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

df = load_data()

df["felt_label"] = df["reported"].map({
    0: "Not Felt",
    1: "Felt"
})


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🔎 What Makes an Earthquake Felt?")

st.markdown(
    """
    Magnitude is only one part of the story.

    This analysis explores several earthquake characteristics
    and how they are associated with whether people reported
    feeling the event.
    """
)


# =========================================================
# 1. MAGNITUDE
# =========================================================

st.markdown(
    '<div class="question">💥 1. Do stronger earthquakes get felt more often?</div>',
    unsafe_allow_html=True
)

mag_df = (
    df.groupby("felt_label")["mag"]
    .mean()
    .round(2)
    .reset_index()
)

fig_mag = px.bar(
    mag_df,
    x="felt_label",
    y="mag",
    title="Average Magnitude: Felt vs Not Felt",
    text_auto=True,
    labels={
        "felt_label": "Reported Status",
        "mag": "Average Magnitude"
    }
)

st.plotly_chart(
    fig_mag,
    use_container_width=True
)

st.markdown(
    """
    <div class="insight">

    <strong>💡 Insight</strong><br><br>

    Magnitude is an important earthquake characteristic,
    but magnitude alone does not fully describe whether
    an earthquake will be felt.

    This motivated the creation of the
    <strong>magnitude-to-depth ratio</strong> feature.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 2. DEPTH
# =========================================================

st.markdown(
    '<div class="question">🌎 2. Does earthquake depth affect the likelihood of being felt?</div>',
    unsafe_allow_html=True
)

depth_df = (
    df.groupby("felt_label")["depth"]
    .mean()
    .round(2)
    .reset_index()
)

fig_depth = px.bar(
    depth_df,
    x="felt_label",
    y="depth",
    title="Average Depth: Felt vs Not Felt",
    text_auto=True,
    labels={
        "felt_label": "Reported Status",
        "depth": "Average Depth"
    }
)

st.plotly_chart(
    fig_depth,
    use_container_width=True
)

st.markdown(
    """
    <div class="insight">

    <strong>💡 Insight</strong><br><br>

    The analysis shows that deeper earthquakes are mostly
    associated with earthquakes that were not reported as felt.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. LOCATION
# =========================================================

st.markdown(
    '<div class="question">🏙️ 3. Are earthquakes near populated areas felt more often?</div>',
    unsafe_allow_html=True
)

location_df = (
    df.groupby("is_remote")["reported"]
    .mean()
    .reset_index(name="reporting_rate")
)

location_df["reporting_rate"] *= 100

location_df["location"] = location_df["is_remote"].map({
    False: "Near Location",
    True: "Remote"
})

fig_location = px.bar(
    location_df,
    x="location",
    y="reporting_rate",
    title="Reporting Rate by Location Type",
    text_auto=".2f",
    labels={
        "location": "Location Type",
        "reporting_rate": "Reporting Rate (%)"
    }
)

fig_location.update_yaxes(
    title="Reporting Rate (%)"
)

st.plotly_chart(
    fig_location,
    use_container_width=True
)


st.markdown(
    """
    <div class="insight">

    <strong>💡 Insight</strong><br><br>

    Earthquakes classified as near locations have a substantially
    higher reporting rate than remote events.

    In the analysis, the reporting rate was approximately
    <strong>13.6%</strong> for near locations compared with
    approximately <strong>1.6%</strong> for remote ones.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. TIME
# =========================================================

st.markdown(
    '<div class="question">🕐 4. Does the time of day affect reporting?</div>',
    unsafe_allow_html=True
)

hour_df = (
    df.groupby("hour")["reported"]
    .sum()
    .reset_index()
)

fig_hour = px.bar(
    hour_df,
    x="hour",
    y="reported",
    title="Reported Earthquakes by Hour of Day",
    labels={
        "hour": "Hour of Day",
        "reported": "Reported Earthquakes"
    }
)

fig_hour.update_xaxes(
    dtick=1
)

st.plotly_chart(
    fig_hour,
    use_container_width=True
)


# =========================================================
# 5. MONTH
# =========================================================

st.markdown(
    '<div class="question">📅 5. Does reporting vary by month?</div>',
    unsafe_allow_html=True
)

month_map = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

month_df = (
    df.groupby("month")["reported"]
    .mean()
    .reset_index(name="reporting_rate")
)

month_df["reporting_rate"] *= 100

month_df["month_name"] = month_df["month"].map(month_map)

month_df = month_df.sort_values("month")


fig_month = px.line(
    month_df,
    x="month_name",
    y="reporting_rate",
    markers=True,
    title="Average Reporting Rate by Month",
    labels={
        "month_name": "Month",
        "reporting_rate": "Reporting Rate (%)"
    }
)

fig_month.update_yaxes(
    title="Reporting Rate (%)"
)

st.plotly_chart(
    fig_month,
    use_container_width=True
)


# =========================================================
# 6. TSUNAMI
# =========================================================

st.markdown(
    '<div class="question">🌊 6. Are tsunami-associated earthquakes reported differently?</div>',
    unsafe_allow_html=True
)

tsunami_df = (
    df.groupby("tsunami")["reported"]
    .mean()
    .reset_index(name="reporting_rate")
)

tsunami_df["reporting_rate"] *= 100

tsunami_df["tsunami_label"] = tsunami_df["tsunami"].map({
    0: "No Tsunami",
    1: "Tsunami"
})

fig_tsunami = px.bar(
    tsunami_df,
    x="tsunami_label",
    y="reporting_rate",
    title="Reporting Rate by Tsunami Association",
    text_auto=".2f",
    labels={
        "tsunami_label": "Tsunami Association",
        "reporting_rate": "Reporting Rate (%)"
    }
)

fig_tsunami.update_yaxes(
    title="Reporting Rate (%)"
)

st.plotly_chart(
    fig_tsunami,
    use_container_width=True
)


# =========================================================
# FINAL FINDINGS
# =========================================================

st.markdown("---")

st.header("💡 Key Findings")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        ### 💥 Magnitude

        Stronger earthquakes tend to be associated
        with higher likelihood of being felt, but
        magnitude alone is not sufficient.
        """
    )

with col2:

    st.markdown(
        """
        ### 🌎 Depth & Location

        Depth and proximity to populated locations
        provide additional information about human
        perception of an earthquake.
        """
    )

with col3:

    st.markdown(
        """
        ### 🤖 ML Connection

        These patterns motivated the feature engineering
        and Machine Learning stage of the project.
        """
    )