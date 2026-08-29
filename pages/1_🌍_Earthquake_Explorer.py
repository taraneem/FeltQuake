import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_data


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Earthquake Explorer",
    page_icon="🌍",
    layout="wide"
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1B20;
}

.section-box {
    background-color: #172A31;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #29434B;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

df = load_data()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🌍 Earthquake Explorer")

st.markdown(
    """
    Explore the earthquake dataset interactively.
    Use the filters to investigate how earthquake characteristics
    vary across locations and felt status.
    """
)


# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------

st.sidebar.header("🔎 Filters")


# Magnitude
min_mag = float(df["mag"].min())
max_mag = float(df["mag"].max())

mag_range = st.sidebar.slider(
    "Magnitude",
    min_value=min_mag,
    max_value=max_mag,
    value=(min_mag, max_mag)
)


# Depth
min_depth = float(df["depth"].min())
max_depth = float(df["depth"].max())

depth_range = st.sidebar.slider(
    "Depth (km)",
    min_value=min_depth,
    max_value=max_depth,
    value=(min_depth, max_depth)
)


# Felt Status
felt_filter = st.sidebar.multiselect(
    "Felt Status",
    options=["Felt", "Not Felt"],
    default=["Felt", "Not Felt"]
)


# Type
type_options = sorted(df["type"].dropna().unique())

selected_types = st.sidebar.multiselect(
    "Event Type",
    options=type_options,
    default=type_options
)


# Tsunami
tsunami_options = sorted(df["tsunami"].dropna().unique())

selected_tsunami = st.sidebar.multiselect(
    "Tsunami",
    options=tsunami_options,
    default=tsunami_options
)


# ---------------------------------------------------------
# Apply Filters
# ---------------------------------------------------------

filtered_df = df[
    (df["mag"].between(mag_range[0], mag_range[1]))
    &
    (df["depth"].between(depth_range[0], depth_range[1]))
    &
    (df["type"].isin(selected_types))
    &
    (df["tsunami"].isin(selected_tsunami))
].copy()


filtered_df["felt_label"] = filtered_df["reported"].map({
    0: "Not Felt",
    1: "Felt"
})


if felt_filter:

    filtered_df = filtered_df[
        filtered_df["felt_label"].isin(felt_filter)
    ]


# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Events",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "Avg Magnitude",
        f"{filtered_df['mag'].mean():.2f}"
        if len(filtered_df) else "N/A"
    )

with col3:

    felt_rate = (
        filtered_df["reported"].mean() * 100
        if len(filtered_df)
        else 0
    )

    st.metric(
        "Felt Rate",
        f"{felt_rate:.2f}%"
    )

with col4:

    st.metric(
        "Avg Depth",
        f"{filtered_df['depth'].mean():.2f} km"
        if len(filtered_df)
        else "N/A"
    )


st.markdown("---")


# ---------------------------------------------------------
# Earthquake Map
# ---------------------------------------------------------

st.subheader("🌎 Earthquake Map")

if len(filtered_df) > 0:

    fig_map = px.scatter_map(
        filtered_df,
        lat="latitude",
        lon="longitude",
        size="mag",
        size_max=10,
        color="felt_label",
        hover_name="type",
        hover_data={
            "mag": ":.2f",
            "depth": ":.2f",
            "latitude": ":.3f",
            "longitude": ":.3f",
            "felt_label": True
        },
        color_discrete_sequence= ['#3B5E69' , '#6E93A0'],
        zoom=1,
        height=500,
        opacity= 0.6,
        map_style="open-street-map"
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

else:

    st.warning("No earthquakes match the selected filters.")


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------

st.subheader("📊 Explore Earthquake Characteristics")

col1, col2 = st.columns(2)


# Magnitude distribution
with col1:

    fig_mag = px.histogram(
        filtered_df,
        x="mag",
        color="felt_label",
        nbins=40,
        title="Magnitude Distribution"
    )

    st.plotly_chart(
        fig_mag,
        use_container_width=True
    )


# Depth distribution
with col2:

    fig_depth = px.histogram(
        filtered_df,
        x="depth",
        color="felt_label",
        nbins=40,
        title="Depth Distribution"
    )

    st.plotly_chart(
        fig_depth,
        use_container_width=True
    )


# ---------------------------------------------------------
# Event Types
# ---------------------------------------------------------

st.subheader("🌋 Earthquake Event Types")

type_df = (
    filtered_df["type"]
    .value_counts()
    .reset_index()
)

type_df.columns = ["type", "count"]

fig_type = px.bar(
    type_df,
    x="type",
    y="count",
    title="Events by Type"
)

st.plotly_chart(
    fig_type,
    use_container_width=True
)


# ---------------------------------------------------------
# Data Preview
# ---------------------------------------------------------

with st.expander("📋 View Filtered Data"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=350
    )