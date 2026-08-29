import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Seismic Profiles",
    page_icon="🧩",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0E1B20;
}
.profile-card {
    background-color: #172A31;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #29434B;
    min-height: 230px;
    transition: all 0.3s ease-in-out;
}
.profile-card:hover {
    transform: translateY(-8px);
    border-color: #E27D45;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    background-color: #1A313A;
}
.profile-title {
    color: #5eead4;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 12px;
}
.profile-stat {
    color: #AFC0C5;
    font-size: 14px;
    margin-bottom: 6px;
}
.insight-box {
    background-color: #172A31;
    border-left: 5px solid #E27D45;
    padding: 20px;
    border-radius: 10px;
    margin-top: 25px;
    color: #F4F7F8;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "earthquakes_clusters.csv")

cluster_names = {
    0: "Shallow & Near (High Impact)",
    1: "Intermediate Depth (Moderate)",
    2: "Deep & Remote (Low Impact)"
}

df["cluster_name"] = df["cluster"].map(cluster_names)

st.title("🧩 Seismic Profiles (Unsupervised Learning)")

st.markdown(
    """
    K-Means clustering grouped the earthquakes into **3 natural profiles**
    based on magnitude, depth, location and remoteness — without using
    the target variable at all.
    """
)

summary = df.groupby("cluster_name").agg(
    events=("reported", "size"),
    felt_rate=("reported", "mean"),
    avg_mag=("mag", "mean"),
    avg_depth=("depth", "mean"),
    remote_pct=("is_remote", "mean")
).round(4)

summary["felt_rate"] = (summary["felt_rate"] * 100).round(4)
summary["remote_pct"] = (summary["remote_pct"] * 100).round(2)

order = [
    "Shallow & Near (High Impact)",
    "Intermediate Depth (Moderate)",
    "Deep & Remote (Low Impact)"
]

summary = summary.reindex(order)

cols = st.columns(3)

for col, name in zip(cols, order):
    row = summary.loc[name]
    with col:
        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-title">{name}</div>
                <div class="profile-stat">🌍 Events: <b>{int(row['events']):,}</b></div>
                <div class="profile-stat">⚡ Avg Magnitude: <b>{row['avg_mag']}</b></div>
                <div class="profile-stat">🌊 Avg Depth: <b>{row['avg_depth']} km</b></div>
                <div class="profile-stat">📍 Remote: <b>{row['remote_pct']}%</b></div>
                <div class="profile-stat">🔔 Felt Rate: <b>{row['felt_rate']}%</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="#0E1B20",
        plot_bgcolor="#0E1B20",
        font_color="#cbd5e1",
        title_font_color="#F4F7F8",
        legend_bgcolor="rgba(0,0,0,0)",
        xaxis_gridcolor="#29434B",
        yaxis_gridcolor="#29434B"
    )
    return fig


fig_bar = px.bar(
    summary.reset_index(),
    x="cluster_name",
    y="felt_rate",
    color="felt_rate",
    color_continuous_scale=["#3B5E99", "#3B5E88" , '#3B5E70'],
    title="Felt Rate per Seismic Profile",
    text="felt_rate"
)

fig_bar.update_layout(
    xaxis_title="Seismic Profile",
    yaxis_title="Felt Rate (%)",
    coloraxis_showscale=False
)

st.plotly_chart(style_fig(fig_bar), use_container_width=True)

sample = df.sample(5000, random_state=42)

fig_scatter = px.scatter(
    sample,
    x="mag",
    y="depth",
    color="cluster_name",
    color_discrete_map={
        "Shallow & Near (High Impact)": "#3B5E64",
        "Intermediate Depth (Moderate)": "#b45356",
        "Deep & Remote (Low Impact)": "#505797"
          },
    title="Magnitude vs Depth by Profile (Sample of 5,000 Events)",
    opacity=0.6
)

fig_scatter.update_layout(
    xaxis_title="Magnitude",
    yaxis_title="Depth (km)",
    yaxis=dict(autorange="reversed"),
    legend_title_text="Profile"
)

st.plotly_chart(style_fig(fig_scatter), use_container_width=True)

st.markdown(
    """
    <div class="insight-box">
    <strong>💡 Key Insight:</strong>
    <br><br>
    The <b>Deep & Remote</b> profile has the strongest average magnitude
    yet the lowest felt rate (1.2%), while the <b>Shallow & Near</b> profile
    has the highest felt rate (13.1%).
    <br><br>
    This confirms that magnitude alone does not determine human perception —
    depth and location are the driving factors, which justifies the engineered
    features (<b>mag_depth_ratio</b>, <b>is_remote</b>) used in the supervised model.
    </div>
    """,
    unsafe_allow_html=True
)