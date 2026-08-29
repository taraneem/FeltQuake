import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from utils import load_data, load_model

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0E1B20;
}
.metric-card {
    background-color: #172A31;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #29434B;
    text-align: center;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: all 0.3s ease-in-out;
}
.metric-card:hover {
    transform: translateY(-8px);
    border-color: #E27D45;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
}
.metric-title {
    color: #9FB1B7;
    font-size: 14px;
    margin-bottom: 8px;
}
.metric-value {
    color: #F4F7F8;
    font-size: 30px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

df = load_data()
model = load_model()

st.title("📊 Model Performance")

st.markdown(
    "Evaluate the final Machine Learning model and understand "
    "how it performs on unseen test data."
)

X = df.drop(columns=["reported"])
y = df["reported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_font_color="#F4F7F8",
        legend_bgcolor="rgba(0,0,0,0)",
        xaxis_gridcolor="#29434B",
        yaxis_gridcolor="#29434B"
    )
    return fig


st.header("🏆 Final Model Performance")

cols = st.columns(4)

metrics = [
    ("ACCURACY", accuracy),
    ("PRECISION", precision),
    ("RECALL", recall),
    ("F1-SCORE", f1)
]

for col, (name, value) in zip(cols, metrics):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">{name}</div>'
            f'<div class="metric-value">{value * 100:.2f}%</div></div>',
            unsafe_allow_html=True
        )

st.markdown("---")

st.header("🤖 Model Selection")

st.markdown(
    """
    Several classifiers were compared using cross-validation on the
    training data after addressing the class imbalance problem.
    **XGBoost achieved the highest F1-Score (70.88%)** together with a
    strong Recall (81.11%), which is why it was selected for
    hyperparameter tuning.
    """
)

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression", "Logistic Regression", "Logistic Regression",
        "SVC", "SVC", "SVC",
        "Decision Tree", "Decision Tree", "Decision Tree",
        "Random Forest", "Random Forest", "Random Forest",
        "XGBoost ⭐ (Selected)", "XGBoost ⭐ (Selected)", "XGBoost ⭐ (Selected)"
    ],
    "Metric": [
        "Precision", "Recall", "F1-Score",
        "Precision", "Recall", "F1-Score",
        "Precision", "Recall", "F1-Score",
        "Precision", "Recall", "F1-Score",
        "Precision", "Recall", "F1-Score"
    ],
    "Score": [
        38.76, 79.63, 52.14,
        42.14, 84.17, 56.15,
        57.95, 54.62, 56.20,
        69.62, 69.65, 69.62,
        62.95, 81.11, 70.88
    ]
})

fig_models = px.bar(
    model_results,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    color_discrete_map={
        "Precision": "#2c456b",
        "Recall": "#4c649f",
        "F1-Score": "#4779c4"
    },
    title="Model Comparison (Cross-Validation on Training Data)"
)

fig_models.update_layout(
    xaxis_title="Model",
    yaxis_title="Score (%)",
    yaxis=dict(range=[0, 100])
)

st.plotly_chart(style_fig(fig_models), use_container_width=True)

st.caption(
    "XGBoost was selected because it combines the highest F1-Score with "
    "a high Recall, which is critical for minimizing missed felt earthquakes."
)

st.header("⚖️ Class Imbalance")

class_distribution = (
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
    .reset_index()
)

class_distribution.columns = ["Class", "Percentage"]

class_distribution["Label"] = (
    class_distribution["Class"].map({0: "Not Felt", 1: "Felt"})
)

fig_class = px.bar(
    class_distribution,
    x="Label",
    y="Percentage",
    text="Percentage",
    color="Label",
    color_discrete_map={
        "Not Felt": "#475569",
        "Felt": "#2c456b"
    },
    title="Target Class Distribution"
)

fig_class.update_layout(
    yaxis_title="Percentage (%)",
    showlegend=False
)

st.plotly_chart(style_fig(fig_class), use_container_width=True)

st.info(
    """
    The target is imbalanced, with approximately 88% of events
    belonging to class 0 and 12% to class 1.

    Therefore, Accuracy alone is not sufficient. Recall and F1-Score
    are especially important when evaluating the minority class.
    """
)

st.header("🎯 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Not Felt", "Actual Felt"],
    columns=["Predicted Not Felt", "Predicted Felt"]
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale=["#172A31", "#5eead4"],
    title="Confusion Matrix (Test Set)"
)

fig_cm.update_layout(
    coloraxis_showscale=False
)

st.plotly_chart(style_fig(fig_cm), use_container_width=True)

st.header("📈 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()

fig_roc.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"XGBoost (AUC = {roc_auc:.3f})",
        line=dict(color="#5eead4", width=3)
    )
)

fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(dash="dash", color="#475569")
    )
)

fig_roc.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate"
)

st.plotly_chart(style_fig(fig_roc), use_container_width=True)

st.header("🎯 Precision–Recall Curve")

precision_values, recall_values, _ = precision_recall_curve(y_test, y_proba)

fig_pr = go.Figure()

fig_pr.add_trace(
    go.Scatter(
        x=recall_values,
        y=precision_values,
        mode="lines",
        name="XGBoost",
        line=dict(color="#E27D45", width=3)
    )
)

fig_pr.update_layout(
    title="Precision–Recall Curve",
    xaxis_title="Recall",
    yaxis_title="Precision",
    showlegend=False
)

st.plotly_chart(style_fig(fig_pr), use_container_width=True)

st.header("🎚️ Threshold Analysis")

st.markdown(
    """
    The default classification threshold is **0.50**.

    Lowering the threshold can increase Recall, which may be useful
    when missing a truly felt earthquake is considered more costly
    than generating a false alarm.
    """
)

thresholds = np.arange(0.20, 0.81, 0.05)

threshold_results = []

for threshold in thresholds:

    threshold_pred = (y_proba >= threshold).astype(int)

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision_score(y_test, threshold_pred, zero_division=0),
        "Recall": recall_score(y_test, threshold_pred, zero_division=0),
        "F1": f1_score(y_test, threshold_pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

fig_threshold = go.Figure()

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold"],
        y=threshold_df["Precision"],
        mode="lines+markers",
        name="Precision",
        line=dict(color="#475569", width=2)
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold"],
        y=threshold_df["Recall"],
        mode="lines+markers",
        name="Recall",
        line=dict(color="#5eead4", width=2)
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=threshold_df["Threshold"],
        y=threshold_df["F1"],
        mode="lines+markers",
        name="F1",
        line=dict(color="#E27D45", width=2)
    )
)

fig_threshold.update_layout(
    title="Precision, Recall and F1 vs Classification Threshold",
    xaxis_title="Threshold",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(style_fig(fig_threshold), use_container_width=True)

st.markdown("---")

st.header("🏆 Final Model: Tuned XGBoost")

st.success(
    """
    The final model is a tuned XGBoost classifier selected after
    model comparison, class-imbalance handling and hyperparameter tuning.
    """
)

summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [
        f"{accuracy * 100:.2f}%",
        f"{precision * 100:.2f}%",
        f"{recall * 100:.2f}%",
        f"{f1 * 100:.2f}%"
    ]
})

st.table(summary_df)