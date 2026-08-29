import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "earthquakes_cleaned.csv"
MODEL_PATH = BASE_DIR / "earthquake_model_final.pkl"


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


# ---------------------------------------------------------
# Rare Categories
# Same logic used in the notebook
# ---------------------------------------------------------

def prepare_prediction_input(data):

    data = data.copy()

    # Same rare category grouping used during training

    rare_types = [
        "mining explosion",
        "explosion",
        "other event",
        "experimental explosion",
        "quarry blast"
    ]

    if "type" in data.columns:
        data["type"] = data["type"].replace(
            rare_types,
            "other"
        )

    if "net" in data.columns:

        # During training categories with frequency < 20
        # were grouped into "other".
        #
        # For manually entered prediction values,
        # unknown values are safely handled by the trained
        # OneHotEncoder(handle_unknown='ignore').

        pass

    if "magType" in data.columns:

        pass

    # Ensure categorical columns are strings
    categorical_columns = [
        "status",
        "net",
        "magType",
        "type"
    ]

    for col in categorical_columns:

        if col in data.columns:
            data[col] = data[col].astype("string")

    # Boolean feature was converted to 0/1 in training
    if "is_remote" in data.columns:

        data["is_remote"] = data["is_remote"].astype(int)

    return data


# ---------------------------------------------------------
# Label
# ---------------------------------------------------------

def prediction_label(prediction):

    if prediction == 1:
        return "Felt"

    return "Not Felt"


# ---------------------------------------------------------
# Cluster Model
# ---------------------------------------------------------

CLUSTER_MODEL_PATH = BASE_DIR / "Unsupervised_model.pkl"


@st.cache_resource
def load_cluster_model():

    bundle = joblib.load(CLUSTER_MODEL_PATH)

    return bundle["scaler"], bundle["model"]


CLUSTER_PROFILES = {
    0: "Shallow & Near (High Impact)",
    1: "Intermediate Depth (Moderate)",
    2: "Deep & Remote (Low Impact)",
}


def cluster_profile_label(cluster_id):

    return CLUSTER_PROFILES.get(cluster_id, f"Cluster {cluster_id}")