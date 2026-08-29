import requests
import pandas as pd

# Loading the latest earthquake data via USGS API
url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"

response = requests.get(url)
data = response.json()

latest = data["features"][0]
props = latest["properties"]

mag = props["mag"]
place = props["place"]
lon, lat, depth = latest["geometry"]["coordinates"]
felt = props["felt"]
mag_depth_ratio = mag / (depth + 1)
is_remote = int(not ("km" in place.lower()))
time = pd.to_datetime(props["time"], unit="ms", utc=True)

month = time.month
hour = time.hour
day_of_week = time.dayofweek 

status = props["status"]
tsunami = props["tsunami"]
sig = props["sig"]
net = props["net"]
nst = props["nst"]
dmin = props["dmin"]
rms = props["rms"]
gap = props["gap"]
magType = props["magType"]
quake_type = props["type"]


print("Magnitude:", mag)
print("Place:", place)
print("Depth:", depth)
print("Latitude:", lat)
print("Longitude:", lon)
print("Felt reports:", felt)
print("Mag/Depth ratio:", mag_depth_ratio)
print("Is remote:", is_remote)
print("Month:", month)
print("Hour:", hour)
print("Day of week:", day_of_week)
print("Status:", status)
print("Tsunami:", tsunami)
print("Sig:", sig)
print("Net:", net)
print("Nst:", nst)
print("Dmin:", dmin)
print("Rms:", rms)
print("Gap:", gap)
print("MagType:", magType)
print("Type:", quake_type)

#=========================================

# Preparing the input data for the model
input_data = {
    "mag": mag,
    "status": status,
    "tsunami": tsunami,
    "sig": sig,
    "net": net,
    "nst": nst,
    "dmin": dmin,
    "rms": rms,
    "gap": gap,
    "magType": magType,
    "type": quake_type,
    "longitude": lon,
    "latitude": lat,
    "depth": depth,
    "hour": hour,
    "month": month,
    "day_of_week": day_of_week,
    "mag_depth_ratio": mag_depth_ratio,
    "is_remote": is_remote,
}

# =========================================

# Converting the input data to a DataFrame
X_live = pd.DataFrame([input_data])
print(X_live)


# =========================================

# Loading the trained model (Supervised - XGBoost)

import joblib

model = joblib.load("earthquake_model_final.pkl")

prediction = model.predict(X_live)
probability = model.predict_proba(X_live)[0][1]

print("Prediction (0 = not felt, 1 = felt):", prediction[0])
print("Probability of being felt:", probability)


# =========================================

# Loading the trained model (Unsupervised - KMeans)

cluster_bundle = joblib.load("Unsupervised_model.pkl")
scaler = cluster_bundle["scaler"]
kmeans = cluster_bundle["model"]

cluster_input = X_live[["mag", "longitude", "latitude", "depth", "mag_depth_ratio", "is_remote"]]
scaled = cluster_input if scaler is None else scaler.transform(cluster_input)
cluster_id = kmeans.predict(scaled)[0]

cluster_names = {
    0: "Shallow & Near (High Impact)",
    1: "Intermediate Depth (Moderate)",
    2: "Deep & Remote (Low Impact)",
}

print("Cluster ID:", cluster_id)
print("Cluster Profile:", cluster_names.get(cluster_id, f"Cluster {cluster_id}"))