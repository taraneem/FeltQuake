# 🌍 FeltQuake — Will People Feel This Earthquake?

FeltQuake is an end-to-end machine learning project built on real USGS earthquake data. It combines a **supervised classification model** (predicting whether an earthquake will be felt/reported by people) with an **unsupervised clustering analysis** (grouping earthquakes into distinct seismic profiles), deployed together as an interactive Streamlit application.

---

## 📌 Motivation

Earlier this year I experienced a strongly felt earthquake here in Egypt. What stuck with me wasn't just the shaking — it was noticing that earthquake warning alerts went out to Android phones but not to iPhones, and that people still panicked even with a warning in hand. That gap between *what the data says about an earthquake* and *how people actually experience and react to it* became the core question behind this project: **can we predict, from an earthquake's physical characteristics alone, whether it's likely to be felt and reported by people?**

---

## 🎯 Project Overview

| | |
|---|---|
| **Type** | Supervised (Classification) + Unsupervised (Clustering) |
| **Main task** | Predict whether an earthquake will be felt/reported (binary classification) |
| **Secondary task** | Cluster earthquakes into seismic profiles based on similarity |
| **Data source** | USGS Earthquake Catalog (historical sample via API/site download) |
| **Deployment** | Streamlit web app |

The project is split into two connected components:

1. **Supervised Learning — FeltQuake Classifier**
   A tuned XGBoost classifier trained to predict whether an earthquake event was felt/reported by people, based on features like magnitude, depth, and location.

2. **Unsupervised Learning — Seismic Profile Clustering**
   K-Means clustering applied to the same feature space to uncover natural groupings of earthquakes — independent of the felt/not-felt label — then cross-referenced against the actual felt rate to see how well "natural" clusters align with real-world impact.

---

## 🗂️ Dataset

- **Source:** USGS (United States Geological Survey) earthquake catalog
- **Access method:** Filtered historical sample downloaded via the USGS API/site (not only the live feed)
- **Size:** 38,000+ earthquake events
- **Target (supervised task):** Whether the earthquake was felt/reported (`reported`)
- **Key features used:**
  - `mag` — magnitude
  - `depth` — depth of the earthquake (km)
  - `latitude`, `longitude` — location
  - `mag_depth_ratio` — engineered feature combining magnitude and depth
  - `is_remote` — whether the event occurred in a remote area

---

## 🧠 Methodology

### 1. Data Cleaning & Feature Engineering
Raw USGS data was cleaned and enriched with engineered features (e.g. `mag_depth_ratio`, `is_remote`) to better capture the relationship between an earthquake's physical properties and its likelihood of being felt.

### 2. Supervised Model — FeltQuake
- Framed as a **binary classification** problem (felt vs. not felt/reported)
- Model: **tuned XGBoost classifier**
- Careful preprocessing pipeline (train/test split before any transformation, to avoid data leakage)
- Evaluated beyond accuracy alone, given class imbalance in the target

**Final test-set performance:**

| Metric | Score |
|---|---|
| Accuracy | 92.51% |
| Precision | 64.67% |
| Recall | 84.75% |
| F1-Score | 73.36% |

Recall was prioritized during model selection since missing a truly felt earthquake is more costly than a false alarm, given the ~88/12 class imbalance in the data.

### 3. Unsupervised Model — Seismic Profile Clustering
- Features scaled using `RobustScaler` inside a `ColumnTransformer` pipeline (robust to outliers, important for skewed seismic data)
- Optimal number of clusters selected using **both the Elbow Method and Silhouette Score**
- **K = 3** was chosen: although Silhouette Score peaked at K=2, K=3 was selected for interpretability — it separates events into three seismic profiles that align meaningfully with real felt-rate patterns, rather than a coarser two-way split
- Each cluster was interpreted using the average magnitude, depth, remoteness, and felt rate

**Resulting seismic profiles:**

| Cluster | Profile | Events | Felt Rate | Avg. Depth | Remote % |
|---|---|---|---|---|---|
| 0 | Shallow & Near (High Impact) | 30,824 | 13.1% | ~20 km | 10.0% |
| 1 | Intermediate Depth (Moderate) | 6,052 | 10.1% | ~132 km | 10.6% |
| 2 | Deep & Remote (Low Impact) | 1,394 | 1.2% | ~519 km | 52.4% |

This confirms an intuitive geophysical pattern: **shallow, nearby earthquakes are far more likely to be felt than deep, remote ones** — even before the supervised model sees a single label.

### 4. Deployment — Streamlit App
Both the classification model and the clustering insights are integrated into a multi-page Streamlit application: an explorable dataset view, an EDA walkthrough, a manual prediction form, model performance diagnostics, the clustering profiles, and a **live page that pulls the most recent earthquake directly from the USGS feed and runs it through both models in real time**.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** pandas
- **Modeling:** scikit-learn (KMeans, RobustScaler, ColumnTransformer, Pipeline), XGBoost
- **Evaluation:** Silhouette Score, Elbow Method
- **Visualization:** Plotly Express
- **Deployment:** Streamlit
- **Model persistence:** joblib

---

## 📁 Project Structure

```
FeltQuake-Epsilon-Final-Project/
│
├── app.py                                  # Home page
├── utils.py                                # Shared data/model loading + preprocessing helpers
├── pages/
│   ├── 1_🌍_Earthquake_Explorer.py
│   ├── 2_🔎_What_Makes_It_Felt.py
│   ├── 3_🤖_Felt_Prediction.py
│   ├── 4_📊_Model_Performance.py
│   ├── 5_💫Clusters.py
│   └── 6_🌪️Latest_Earthquake_Prediction.py
│
├── Data Preparation + EDA.ipynb
├── Preprocessing + Model Selection.ipynb
├── Unsupervised - Clustering.ipynb
│
├── earthquakes_cleaned.csv
├── earthquakes_clusters.csv
├── earthquake_model_final.pkl              # Full pipeline: preprocessing + tuned XGBoost
├── Unsupervised_model.pkl                  # {'scaler': ColumnTransformer, 'model': KMeans}
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/tarneemmedhat/FeltQuake-Epsilon-Final-Project.git
   cd FeltQuake-Epsilon-Final-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
