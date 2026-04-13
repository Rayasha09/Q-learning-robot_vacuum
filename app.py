from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train import FEATURE_COLUMNS, build_card_level_dataset, train

BASE_DIR = Path(__file__).resolve().parent
CC_INFO_PATH = BASE_DIR / "cc_info.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
CLUSTERED_DATA_PATH = BASE_DIR / "clustered_cards.csv"
FIGURE_PATH = BASE_DIR / "cluster_scatter.png"

st.set_page_config(page_title="Unsupervised Credit Card Clustering", layout="wide")


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists() or not CLUSTERED_DATA_PATH.exists() or not FIGURE_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_cc = pd.read_csv(CC_INFO_PATH)
    processed = pd.read_csv(CLUSTERED_DATA_PATH) if CLUSTERED_DATA_PATH.exists() else build_card_level_dataset()
    return raw_cc, processed


bundle = load_model_bundle()
raw_dataset, clustered_dataset = load_datasets()

st.title("Unsupervised Learning Assignment")

st.header("Dataset Name")
st.write("Credit Card Transactions Synthetic Data Generation")
st.caption("Files used in this project: `cc_info.csv` and `transactions.csv`.")

st.header("Explanation of Unsupervised Learning")
st.write(
    "Unsupervised learning is a type of machine learning where the data does not have target "
    "labels. The algorithm tries to find hidden patterns or groups inside the data. In this "
    "project, KMeans is used to group similar credit cards based on their spending behavior."
)

st.header("Project Description")
st.write(
    "This project combines credit card information with transaction history, creates numeric "
    "features for each card, scales the data, and applies KMeans clustering. PCA is then used "
    "to reduce the features to two dimensions so the clusters can be visualized more clearly."
)

st.header("Figure")
st.image(str(FIGURE_PATH), caption="Cluster visualization after PCA", use_container_width=True)

st.header("10 Samples from the Dataset")
st.dataframe(raw_dataset.head(10), use_container_width=True)

st.header("Model Information")
col1, col2 = st.columns(2)
col1.metric("Number of clusters", bundle["best_k"])
col2.metric("Silhouette score", f'{bundle["silhouette_score"]:.4f}')

st.header("Predict a Cluster")
defaults = clustered_dataset[FEATURE_COLUMNS].median(numeric_only=True).to_dict()
input_values: dict[str, float] = {}
cols = st.columns(3)
for index, feature in enumerate(FEATURE_COLUMNS):
    with cols[index % 3]:
        input_values[feature] = st.number_input(feature, value=float(defaults[feature]), format="%.4f")

if st.button("Predict Cluster"):
    frame = pd.DataFrame([input_values], columns=bundle["feature_columns"])
    transformed = bundle["preprocessor"].transform(frame)
    cluster = int(bundle["model"].predict(transformed)[0])
    st.success(f"Predicted cluster: {cluster}")
