from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
CC_INFO_PATH = BASE_DIR / "cc_info.csv"
TRANSACTIONS_PATH = BASE_DIR / "transactions.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
CLUSTERED_DATA_PATH = BASE_DIR / "clustered_cards.csv"
METRICS_PATH = BASE_DIR / "metrics.json"
FIGURE_PATH = BASE_DIR / "cluster_scatter.png"

FEATURE_COLUMNS = [
    "credit_card_limit",
    "total_transactions",
    "total_amount",
    "avg_amount",
    "std_amount",
    "max_amount",
    "active_days",
    "avg_daily_transactions",
    "weekend_ratio",
    "mean_hour",
    "mean_long",
    "mean_lat",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cc_info = pd.read_csv(CC_INFO_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
    return cc_info, transactions


def build_card_level_dataset() -> pd.DataFrame:
    cc_info, transactions = load_data()
    tx = transactions.copy()
    tx["tx_date"] = tx["date"].dt.date
    tx["is_weekend"] = tx["date"].dt.dayofweek >= 5
    tx["hour"] = tx["date"].dt.hour

    aggregated = tx.groupby("credit_card").agg(
        total_transactions=("transaction_dollar_amount", "size"),
        total_amount=("transaction_dollar_amount", "sum"),
        avg_amount=("transaction_dollar_amount", "mean"),
        std_amount=("transaction_dollar_amount", "std"),
        max_amount=("transaction_dollar_amount", "max"),
        active_days=("tx_date", "nunique"),
        weekend_ratio=("is_weekend", "mean"),
        mean_hour=("hour", "mean"),
        mean_long=("Long", "mean"),
        mean_lat=("Lat", "mean"),
    )
    aggregated["avg_daily_transactions"] = (
        aggregated["total_transactions"] / aggregated["active_days"].clip(lower=1)
    )
    aggregated["std_amount"] = aggregated["std_amount"].fillna(0.0)

    return cc_info.merge(aggregated.reset_index(), on="credit_card", how="inner")


def build_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def choose_best_k(features: pd.DataFrame) -> tuple[int, float, list[dict[str, float]]]:
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    best_k = 2
    best_score = -1.0
    all_scores: list[dict[str, float]] = []

    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(transformed)
        score = silhouette_score(transformed, labels)
        all_scores.append({"k": int(k), "silhouette_score": float(score)})
        if score > best_score:
            best_k = k
            best_score = score

    return best_k, best_score, all_scores


def save_figure(dataset: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=dataset,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        palette="tab10",
        s=80,
    )
    plt.title("KMeans Clusters in PCA Space")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()


def train() -> None:
    dataset = build_card_level_dataset()
    features = dataset[FEATURE_COLUMNS].copy()

    best_k, best_score, all_scores = choose_best_k(features)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    dataset["cluster"] = model.fit_predict(transformed)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(transformed)
    dataset["pca_1"] = components[:, 0]
    dataset["pca_2"] = components[:, 1]

    model_bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "pca": pca,
        "feature_columns": FEATURE_COLUMNS,
        "best_k": best_k,
        "silhouette_score": best_score,
    }

    joblib.dump(model_bundle, MODEL_PATH)
    dataset.to_csv(CLUSTERED_DATA_PATH, index=False)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "dataset_name": "Credit Card Transactions Synthetic Data Generation",
                "best_k": best_k,
                "silhouette_score": best_score,
                "scores": all_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_figure(dataset)

    print(f"Dataset: Credit Card Transactions Synthetic Data Generation")
    print(f"Model saved to: {MODEL_PATH.name}")
    print(f"Clustered dataset saved to: {CLUSTERED_DATA_PATH.name}")
    print(f"Best number of clusters: {best_k}")
    print(f"Silhouette score: {best_score:.4f}")


if __name__ == "__main__":
    train()
