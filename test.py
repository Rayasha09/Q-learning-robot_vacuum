from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.joblib not found. Run train.py first.")
    return joblib.load(MODEL_PATH)


def get_input(feature_columns: list[str]) -> dict[str, float]:
    print("Enter values to predict a cluster:")
    values: dict[str, float] = {}
    for feature in feature_columns:
        values[feature] = float(input(f"{feature}: ").strip())
    return values


def main() -> None:
    bundle = load_model()
    values = get_input(bundle["feature_columns"])
    frame = pd.DataFrame([values], columns=bundle["feature_columns"])
    transformed = bundle["preprocessor"].transform(frame)
    cluster = int(bundle["model"].predict(transformed)[0])
    print(f"Predicted cluster: {cluster}")


if __name__ == "__main__":
    main()
