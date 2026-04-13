# unsupervised_creditcard_fraud_detection

This repository is prepared for the assignment **Unsupervised Learning**.

## Dataset

- Kaggle dataset name: **Credit Card Transactions Synthetic Data Generation**
- Files used:
  - `cc_info.csv`
  - `transactions.csv`

## Algorithm

- `KMeans` clustering
- `StandardScaler` for feature scaling
- `PCA` for 2D visualization

## Files

- `train.py` - trains the model and saves artifacts
- `test.py` - tests the saved model from the terminal
- `app.py` - Streamlit web page
- `model.joblib` - saved trained model
- `clustered_cards.csv` - processed dataset with cluster labels
- `metrics.json` - silhouette score and selected `k`
- `cluster_scatter.png` - figure used in the web page

## Run

```bash
cd unsupervised_creditcard_fraud_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 train.py
python3 test.py
streamlit run app.py
```

## Note

The folder name already follows the assignment format:

`unsupervised_[name of your dataset]`
