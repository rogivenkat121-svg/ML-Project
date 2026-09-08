# Slab 2 [For Intermediate] — Applied ML Projects

Three end-to-end, intermediate-level applied machine learning projects: data
cleaning → EDA → feature engineering → model comparison → tuning → evaluation
→ practical recommendations. Each task is self-contained in its own folder
with its own data, script, README, and generated outputs.

| # | Task | Type | Dataset |
|---|---|---|---|
| 1 | [Customer Churn Prediction](task1_churn_prediction/) | Binary classification | IBM Telco Customer Churn (7,043 customers) |
| 2 | [Time Series Sales Forecasting](task2_sales_forecasting/) | Regression / forecasting | Superstore Sales, 2009–2012 (weekly-aggregated) |
| 3 | [Fraud Detection & Risk Analysis](task3_fraud_detection/) | Imbalanced classification | Synthetic transactions (see note below) |

## Quick start
```bash
git clone <this-repo-url>
cd <repo-name>
pip install -r requirements.txt

cd task1_churn_prediction && python churn_prediction.py && cd ..
cd task2_sales_forecasting  && python sales_forecasting.py  && cd ..
cd task3_fraud_detection    && python fraud_detection.py    && cd ..
```
Each script prints a run log to the console and writes plots + `metrics.json`
to that task's `outputs/` folder.

## Results at a glance

**Task 1 — Churn Prediction:** tuned Gradient Boosting reaches **ROC-AUC 0.83**
on held-out customers; contract length, payment method, and tenure are the
strongest churn drivers. Full writeup in [task1's README](task1_churn_prediction/README.md).

**Task 2 — Sales Forecasting:** tree-based models cut forecast RMSE by
**~33%** versus a naive last-value baseline over a 12-week holdout, with clear
December-peak / June-trough seasonality. Full writeup in
[task2's README](task2_sales_forecasting/README.md).

**Task 3 — Fraud Detection:** tuned Random Forest achieves **97% precision /
96% recall** (PR-AUC 0.99) on a held-out test set, with transaction amount,
distance-from-home, and transaction velocity as the strongest fraud signals.
Full writeup in [task3's README](task3_fraud_detection/README.md).

## A note on datasets
This environment's network access is restricted to a small allowlist of
domains (GitHub, PyPI, npm, etc.) — Kaggle, the usual home for these three
classic datasets, isn't reachable. Tasks 1 and 2 use real public datasets
mirrored on GitHub; Task 3 uses a documented synthetic dataset built to mirror
well-known real-world fraud patterns (see
[`task3_fraud_detection/data/generate_data.py`](task3_fraud_detection/data/generate_data.py)
for exactly how and why). Swapping in the real Kaggle `creditcard.csv` is a
one-line change if you have it locally.

## Repo structure
```
.
├── README.md                       (this file)
├── requirements.txt
├── task1_churn_prediction/
│   ├── data/Telco-Customer-Churn.csv
│   ├── churn_prediction.py
│   ├── outputs/                    (plots + metrics.json, generated on run)
│   └── README.md
├── task2_sales_forecasting/
│   ├── data/superstore_sales.csv
│   ├── sales_forecasting.py
│   ├── outputs/
│   └── README.md
└── task3_fraud_detection/
    ├── data/generate_data.py, fraud_transactions.csv
    ├── fraud_detection.py
    ├── outputs/
    └── README.md
```

## Tech stack
Python, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn.
