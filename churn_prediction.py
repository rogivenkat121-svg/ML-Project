"""
E-Commerce / Telecom Customer Churn Prediction — End-to-End Pipeline
=====================================================================
Dataset : IBM Telco Customer Churn (7,043 customers, 21 columns)
Source  : https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Pipeline stages:
  1. Load & clean data
  2. Exploratory data analysis (churn rate, key drivers) -> outputs/*.png
  3. Feature engineering (tenure buckets, encoding, scaling)
  4. Handle class imbalance (SMOTE on training fold only)
  5. Train & compare Logistic Regression, Random Forest, Gradient Boosting
  6. Hyperparameter-tune the best model (RandomizedSearchCV)
  7. Evaluate: precision, recall, F1, ROC-AUC, confusion matrix
  8. Feature importance + business recommendations
  9. Persist metrics.json + plots + trained model
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "Telco-Customer-Churn.csv"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges has blank strings for customers with 0 tenure -> coerce & impute
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets (a strong, interpretable churn signal)
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True, include_lowest=True)

    # Average monthly spend per tenure month (captures pricing pressure)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Count of subscribed "add-on" services (a proxy for engagement/stickiness)
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_addon_services"] = (df[addon_cols] == "Yes").sum(axis=1)

    return df


def build_matrices(df: pd.DataFrame):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X_encoded, y, num_cols


def run_eda(df: pd.DataFrame):
    churn_rate = df["Churn"].mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    df["Churn"].map({0: "No", 1: "Yes"}).value_counts().plot.bar(
        ax=axes[0], color=["#4C72B0", "#C44E52"]
    )
    axes[0].set_title(f"Churn Distribution (overall rate: {churn_rate:.1f}%)")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Customers")

    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values() * 100
    contract_churn.plot.bar(ax=axes[1], color="#55A868")
    axes[1].set_title("Churn Rate by Contract Type")
    axes[1].set_ylabel("Churn rate (%)")

    sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[2], palette=["#4C72B0", "#C44E52"])
    axes[2].set_xticklabels(["No", "Yes"])
    axes[2].set_title("Tenure by Churn Outcome")

    plt.tight_layout()
    plt.savefig(OUT / "eda_overview.png", dpi=150)
    plt.close()

    return {"overall_churn_rate_pct": round(churn_rate, 2)}


def train_and_compare(X_train, y_train, X_test, y_test):
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
    return results


def tune_best_model(name, X_train, y_train):
    if name == "random_forest":
        base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "n_estimators": [200, 300, 400, 600],
            "max_depth": [None, 6, 10, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    elif name == "gradient_boosting":
        base = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_dist = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.7, 0.85, 1.0],
        }
    else:
        base = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        param_dist = {"C": [0.01, 0.1, 1, 3, 10], "penalty": ["l2"]}

    search = RandomizedSearchCV(
        base, param_dist, n_iter=20, scoring="roc_auc", cv=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_final(model, X_test, y_test, feature_names):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels=["No Churn", "Churn"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("Confusion Matrix (Tuned Model)")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].set_title(f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(OUT / "final_model_evaluation.png", dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top = importances.sort_values(ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        top.sort_values().plot.barh(color="#4C72B0")
        plt.title("Top 15 Churn Drivers (Feature Importance)")
        plt.tight_layout()
        plt.savefig(OUT / "feature_importance.png", dpi=150)
        plt.close()
        metrics["top_features"] = top.round(4).to_dict()

    return metrics


def main():
    print("Loading & cleaning data...")
    df = load_and_clean(DATA_PATH)

    print("Running EDA...")
    eda_summary = run_eda(df)

    print("Engineering features...")
    df = engineer_features(df)
    X, y, num_cols = build_matrices(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    print("Balancing training data with SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"  After  SMOTE: {y_train_bal.value_counts().to_dict()}")

    print("Training & comparing baseline models...")
    comparison = train_and_compare(X_train_bal, y_train_bal, X_test_scaled, y_test)
    comparison_summary = {
        name: {k: round(v, 4) for k, v in r.items() if k != "model"}
        for name, r in comparison.items()
    }
    for name, m in comparison_summary.items():
        print(f"  {name:20s} | precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f}")

    best_name = max(comparison_summary, key=lambda n: comparison_summary[n]["roc_auc"])
    print(f"Best baseline model: {best_name}. Tuning hyperparameters...")
    tuned_model, best_params = tune_best_model(best_name, X_train_bal, y_train_bal)
    print(f"  Best params: {best_params}")

    print("Evaluating tuned model on held-out test set...")
    final_metrics = evaluate_final(tuned_model, X_test_scaled, y_test, X.columns)

    report = {
        "dataset": "IBM Telco Customer Churn",
        "n_customers": int(df.shape[0]),
        "eda_summary": eda_summary,
        "best_baseline_model": best_name,
        "baseline_comparison": comparison_summary,
        "tuned_model_params": best_params,
        "final_test_metrics": {k: v for k, v in final_metrics.items()
                                if k not in ("classification_report",)},
        "classification_report": final_metrics["classification_report"],
        "top_churn_drivers": final_metrics.get("top_features", {}),
    }

    with open(OUT / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone. Metrics -> {OUT/'metrics.json'}  |  Plots -> {OUT}")
    print(f"Final tuned model: precision={final_metrics['precision']}, "
          f"recall={final_metrics['recall']}, f1={final_metrics['f1_score']}, "
          f"roc_auc={final_metrics['roc_auc']}")


if __name__ == "__main__":
    main()
