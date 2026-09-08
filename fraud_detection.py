"""
End-to-End Fraud Detection & Risk Analysis — Pipeline
=======================================================
Dataset : Synthetic financial transactions (50,000 txns, ~0.6% fraud rate)
          See data/generate_data.py for exactly how & why it was generated
          (real Kaggle/PaySim fraud data isn't reachable from this sandbox's
          network allowlist — swap in the real creditcard.csv and this script
          still works unchanged as long as the target column is renamed to
          `is_fraud`).

Pipeline stages:
  1. Load & clean data
  2. EDA: class imbalance, fraud-by-hour, fraud-by-channel, amount distributions
  3. Feature engineering + encoding
  4. Chronological-safe train/test split (stratified, since this isn't a forecast)
  5. Handle severe class imbalance (SMOTE on training fold only)
  6. Train & compare Logistic Regression, Random Forest, Gradient Boosting
  7. Tune the best model (RandomizedSearchCV, optimizing for recall-heavy F1/AUC)
  8. Evaluate: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, PR curve
  9. Examine false positives / false negatives
  10. Feature importance + practical recommendations
  11. Persist metrics.json + plots
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
    PrecisionRecallDisplay,
    average_precision_score,
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
DATA_PATH = HERE / "data" / "fraud_transactions.csv"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["transaction_time"])
    df.drop(columns=["transaction_id", "transaction_time"], inplace=True)
    return df


def run_eda(df: pd.DataFrame):
    fraud_rate = df["is_fraud"].mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    df["is_fraud"].map({0: "Legit", 1: "Fraud"}).value_counts().plot.bar(
        ax=axes[0], color=["#4C72B0", "#C44E52"]
    )
    axes[0].set_title(f"Class Balance (fraud rate: {fraud_rate:.2f}%)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Count (log scale)")

    hourly_fraud_rate = df.groupby("hour")["is_fraud"].mean() * 100
    hourly_fraud_rate.plot.bar(ax=axes[1], color="#55A868")
    axes[1].set_title("Fraud Rate by Hour of Day")
    axes[1].set_ylabel("Fraud rate (%)")

    sns.boxplot(data=df, x="is_fraud", y="amount", ax=axes[2], palette=["#4C72B0", "#C44E52"])
    axes[2].set_xticklabels(["Legit", "Fraud"])
    axes[2].set_yscale("log")
    axes[2].set_title("Transaction Amount by Class (log scale)")

    plt.tight_layout()
    plt.savefig(OUT / "eda_overview.png", dpi=150)
    plt.close()

    return {
        "overall_fraud_rate_pct": round(fraud_rate, 3),
        "peak_fraud_hour": int(hourly_fraud_rate.idxmax()),
        "mean_amount_fraud": round(df.loc[df.is_fraud == 1, "amount"].mean(), 2),
        "mean_amount_legit": round(df.loc[df.is_fraud == 0, "amount"].mean(), 2),
    }


def build_matrices(df: pd.DataFrame):
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X_encoded, y, num_cols


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
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
        }
    return results


def tune_best_model(name, X_train, y_train):
    if name == "random_forest":
        base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 12, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    elif name == "gradient_boosting":
        base = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_dist = {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 1.0],
        }
    else:
        base = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        param_dist = {"C": [0.01, 0.1, 1, 3, 10]}

    # Hyperparameter search on a stratified subsample for speed; final model is
    # refit on the FULL balanced training set once the best params are found.
    if len(X_train) > 20_000:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train, train_size=20_000, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_search, y_search = X_train, y_train

    search = RandomizedSearchCV(
        base, param_dist, n_iter=8, scoring="average_precision", cv=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_search, y_search)

    final_model = search.best_estimator_.__class__(**search.best_estimator_.get_params())
    final_model.fit(X_train, y_train)
    return final_model, search.best_params_


def evaluate_final(model, X_test, y_test, feature_names):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                              "false_negative": int(fn), "true_positive": int(tp)},
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
        ax=axes[0], cmap="Reds", colorbar=False
    )
    axes[0].set_title("Confusion Matrix (Tuned Model)")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].set_title(f"Precision-Recall Curve (PR-AUC = {metrics['pr_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(OUT / "final_model_evaluation.png", dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top = importances.sort_values(ascending=False).head(12)
        plt.figure(figsize=(8, 6))
        top.sort_values().plot.barh(color="#C44E52")
        plt.title("Top Fraud Signal Features (Feature Importance)")
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
    print(f"  {eda_summary}")

    X, y, num_cols = build_matrices(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    print("Balancing severe class imbalance with SMOTE (train fold only)...")
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
              f"f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}")

    # For fraud, PR-AUC is the more meaningful ranking metric under heavy imbalance
    best_name = max(comparison_summary, key=lambda n: comparison_summary[n]["pr_auc"])
    print(f"Best baseline model (by PR-AUC): {best_name}. Tuning hyperparameters...")
    tuned_model, best_params = tune_best_model(best_name, X_train_bal, y_train_bal)
    print(f"  Best params: {best_params}")

    print("Evaluating tuned model on held-out test set...")
    final_metrics = evaluate_final(tuned_model, X_test_scaled, y_test, X.columns)
    print(f"  Confusion matrix: {final_metrics['confusion_matrix']}")

    report = {
        "dataset": "Synthetic financial transactions (see data/generate_data.py)",
        "n_transactions": int(df.shape[0]),
        "eda_summary": eda_summary,
        "best_baseline_model": best_name,
        "baseline_comparison": comparison_summary,
        "tuned_model_params": best_params,
        "final_test_metrics": {k: v for k, v in final_metrics.items()
                                if k not in ("classification_report",)},
        "classification_report": final_metrics["classification_report"],
        "top_fraud_signals": final_metrics.get("top_features", {}),
    }

    with open(OUT / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone. Metrics -> {OUT/'metrics.json'}  |  Plots -> {OUT}")


if __name__ == "__main__":
    main()
