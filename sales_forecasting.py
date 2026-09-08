"""
Time Series Sales Forecasting — End-to-End Pipeline
=====================================================
Dataset : Superstore Sales (order-level transactions, 2009-2012)
Source  : https://raw.githubusercontent.com/curran/data/gh-pages/superstoreSales/superstoreSales.csv

Pipeline stages:
  1. Load data, parse dates, aggregate to weekly sales (reduces noise vs. daily)
  2. EDA: trend, seasonality (weekly-of-year), holiday-adjacent spikes
  3. Feature engineering: calendar features, lag features, rolling-window stats
  4. Chronological train/test split (no shuffling — this is a time series!)
  5. Build & compare a naive baseline, Random Forest, and Gradient Boosting regressor
  6. Evaluate with MAE, RMSE, MAPE
  7. Forecast the next 12 weeks and plot actual vs. predicted
  8. Persist metrics.json + plots
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "superstore_sales.csv"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
N_LAGS = 8          # weeks of lag features
TEST_WEEKS = 12      # holdout horizon


def load_weekly_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    daily = df.groupby("Order Date")["Sales"].sum().sort_index()

    # Fill missing calendar days with 0 sales (no orders that day) before resampling
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = "date"

    weekly = daily.resample("W-MON").sum()
    return weekly


def run_eda(weekly: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].plot(weekly.index, weekly.values, color="#4C72B0")
    axes[0].plot(weekly.index, weekly.rolling(8).mean(), color="#C44E52", label="8-week rolling mean")
    axes[0].set_title("Weekly Sales — Trend & Smoothed Trend")
    axes[0].legend()

    monthly_avg = weekly.groupby(weekly.index.month).mean()
    axes[1].bar(monthly_avg.index, monthly_avg.values, color="#55A868")
    axes[1].set_title("Average Weekly Sales by Calendar Month (Seasonality)")
    axes[1].set_xlabel("Month")
    axes[1].set_xticks(range(1, 13))

    plt.tight_layout()
    plt.savefig(OUT / "eda_trend_seasonality.png", dpi=150)
    plt.close()

    return {
        "n_weeks": int(weekly.shape[0]),
        "date_range": [str(weekly.index.min().date()), str(weekly.index.max().date())],
        "mean_weekly_sales": round(float(weekly.mean()), 2),
        "peak_month": int(monthly_avg.idxmax()),
        "lowest_month": int(monthly_avg.idxmin()),
    }


def engineer_features(weekly: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"sales": weekly})

    # Calendar features
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year

    # Lag features
    for lag in range(1, N_LAGS + 1):
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # Rolling-window statistics (computed on lagged series to avoid leakage)
    df["roll_mean_4"] = df["sales"].shift(1).rolling(4).mean()
    df["roll_std_4"] = df["sales"].shift(1).rolling(4).std()
    df["roll_mean_8"] = df["sales"].shift(1).rolling(8).mean()

    df.dropna(inplace=True)
    return df


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def main():
    print("Loading data & building weekly series...")
    weekly = load_weekly_series()

    print("Running EDA...")
    eda_summary = run_eda(weekly)
    print(f"  {eda_summary}")

    print("Engineering lag/rolling/calendar features...")
    feat_df = engineer_features(weekly)

    X = feat_df.drop(columns=["sales"])
    y = feat_df["sales"]

    # Chronological split — NEVER shuffle time series data
    X_train, X_test = X.iloc[:-TEST_WEEKS], X.iloc[-TEST_WEEKS:]
    y_train, y_test = y.iloc[:-TEST_WEEKS], y.iloc[-TEST_WEEKS:]
    print(f"  Train weeks: {len(X_train)}, Test weeks: {len(X_test)}")

    # --- Naive baseline: predict last observed value (seasonal-naive at lag 1) ---
    naive_preds = X_test["lag_1"].values

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # --- Gradient Boosting ---
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.85, random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)

    models = {
        "naive_lag1_baseline": naive_preds,
        "random_forest": rf_preds,
        "gradient_boosting": gb_preds,
    }

    comparison = {}
    for name, preds in models.items():
        comparison[name] = {
            "MAE": round(mean_absolute_error(y_test, preds), 2),
            "RMSE": round(mean_squared_error(y_test, preds) ** 0.5, 2),
            "MAPE_pct": round(mape(y_test, preds), 2),
        }
        print(f"  {name:22s} | MAE={comparison[name]['MAE']:.2f} "
              f"RMSE={comparison[name]['RMSE']:.2f} MAPE={comparison[name]['MAPE_pct']:.2f}%")

    best_name = min(comparison, key=lambda n: comparison[n]["RMSE"])
    best_preds = models[best_name]
    print(f"Best model: {best_name}")

    # Plot actual vs predicted for the test horizon
    plt.figure(figsize=(12, 5))
    plt.plot(y_train.index[-20:], y_train.values[-20:], label="Train (recent)", color="#999999")
    plt.plot(y_test.index, y_test.values, label="Actual", color="#4C72B0", marker="o")
    plt.plot(y_test.index, best_preds, label=f"Predicted ({best_name})", color="#C44E52", marker="x")
    plt.title(f"Actual vs. Predicted Weekly Sales — Last {TEST_WEEKS} Weeks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Feature importance for the winning tree model
    winner_model = {"random_forest": rf, "gradient_boosting": gb}.get(best_name)
    top_features = {}
    if winner_model is not None:
        importances = pd.Series(winner_model.feature_importances_, index=X.columns)
        top = importances.sort_values(ascending=False).head(10)
        plt.figure(figsize=(7, 5))
        top.sort_values().plot.barh(color="#4C72B0")
        plt.title("Top 10 Forecast Drivers (Feature Importance)")
        plt.tight_layout()
        plt.savefig(OUT / "feature_importance.png", dpi=150)
        plt.close()
        top_features = top.round(4).to_dict()

    report = {
        "dataset": "Superstore Sales (order-level, aggregated to weekly)",
        "eda_summary": eda_summary,
        "test_horizon_weeks": TEST_WEEKS,
        "model_comparison": comparison,
        "best_model": best_name,
        "top_forecast_drivers": top_features,
        "observations": (
            f"Weekly sales peak around month {eda_summary['peak_month']} and dip around month "
            f"{eda_summary['lowest_month']}, consistent with seasonal retail demand. "
            f"The {best_name.replace('_', ' ')} model reduced RMSE by "
            f"{round(100 * (1 - comparison[best_name]['RMSE'] / comparison['naive_lag1_baseline']['RMSE']), 1)}% "
            "versus the naive last-value baseline."
        ),
    }

    with open(OUT / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone. Metrics -> {OUT/'metrics.json'}  |  Plots -> {OUT}")


if __name__ == "__main__":
    main()
