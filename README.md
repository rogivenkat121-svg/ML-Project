# Task 2 — Time Series Sales Forecasting

Forecasts weekly retail sales using 4 years of order-level transaction data,
so the business can anticipate demand and plan inventory/staffing.

## Dataset
- Source: [Superstore Sales](https://raw.githubusercontent.com/curran/data/gh-pages/superstoreSales/superstoreSales.csv) — 8,399 orders, 2009–2012
- Aggregated to **209 weekly** observations (`W-MON` resample; missing calendar
  days filled with 0 before aggregation)

## Pipeline (`sales_forecasting.py`)
1. **Clean** — parse order dates, fill 42 missing calendar days with 0 sales, aggregate to weekly totals.
2. **EDA** — trend + 8-week rolling mean, average weekly sales by calendar month (seasonality).
3. **Feature engineering** — calendar features (week-of-year, month, quarter, year),
   8 weeks of lag features, rolling mean/std (computed on *lagged* values only, to avoid leakage).
4. **Chronological train/test split** — last **12 weeks held out**, never shuffled.
5. **Models compared** — naive last-value baseline, Random Forest, Gradient Boosting.
6. **Evaluation** — MAE, RMSE, MAPE on the 12-week holdout.

## Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive (last-value) baseline | 36,465 | 51,929 | 48.6% |
| Random Forest | 26,006 | 34,997 | 39.8% |
| **Gradient Boosting (best, by RMSE)** | 27,336 | **34,672** | 40.3% |

Gradient Boosting cut RMSE by **~33%** versus the naive baseline.

### Seasonal patterns observed
- **Peak month: December** — consistent with holiday-season retail demand.
- **Lowest month: June** — a mid-year demand trough.
- Weekly sales are noisy at the order-count scale this dataset has (~40
  orders/week on average), which caps how low error can go — MAPE in the
  high-30s/low-40s is reasonable here, not a modeling defect. A dataset with
  higher transaction volume per period (e.g. daily sales across thousands of
  SKUs) would let lag/rolling features explain more variance.

### Recommendations
- **Plan inventory buildup starting October** ahead of the December peak, and scale back promotional spend heading into the June trough.
- **Use the 4- and 8-week rolling means as a lightweight, explainable planning signal** for teams that don't want a full ML pipeline — they capture most of the trend the tree models rely on.
- **Revisit the model quarterly** as more weeks of data accumulate; with only ~200 weekly points, the model has limited history to learn multi-year seasonal effects from.

## Outputs
- `outputs/eda_trend_seasonality.png` — trend, rolling mean, monthly seasonality
- `outputs/actual_vs_predicted.png` — last 12 weeks, actual vs. forecast
- `outputs/feature_importance.png` — top drivers of the winning model
- `outputs/metrics.json` — full metrics and observations

## Run it
```bash
pip install -r ../requirements.txt
python sales_forecasting.py
```
