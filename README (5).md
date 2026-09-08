# Task 1 — E-Commerce Customer Churn Prediction

Predicts which customers are likely to churn using the **IBM Telco Customer
Churn** dataset (7,043 customers, 21 attributes), so the business can target
retention offers at the right accounts.

## Dataset
- Source: [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- 7,043 rows · overall churn rate **26.5%**

## Pipeline (`churn_prediction.py`)
1. **Clean** — coerce `TotalCharges` to numeric (11 blanks → median imputed), drop ID column.
2. **EDA** — churn rate, churn by contract type, tenure distribution by outcome.
3. **Feature engineering** — tenure buckets, average monthly spend, count of
   subscribed add-on services (security/backup/support/streaming), one-hot
   encoding of all categoricals.
4. **Class imbalance** — SMOTE oversampling applied to the training fold only
   (train: 4,139 no-churn / 1,495 churn → balanced to 4,139/4,139).
5. **Models compared** — Logistic Regression, Random Forest, Gradient Boosting.
6. **Tuning** — `RandomizedSearchCV` (5-fold, optimizing ROC-AUC) on the
   best baseline model.
7. **Evaluation** — precision, recall, F1, ROC-AUC, confusion matrix, feature importance.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.546 | 0.709 | 0.617 | 0.833 |
| Random Forest | 0.540 | 0.602 | 0.569 | 0.821 |
| **Gradient Boosting (baseline)** | 0.540 | 0.751 | 0.629 | 0.838 |
| **Gradient Boosting (tuned)** | 0.549 | 0.634 | 0.588 | 0.829 |

Tuned hyperparameters: `n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.7`

> Note: the untuned baseline actually reached a slightly higher recall/F1 here —
> the tuning objective (ROC-AUC) trades a bit of recall for precision. For a
> retention campaign where missing a churner is costlier than a wasted offer,
> re-run tuning with `scoring="recall"` or lower the classification threshold
> (see `predict_proba` output) instead of the default 0.5.

### Top churn drivers (feature importance)
1. **Tenure** — new customers churn far more than long-tenured ones.
2. **Electronic check payment method** — strongest payment-method risk signal.
3. **Two-year contract** — strongly *protective* against churn.
4. **Monthly charges** / **fiber optic internet** — higher-priced, fiber customers churn more.
5. **Number of add-on services subscribed** — more add-ons → stickier customer.

### Business recommendations
- **Target month-to-month, electronic-check customers in their first year** — this segment has the highest churn probability; prioritize retention outreach here.
- **Incentivize contract upgrades** (1-year/2-year) with modest discounts — contract length is the single strongest protective factor.
- **Bundle add-on services** (security, tech support, streaming) into the first 90 days — customers with 3+ add-ons churn substantially less.
- **Review fiber-optic pricing/support quality** — fiber customers show elevated churn despite being a premium product, suggesting a service-quality or price-perception issue worth investigating separately.

## Outputs
- `outputs/eda_overview.png` — churn distribution, churn by contract, tenure by outcome
- `outputs/final_model_evaluation.png` — confusion matrix + ROC curve
- `outputs/feature_importance.png` — top 15 churn drivers
- `outputs/metrics.json` — full metrics, params, and classification report

## Run it
```bash
pip install -r ../requirements.txt
python churn_prediction.py
```
