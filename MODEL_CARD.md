# Model Card

## Purpose

RetailAI SQL contains two supervised models trained on a deterministic synthetic retail dataset:

1. **Demand forecaster** — estimates item-level daily units sold.
2. **Inventory-risk classifier** — estimates the probability that a store/product combination closes at or below its configured reorder point.

The models are intended for portfolio demonstration and education, not production purchasing decisions.

## Data

- 4 stores
- 15 products across 5 categories
- 2024-01-01 through 2025-12-31
- 43,860 store-product-day rows in the raw `sales_daily` table
- Synthetic effects include weekly/annual seasonality, promotions, product/category demand, store demand differences, replenishment and supplier delays

## Validation design

Feature engineering drops the initial lag window, producing 42,180 usable model rows. The most recent 90 days form the chronological test set:

- Training rows: 36,720
- Holdout rows: 5,460
- Holdout begins: 2025-10-02

This avoids randomly mixing later observations into the training partition.

## Demand forecasting

Candidate models:

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| HistGradientBoostingRegressor | **1.1632** | **1.6045** | **0.9182** | **12.797%** |
| RandomForestRegressor | 1.2275 | 1.7085 | 0.9072 | 13.220% |

Selected artifact: `models/demand_forecaster.joblib`

Features include store/product IDs, weekday/month/weekend, promotion state, unit price, lag-1/7/14 demand, and 7/28-day rolling demand.

## Inventory-risk classification

The target is 1 when the day's closing inventory is at or below the product's configured reorder point.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| HistGradientBoostingClassifier | **0.9668** | **0.9572** | **0.9352** | **0.9461** | **0.9955** |
| LogisticRegression | 0.8941 | 0.7819 | 0.9146 | 0.8430 | 0.9665 |

Selected artifact: `models/inventory_risk_classifier.joblib`

## Limitations

Performance on synthetic data does not demonstrate equivalent performance on real retail data. Real deployment would require leakage audits, stronger backtesting, product/store cold-start handling, holiday/event features, uncertainty estimates, monitoring and retraining on actual operational history.
