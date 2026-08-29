# N₂O Emission Predictor — Wastewater Treatment Plant

Real-time prediction of dissolved N₂O (nitrous oxide) concentration in a full-scale
wastewater treatment plant, using a Random Forest model trained on 2 years of
high-resolution SCADA sensor data.

**Live app:** https://n20-predictor-project.streamlit.app/

## Overview
N₂O is a greenhouse gas ~265x more potent than CO₂, and wastewater treatment is a
major source of it. This project builds an ML pipeline to predict N₂O concentration
from process sensor readings (ammonia, nitrate, oxygen, temperature, airflow, etc.),
so plant operators could get real-time estimates without relying solely on an
expensive, maintenance-heavy N₂O sensor.

## Data
- Source: [Hansen et al. (2024), Mendeley Data](https://data.mendeley.com/datasets/xmbxhscgpr/4) — 2 years of SCADA data from a full-scale Danish WWTP, 2-minute sampling interval.
- 906,815 raw rows → cleaned, resampled to 10-min intervals, feature-engineered down to 39,906 rows × 29 features (lag variables, rolling statistics, interaction terms).

## Approach
- 5 models benchmarked: Linear Regression, Random Forest, XGBoost, SVR, MLP
- **Chronological train/test split** (not random shuffling) to avoid leakage — the model is validated the way it would actually be used, forecasting forward in time
- Hyperparameter tuning via `RandomizedSearchCV` with `TimeSeriesSplit`
- Model interpretation via SHAP

## Results
| Model | Test R² | RMSE (nmol/L) |
|---|---|---|
| **Random Forest** | **0.594** | **4,113** |
| XGBoost (Tuned) | 0.584 | 4,165 |
| XGBoost | 0.571 | 4,230 |
| Linear Regression | 0.526 | 4,446 |

SHAP analysis shows the model relies heavily (~63% importance) on N₂O's own recent
history (autoregressive lag features), with O₂ setpoint and suspended solids as the
next most important process variables — consistent with known nitrification/
denitrification biochemistry.

## Limitations (stated plainly)
- R² ≈ 0.59 is moderate, not high — this is a decision-support tool, not a
  precision instrument, and the report frames it that way.
- The model's strong reliance on N₂O lag features means it's not yet a true
  "soft sensor" — it still needs recent N₂O readings as input, not just
  independent process variables. Removing lag features entirely is listed as
  future work.
- The live demo above exposes 10 of the model's 29 features directly; the rest
  default to representative values for demonstration purposes.

## Stack
Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit

## Repo contents
- `app.py` — Streamlit inference app
- `requirements.txt` — dependencies
- `scaler.pkl`, `features.pkl` — trained preprocessing artifacts
- `rf_model.pkl` — attached as a [GitHub Release](../../releases) (too large for a normal commit); downloaded automatically by `app.py` on first run

## Full report
Full methodology, EDA, residual analysis, and learning curves are in the project report (available on request / linked from CV).
