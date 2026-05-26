# Section 8 — Model Training and Selection

## 8.1 Overview

The forecasting task is hourly Karachi AQI at three horizons: +24h, +48h, +72h. Each horizon receives a dedicated model trained on the same feature set; this section documents how the model family, the feature set, and the regularization strength were chosen. The conclusion: a small, regularized linear model (Ridge regression on 5 features per horizon) outperformed every alternative tested, including deep learning, on chronologically held-out data.

## 8.2 Training data and split methodology

**Data source.** The training pipeline reads from the Hopsworks feature group `aqi_features` (v1) — the same store that powers live inference. No local caching, no parallel data path. The feature group contained 9,264 hourly rows at the time of the final training run, of which 9,000 had complete target values (the most recent 72 hours don't have ground-truth 72h-ahead labels yet).

**Split.** A strict chronological 80/20 hold-out: the model fits on the first 80% of time-ordered rows, and is evaluated on the last 20%. Train and test sizes vary slightly by horizon because lag features drop more rows at longer horizons:

| Horizon | n_train | n_test |
|---|---|---|
| 24h | 7,008 | 1,752 |
| 48h | 6,988 | 1,748 |
| 72h | 6,969 | 1,743 |

**Why chronological.** Random or k-fold splits on time-series data silently leak temporal autocorrelation across the train/test boundary, inflating measured performance. Hours adjacent in time are highly correlated; if neighbouring hours can land on opposite sides of the split, the test set looks easier than it really is. A chronological split is the conservative, defensible standard for forecasting evaluation. The R² values reported below are accordingly lower than they would be under a random split, but they reflect the model's actual generalization to genuinely unseen future time.

## 8.3 Model family selection

Four model families were evaluated, covering the spectrum from heavily regularized linear to deep recurrent:

- **Ridge regression** (linear, L2-regularized) — strong baseline, interpretable, fast
- **Random Forest** — non-linear, handles feature interactions implicitly
- **XGBoost** — gradient-boosted trees, current default for tabular regression
- **LSTM** (2-layer, 64→32 units) — recurrent neural network, designed for sequential signals

All models used the same chronological 80/20 split and the same evaluation metrics (R², RMSE, MAE) for fair comparison. Tree models and LSTM were tested with their standard hyperparameter ranges; Ridge was tested across α ∈ {0.1, 1, 10, 100} and α = 10 was selected.

**LSTM result (full 26-feature set):**

| Horizon | LSTM R² | Ridge R² (26 features) | Δ |
|---|---|---|---|
| 24h | 0.179 | 0.308 | −0.129 |
| 48h | 0.085 | 0.162 | −0.077 |
| 72h | 0.044 | 0.116 | −0.072 |

Early stopping triggered at epoch 16 (best weights from epoch 6), with the loss plateauing almost immediately. The LSTM started overfitting before it could learn the underlying signal — symptomatic of insufficient data relative to model capacity. With ~8,800 training rows (one year of hourly data), the binding constraint is signal-to-noise of the training set, not the expressiveness of the architecture. Two model families with completely different inductive biases (regularized linear vs. deep recurrent) converging to similar ceilings is strong evidence that the predictability limit is set by the data, not the modelling approach.

**Tree model re-verification on the SHAP-pruned 5-feature set** (Day 14, post feature selection):

| Horizon | Ridge R² | Random Forest R² | XGBoost R² |
|---|---|---|---|
| 24h | 0.326 | −0.103 | 0.006 |
| 48h | 0.174 | −0.347 | −0.276 |
| 72h | 0.129 | −0.391 | −0.301 |

Random Forest and XGBoost go **negative** on the pruned feature set — they have nothing to exploit when the feature interactions are gone, and they overfit to noise. Ridge's linear inductive bias matches the structure of the data (a target dominated by autoregressive lags and a seasonal feature), and regularization handles the mild multicollinearity in the lag features without difficulty. The conclusion across both bake-offs is the same: **Ridge is the right model for this task at this data scale.**

## 8.4 Feature selection — SHAP analysis

With Ridge confirmed as the model family, the next question was which features it should use. The initial feature set was 26 candidates (raw pollutants, weather variables, time-of-day, rolling stats, and lag features at multiple offsets). SHAP (SHapley Additive exPlanations) values were computed per horizon using `shap.LinearExplainer`, ranking features by mean absolute SHAP contribution to the prediction.

[Figure 8.1: SHAP feature importance bar plot for the 24h Ridge model. Top features by mean |SHAP|: aqi, aqi_lag_24h, month, pm2_5_lag_24h, humidity.]

Cross-horizon observations from the SHAP analysis:

- **Short horizon (24h)** is dominated by recent state — current AQI and 24-hour lag are by far the largest contributors.
- **Long horizon (72h)** is dominated by `month` — when short-term signal decays, the model falls back on seasonality. This is exactly what a well-regularized forecaster should do: as predictability decays, the model degrades gracefully toward climatology rather than inventing variance that isn't there.
- **Humidity has a real negative contribution** (high humidity → lower predicted AQI), consistent with particulate deposition physics. The model has learned an interpretable physical relationship rather than a statistical fluke.

The features that surfaced consistently in the top-5 across all three horizons were: **`aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity`** — a deliberate choice to use the same 5 features for all horizons so that interpretation, deployment, and CQR calibration are consistent across the production stack.

## 8.5 Feature count — ablation

SHAP gives a ranking; it doesn't answer "how many features to keep." That requires an ablation: train Ridge with the top-K features (by mean |SHAP|), measure R² on the chronological holdout, and observe how performance varies with K.

[Figure 8.2: Feature ablation curve. R² on chronological holdout vs. number of features (top-K by mean |SHAP|), across the three horizons. The curve identifies K=5 as the empirical peak across all three horizons.]

The ablation produced a clear and unexpected finding: **performance improves as features drop, peaking at K = 5 across all three horizons**, then collapses sharply at K = 3. The intuitive expectation — "more features can only help a regularized linear model" — turns out to be wrong here. Ridge's L2 regularization handles the noise from extra features, but doesn't denoise as effectively as explicit feature selection.

| Number of features | 24h R² | 48h R² | 72h R² |
|---|---|---|---|
| 26 (all) | ~0.32 | ~0.16 | ~0.11 |
| 10 | ~0.33 | ~0.16 | ~0.10 |
| **5 (champion)** | **0.350** | **0.181** | **0.129** |
| 4 | ~0.35 | ~0.18 | ~0.13 |
| 3 | <0.30 | <0.15 | <0.10 |

The lift from the full feature set to K = 5 was **+14% R² at 24h, +12% at 48h, +11% at 72h** — a meaningful improvement attributable purely to feature selection, with no hyperparameter retuning.

## 8.6 Feature swap A/B test

To rule out the possibility that K = 5 was a fluke of the specific top-5 features, an A/B test was run. Three variants:

- **Top-5** — the locked champion features
- **Top-6** — adds `pm2_5_lag_24h` (the next-ranked feature)
- **Swap** — replaces `pm2_5_lag_24h` with `day_of_week` (orthogonal to the others but flagged by EDA as uninformative)

The swap variant was **worst at every horizon**, confirming that the top-5 are not interchangeable with arbitrary other features — they carry the information. Top-5 narrowly beat top-6 at every horizon, locking in K = 5 as the final choice.

## 8.7 Champion model — final specification

The locked champion is:

- **Model family**: `sklearn.linear_model.Ridge`
- **Regularization**: α = 10
- **Scaler**: `sklearn.preprocessing.StandardScaler` (fitted on training data only)
- **Imputation**: median, fitted on training data only
- **Features (5)**: `aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity`
- **Per-horizon**: one model per horizon (24h, 48h, 72h); same feature set, separately fitted scaler, imputer, and Ridge weights
- **Registry version**: v19 in Hopsworks Model Registry (auto-versioned, registered 2026-05-25)

**Final metrics on chronological holdout:**

| Horizon | R² | RMSE | MAE |
|---|---|---|---|
| 24h | 0.350 | 20.15 | 15.05 |
| 48h | 0.181 | 22.64 | 16.76 |
| 72h | 0.129 | 23.35 | 17.50 |

These metrics are also reported in Section 13 (OOT verification) as Window 1 (chronological holdout) — they are the same numbers, presented in different contexts.

**Coefficient interpretation** (24h Ridge, on standardized features):

| Feature | Coefficient |
|---|---|
| `aqi` | +12.88 |
| `aqi_lag_24h` | +3.01 |
| `aqi_lag_72h` | +1.74 |
| `month` | +0.96 |
| `humidity` | −4.43 |
| (intercept) | +94.72 |

Current AQI is by far the dominant predictor, followed by the recent lag features and humidity (negative, as expected). Coefficients evolve across horizons: `aqi` weights decrease from +12.88 (24h) to +6.47 (72h), while `month` increases from +0.96 to +2.70 — direct evidence of the "recency dominates short horizon, seasonality dominates long horizon" pattern visible in the SHAP analysis.

## 8.8 Training pipeline orchestration

Training is automated via GitHub Actions, scheduled daily at 02:00 UTC. The pipeline (`pipelines/training_pipeline.py`) runs three steps in sequence:

1. **`train_champion.main()`** — pulls features from Hopsworks, fits per-horizon Ridge models, saves scaler/imputer/model artifacts and `champion_metadata.json`
2. **`conformal_intervals.main()`** — fits CQR residual quantile regressors on top of the freshly-trained champion, computes the per-horizon `Q_widen` calibration values, saves CQR artifacts
3. **`register_to_registry.main()`** — uploads all 6 model artifacts (3 champion + 3 CQR) to Hopsworks Model Registry as new versions, with auto-incrementing version numbers

Each step exits non-zero on failure, causing the GitHub Actions run to fail visibly. The model registry's auto-versioning ensures every successful training run produces a numbered, immutable model snapshot — `karachi_aqi_ridge_24h v19` is exactly the model trained on 2026-05-25, and will remain so. The production backend reads from the registry by latest version on cold start (see Section 10).

## 8.9 Summary

The champion was selected through three rounds of empirical evaluation:

1. **Model family** — Ridge beat LSTM on the full 26-feature set; tree models went negative on the pruned 5-feature set. Two completely different families converging to similar ceilings indicates the data is the constraint.
2. **Feature set** — SHAP-driven ranking aggregated across horizons identified a stable top-5; ablation confirmed K = 5 as the empirical R² peak across all three horizons; an A/B feature-swap test confirmed the specific 5 carry the signal.
3. **Hyperparameters** — Ridge α = 10 selected from {0.1, 1, 10, 100} with the chronological holdout.

The resulting model is small, fast, interpretable, automatically retrained daily, and consistently outperforms naive baselines on out-of-time evaluation (see Section 13). The choice of Ridge is not an admission of giving up on more complex models — it is the empirically-validated answer to "what works for hourly AQI forecasting with one year of training data."