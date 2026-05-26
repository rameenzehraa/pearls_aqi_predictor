# Section 9 — Conformal prediction intervals

## 9.1 Why prediction intervals — and why conformal

A point forecast (a single AQI number) is not enough for a decision-support tool. A user planning outdoor activity needs to know not only "the model predicts AQI 110 tomorrow" but also **how confident the model is in that prediction**. A point forecast with no uncertainty information is operationally indistinguishable from a guess.

Several methods exist to produce prediction intervals around a point forecast:

- **Gaussian assumption** — assume residuals are normally distributed, derive ±1.96σ intervals. Requires the residuals to actually be Gaussian, which they typically aren't for skewed targets like AQI.
- **Bootstrap** — resample the training set, retrain, observe the spread. Expensive, and the resulting intervals make implicit smoothness assumptions.
- **Bayesian models** — produce posterior predictive distributions directly. Computationally heavier, requires specifying priors.
- **Conformal prediction** — distribution-free, finite-sample coverage guarantee, model-agnostic. Adds a small computational overhead at calibration time and produces intervals with a provable coverage rate under the assumption of exchangeability between calibration and test data.

For this project, conformal prediction was selected. Two main reasons: it does not require Gaussian residuals (the AQI target is right-skewed; see Section 5.3), and its coverage guarantee is **distribution-free** — the intervals hold their nominal coverage regardless of the underlying data distribution, subject only to exchangeability between calibration and test data.

Within the conformal family, the specific method used is **Conformalized Quantile Regression** (CQR; Romano, Patterson & Candès, 2019), in a hybrid variant adapted to keep the production champion's point estimates intact.

## 9.2 The hybrid CQR construction

The standard CQR construction trains a single quantile regression model on the training set, then calibrates against a held-out calibration set. This project departs from the textbook variant in a deliberate way: **the point predictions on the holdout come from the production champion Ridge model (trained on the full pre-holdout pool), while the prediction intervals are derived from a separately-trained quantile regression system fit on a strict subset of that data.**

This trade-off is documented in the conformal literature as **"locally-valid" conformal** — strict exchangeability between calibration and holdout is relaxed in exchange for usable point accuracy from the deployed model. The empirical consequence is that coverage on the holdout typically meets or slightly exceeds the nominal target (the intervals are slightly conservative), which is the right direction to err for a public health-relevant forecasting system.

### 9.2.1 Data splits

The data is split chronologically into three contiguous segments:

| Segment | Fraction of data | Purpose |
|---|---|---|
| Train (proper) | ~56% | Fits the local Ridge used to derive residuals for the quantile regressors |
| Calibration | ~24% | Computes the conformity score `Q_widen` |
| Holdout | 20% | Final per-horizon coverage and interval-width measurement |

The fractions correspond to `HOLDOUT_FRAC = 0.20` and `CALIBRATION_FRAC_OF_TRAIN = 0.30` in `models/conformal_intervals.py`. Concretely, at 24h horizon: ~4,905 train + ~2,103 calibration = 7,008 pre-holdout rows, with 1,752 in the holdout. Exact sizes vary across horizons because lag-feature dropping affects row counts (full details in `calibration.json` per horizon).

### 9.2.2 Construction steps, in order

1. **Fit a local Ridge** on the train (proper) split, using the same 5 features as the production champion (Section 8.7). This local Ridge exists only to generate residuals for the quantile regressors — it is never used for predictions on the holdout.

2. **Compute residuals** `r = y_train − ŷ_local_ridge` on the train (proper) split.

3. **Fit two quantile regressors** on the residuals: `qr_lo` for the 10th percentile and `qr_hi` for the 90th percentile. Both use `sklearn.linear_model.QuantileRegressor` with `alpha=0.001` (light L1 regularization).

4. **Calibration**: on the calibration set, compute predicted residual intervals `[y_pred_local + qr_lo, y_pred_local + qr_hi]` and define the conformity score:
   
   `score = max(predicted_lower − y_actual, y_actual − predicted_upper)`
   
   Take the `⌈(n_calibration + 1) × (1 − α)⌉`-th largest score (with α = 0.20 for nominal 80% coverage) as the conformity quantile `Q`. Clamp at zero to produce `Q_widen = max(Q, 0)`.

5. **Holdout intervals**: for each holdout point, the point prediction is `ŷ = production_champion(x)`. The interval is:
   
   `[ŷ + qr_lo(x) − Q_widen, ŷ + qr_hi(x) + Q_widen]`
   
   The `Q_widen` term symmetrically widens the quantile-regression interval to ensure the conformity guarantee holds on the calibration set.

The asymmetry between point prediction (from production champion) and interval (from a quantile system on local Ridge) is the "hybrid" aspect — calibration is preserved on the calibration set, point accuracy is preserved on the holdout.

## 9.3 Per-horizon calibration values

The trained CQR system is registered in Hopsworks Model Registry as `karachi_aqi_cqr_24h`, `karachi_aqi_cqr_48h`, `karachi_aqi_cqr_72h` (current version 19, registered 2026-05-25). The per-horizon calibration values are:

| Horizon | `Q_widen` | Train/calibration split | Method note |
|---|---|---|---|
| 24h | 13.64 | 4,905 / 2,103 | Symmetric, hybrid |
| 48h | 12.30 | 4,891 / 2,097 | Symmetric, hybrid |
| 72h | 19.27 | 4,878 / 2,091 | Symmetric, hybrid |

`Q_widen` is largest at the 72h horizon (19.27 vs ~13 at the shorter horizons). This is the conformal system honestly registering that 72h-ahead residuals are harder to bound: the data property documented in Section 5.9 (autocorrelation decay) produces wider residuals at longer horizons, which the conformal calibration translates directly into wider intervals.

## 9.4 Empirical coverage on the holdout

The conformal guarantee is that holdout coverage will meet or exceed the nominal 80% target under exchangeability. Measured holdout coverage:

| Horizon | Target coverage | Achieved coverage | Average interval width |
|---|---|---|---|
| 24h | 80% | **86.4%** | 57.7 AQI units |
| 48h | 80% | **82.9%** | 57.3 AQI units |
| 72h | 80% | **86.8%** | 63.6 AQI units |

All three horizons meet or exceed the 80% target. The system is slightly conservative — intervals are a few percentage points wider than the minimum required to hit 80% — which is the correct direction to err.

Interval widths are roughly stable at 57–64 AQI units across horizons. At 24h, where the point R² is 0.350 and the model has the strongest grip on the signal, the interval expresses moderate uncertainty (±29 AQI either side of the point estimate at most). At 72h, where R² drops to 0.129 and predictability has decayed substantially, the interval widens to express the larger uncertainty — wider, but only modestly so, because the underlying target variance is similar at all horizons (Section 5.9 noted: same mean, same std across horizons, only autocorrelation decays).

## 9.5 What the intervals look like in practice

A concrete example from the live system: with current AQI at ~70 (Moderate) and a 24h forecast of point AQI = 69, the conformal interval is approximately [49, 93] — wide enough to express that the model cannot rule out either a recovery to Good or a slip into Unhealthy for Sensitive Groups, but tight enough to rule out Hazardous-range outcomes.

This is honest uncertainty quantification. Users see not just "tomorrow's AQI will be 69" but "tomorrow's AQI is most likely between 49 and 93, with 80% confidence". A decision-support system without that second part is operating on false confidence.

## 9.6 Coverage drift over the course of the project

The notes file `models/conformal_intervals.py` documents the historical progression of coverage:

| Approach | 24h coverage |
|---|---|
| Day 9 — XGBoost intervals on raw target (no conformal) | 57.7% |
| Day 11a — Residual quantile regression on Ridge, no conformal calibration | 73–74% |
| Day 11b — Hybrid CQR with conformal calibration (current) | 86.4% |

The progression — from 58% to 74% to 86% — is the empirical case for adding the conformal calibration step. Quantile regression alone produces under-coverage on real data (the model's residuals don't behave as smoothly as quantile regression assumes); the conformal calibration step corrects for this empirical under-coverage with a finite-sample guarantee.

## 9.7 Limitations and scope of the coverage claim

The coverage figures above are measured on the chronological holdout — the same data segment used for the point-prediction evaluation in Section 8.7 and as Window 1 in Section 13. Re-evaluation of CQR coverage on the Nov 2025 noisy-window simulation and the May 2026 post-deployment window was not performed: the out-of-time verification in Section 13 focused on point-prediction skill against a naive baseline. The 86.4% / 82.9% / 86.8% coverage values should therefore be read as in-distribution holdout coverage — a calibration check on the standard test set, not a claim about coverage stability under distribution shift.

A natural extension would be to repeat the calibration on rolling windows and observe how `Q_widen` evolves as data accumulates. This is documented in Section 15 (Future Work) as a follow-up exercise for an operational deployment.

## 9.8 Reference

Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized Quantile Regression*. Advances in Neural Information Processing Systems 32 (NeurIPS 2019).