# Section 6 — Feature engineering

## 6.1 Raw inputs

Each hourly observation begins as raw measurements from the two Open-Meteo endpoints (Section 4), keyed by UTC timestamp:

- **Six pollutant concentrations** — PM2.5, PM10, NO₂, O₃, SO₂, CO (from the CAMS air-quality endpoint)
- **Four meteorological variables** — temperature (2m), relative humidity (2m), wind speed (10m), surface pressure (from the ECMWF weather endpoint)

From these, `utils/features.py` derives the modelling features through a deterministic, idempotent transform — `clean_raw_data` followed by `engineer_features`. Idempotency matters because the hourly pipeline (Section 7) re-processes an overlapping window on every run; the same input row must always produce the same engineered output.

## 6.2 AQI is computed, not taken from the API

Open-Meteo's air-quality endpoint returns a `us_aqi` field, but the project does not use it as the modelling target. AQI is computed independently from the six pollutant concentrations using the US EPA piecewise-linear formula (`utils/aqi_calculator.py`), applying the dominant-pollutant rule: the overall AQI is the maximum of the six pollutant sub-indices. Computing AQI in-house keeps the target definition under the project's control and consistent across the historical backfill and the live pipeline.

The breakpoint logic was hardened during development: the EPA breakpoint tables have small numeric gaps between adjacent bands, and a naive lookup let readings that fell in a gap skip every band and return a spurious AQI of 500. The sub-index function was rewritten to treat each band's upper bound as the next band's lower bound, making the table effectively continuous, with regression tests locking the fix in across all six pollutants.

## 6.3 Engineered features

The raw inputs and the computed AQI are expanded into a candidate feature set spanning four families, each motivated by a finding in the EDA (Section 5):

- **Temporal** — `hour`, `month`, `day_of_week`, `is_weekend`, derived from the timestamp. The diurnal and seasonal cycles in Section 5 motivate `hour` and `month`; `day_of_week` and `is_weekend` were included as candidates despite the EDA showing no weekly cycle, so the selection step could confirm their irrelevance empirically rather than by assumption.
- **Lag** — AQI at 24h, 48h, and 72h prior, plus pollutant lags (e.g. `pm2_5_lag_24h`). These are the autoregressive backbone of the model, motivated directly by the autocorrelation structure measured in Section 5.9.
- **Rolling statistics** — 6-hour and 24-hour rolling means, which smooth short-term noise and give the model a recent-trend signal.
- **Change-rate** — short-term deltas capturing how fast conditions are moving.

Alongside these, the raw pollutants, the weather variables, and the computed AQI are themselves carried as candidate predictors.

## 6.4 Targets and the `has_target` flag

The model is trained to predict AQI at three forward horizons, so each row carries three target columns: AQI shifted forward by 24h, 48h, and 72h. The most recent rows in the store cannot have complete targets — the ground-truth future values do not exist yet — so a `has_target` flag marks which rows are fully labelled. Training filters on this flag, ensuring the model only ever fits on rows whose targets are real observations rather than placeholders. The target shifting is idempotent, consistent with the rest of the transform.

## 6.5 Feature group schema

The engineered rows are stored in the Hopsworks feature group `aqi_features` (version 1), configured in **Stream mode** so writes land in the online store immediately and materialise to the offline store for training and batch reads. The group holds **31 columns**:

| Column class | Count | Notes |
|---|---|---|
| `timestamp` (primary key) | 1 | UTC; the deduplication key |
| Candidate predictor features | 26 | Raw pollutants + weather + computed AQI + temporal + lag + rolling + change-rate |
| Horizon target columns | 3 | AQI shifted +24h / +48h / +72h |
| `has_target` flag | 1 | Marks fully-labelled rows |

Because the primary key is `timestamp`, inserting a row whose timestamp already exists is idempotent — the basis for the safe re-runnable pipeline in Section 7.

## 6.6 From 26 candidates to the 5-feature champion

The 26 candidate predictors were narrowed to the 5 features used by every champion model — `aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity` — through SHAP-based ranking and an ablation sweep. The full method and the supporting figures (SHAP importance and the feature-count ablation curve) are in Section 8.4–8.5; the short version is that performance peaked at exactly five features across all three horizons, and an A/B swap test confirmed those five carry the signal rather than being interchangeable with arbitrary alternatives.

Critically, the feature group still stores all 31 columns even though the champion uses only 5. Keeping the wider set materialised means feature selection can be revisited — a different model, or a re-run of SHAP on more data — without re-engineering or re-backfilling anything. The model selects its five at training time; the store remains the superset.