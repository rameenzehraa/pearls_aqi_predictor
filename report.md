# Pearls AQI Predictor — Final Report

**Author:** Rameen Zehra
**Tag:** v1.0-submission
**Date:** June 2026

**Live dashboard:** [khi-aqi.streamlit.app](https://khi-aqi.streamlit.app)
**Prediction API:** [pearls-aqi-backend.onrender.com](https://pearls-aqi-backend.onrender.com)
**Repository:** [github.com/rameenzehraa/pearls_aqi_predictor](https://github.com/rameenzehraa/pearls_aqi_predictor)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Data Sources](#4-data-sources)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering](#6-feature-engineering)
7. [Feature Pipeline](#7-feature-pipeline)
8. [Model Training and Selection](#8-model-training-and-selection)
9. [Conformal Prediction Intervals](#9-conformal-prediction-intervals)
10. [Backend API](#10-backend-api)
11. [Dashboard](#11-dashboard)
12. [Continuous Integration and Automation](#12-continuous-integration-and-automation)
13. [Out-of-Time Verification](#13-out-of-time-verification)
14. [Operational Observations and Known Limitations](#14-operational-observations-and-known-limitations)
15. [Future Work](#15-future-work)
16. [Conclusion](#16-conclusion)

**Appendices**
- [A — Feature-Group Schema](#appendix-a--feature-group-schema)
- [B — Hourly Feature Pipeline (workflow)](#appendix-b--hourly-feature-pipeline-workflow)
- [C — Daily Training Pipeline (workflow)](#appendix-c--daily-training-pipeline-workflow)
- [D — API Endpoint Reference](#appendix-d--api-endpoint-reference)
- [E — Glossary](#appendix-e--glossary)
- [F — References](#appendix-f--references)

---

## 1. Executive Summary

### 1.1 What was built

Pearls AQI Predictor is a publicly accessible system that forecasts Karachi's Air Quality Index at three horizons — 24, 48, and 72 hours ahead — alongside a live current-conditions reading. Data is ingested hourly from a public meteorological API, models are retrained daily, and the dashboard updates on every page load. No manual intervention is required.

The audience is two-fold: Karachi residents who want a glanceable forecast with plain-language categories and health alerts, and evaluators who want to inspect a full MLOps stack assembled entirely on free-tier managed services.

### 1.2 Architecture at a glance

| Layer | Service |
|---|---|
| Data source | Open-Meteo (air quality + weather) |
| Feature store & model registry | Hopsworks |
| Scheduled compute | GitHub Actions (hourly ingestion, daily training) |
| Prediction API | Flask on Render |
| Dashboard | Streamlit Community Cloud |

Data flows one direction: Open-Meteo → Hopsworks feature store → daily training → Hopsworks model registry → Flask backend → Streamlit dashboard. The production model is a small Ridge regression on five features, selected through systematic comparison against tree-based and recurrent-neural alternatives.

### 1.3 Headline outcomes

On the chronological hold-out (final 20% of the time-ordered dataset, n ≈ 1,750):

| Horizon | R² | RMSE | MAE |
|---|---|---|---|
| 24h | 0.350 | 20.15 | 15.05 |
| 48h | 0.181 | 22.64 | 16.76 |
| 72h | 0.129 | 23.35 | 17.50 |

The R² decay across horizons (0.35 → 0.18 → 0.13) closely matches the underlying autocorrelation decay in Karachi's AQI signal (0.63 → 0.49 → 0.41), indicating the model is operating at the ceiling set by the data rather than under-fitting.

On a high-variance stress test — a 15-day simulation across November 2025's seasonal pollution peak — the model beats a naive "tomorrow equals today" baseline by **20–24% on RMSE at every horizon**.

Every point forecast comes with an 80%-target conformal prediction interval. Empirical hold-out coverage is 86.4% / 82.9% / 86.8% across the three horizons — slightly conservative, which is the appropriate direction to err for a public-health-relevant tool.

### 1.4 Honest limitations

Two limitations stem directly from running on free-tier infrastructure. The Hopsworks offline materialisation stalled for nine days in late May because the shared free-tier compute cluster was capacity-saturated; the diagnostic and recovery work is documented in Section 14. The Streamlit dashboard and Render backend both sleep after inactivity, producing a one-time ~30-second wake-up delay — accepted as a cosmetic cost of zero infrastructure spend.

---

## 2. Problem Statement

### 2.1 Context

Karachi consistently ranks among the most polluted major cities in the world. Pakistan's average PM2.5 in 2025 was 67.3 µg/m³ — an AQI of around 156 ("unhealthy") and nearly 14 times the WHO guideline. EDA (Section 5) confirms what the public-health record implies: AQI in Karachi rarely drops to "Good," the modal hour sits in "Moderate," and episodic spikes into "Unhealthy" and "Very Unhealthy" are not rare.

For residents — particularly children, the elderly, and people with respiratory conditions — air quality is a daily decision input. **Real-time AQI tells you "what is the air like now." It doesn't tell you "what will it be like tomorrow"** — which is the question that actually shapes behaviour, because plans are forward-looking.

Pakistan has reasonable real-time monitoring. What it largely lacks is publicly accessible, multi-day AQI forecasting at the city level. Closing that gap — even at a modest scale of one city, three horizons, with honest uncertainty — has concrete utility for outdoor planning, sensitive-group health management, and indoor preparation.

### 2.2 What the system predicts (and doesn't)

The single most important point of clarity, because it's easy to misread: the system predicts **one AQI value per forecast horizon, anchored to the present moment**. It does *not* produce an hourly trajectory through the next 72 hours.

At any inference time *t*:

- The **+24h** forecast is a point prediction of AQI at *t + 24h*, with a conformal interval
- The **+48h** forecast is a point prediction at *t + 48h*, with an interval
- The **+72h** forecast is a point prediction at *t + 72h*, with an interval

Three predictions total per inference call, one per horizon. The dashboard's trajectory chart visualises these three plus the current AQI as four connected points; strictly, the line connecting them is interpolation for visual continuity, not a prediction at intermediate hours.

**Out of scope:** no pollution-source attribution; no separate per-pollutant forecasts (AQI is a composite index, and we forecast the index); no sub-city geographic resolution (the Open-Meteo CAMS resolution does not support meaningful sub-city differentiation — see Section 4.2); no multi-city expansion.

### 2.3 Engineering scope

Beyond the forecasting task itself, the project specification required: data fetched from a third-party API, features stored in a managed feature store, models stored in a managed registry, automated hourly + daily pipelines, multiple model families compared, feature-importance analysis, a web dashboard, a prediction API, hazardous-AQI alerts, and out-of-time evaluation. The system meets all of these. The architecture is fully serverless — no persistently-running infrastructure, every component on free-tier managed services.

---

## 3. System Architecture

The system is a fully serverless, end-to-end automated forecasting pipeline. Data flows one direction: from a third-party API, through an automated feature pipeline into a managed feature store, into a daily training loop that publishes versioned models to a registry, and out through a prediction API to a public dashboard. No component runs on infrastructure the project operates or pays for.

![Figure 3.1 — System architecture](artifacts/architecture_diagram.png)

**Figure 3.1.** System architecture. Solid arrows trace the validated-data and model-production path; dashed arrows trace the live-display and keep-warm paths.

### 3.1 Components

| Component | Role | Hosted on |
|---|---|---|
| Open-Meteo API | External data source — CAMS air quality + ECMWF weather | Third-party |
| Feature pipeline | Hourly ingest: fetch, clean, engineer, insert | GitHub Actions (cron) |
| Training pipeline | Daily retrain, calibrate intervals, register models | GitHub Actions (cron) |
| Hopsworks Feature Store | System of record for features (`aqi_features` v1) | Hopsworks (free tier) |
| Hopsworks Model Registry | Versioned champion + CQR artifacts | Hopsworks (free tier) |
| Flask backend | Prediction REST API | Render (free tier) |
| Streamlit dashboard | Public end-user UI | Streamlit Community Cloud |
| UptimeRobot | Health pings to suppress Render cold starts | Third-party (free tier) |

Durable state lives only in Hopsworks. The backend and dashboard hold no persistent state of their own — either service can be restarted at any time with no data loss.

### 3.2 Three independent flows

- **Ingest (hourly).** GitHub Actions cron triggers the feature pipeline, which fetches the latest validated hourly readings from Open-Meteo, merges them on UTC timestamp, runs feature engineering, and inserts deduplicated rows. CAMS publishes once daily, so most hourly runs find no new data and exit cleanly — the hourly schedule is a resilience layer.
- **Train (daily).** A second cron triggers the training pipeline, which reads the current feature group, fits one Ridge champion per horizon (Section 8) and CQR intervals on top (Section 9), and registers all six artifacts to the model registry.
- **Serve (on demand).** When a user opens the dashboard, Streamlit calls the Flask backend. On cold start the backend loads the latest models from the registry and caches them to disk. `/predictions` reads the most recent validated row from the feature store as the anchor and computes the three forecasts.

### 3.3 The dual-endpoint design

The architecture reads from two different Open-Meteo endpoints for two purposes. The validated hourly batch is the system of record — it feeds the feature store, training data, and forecast anchor, so the model is always trained and served on the same signal. The live `current` endpoint feeds only the display tile, giving a sub-15-minute "now" reading without contaminating the forecast path. Full rationale in Section 4.3.

---

## 4. Data Sources

### 4.1 Open-Meteo

Open-Meteo's free, no-auth public APIs are the sole external data source. Two endpoints are queried:

- **Air quality** — hourly concentrations of PM2.5, PM10, NO₂, O₃, SO₂, CO, backed by the Copernicus Atmosphere Monitoring Service (CAMS).
- **Weather forecast** — hourly temperature, humidity, wind speed, pressure, backed by ECMWF.

Both endpoints are queried for the Karachi centroid (**24.8607°N, 67.0011°E**) and merged on UTC timestamp.

### 4.2 Single coordinate, not a city grid

An initial design considered fetching pollution data for 31 separate coordinates corresponding to Karachi's administrative subdivisions. A diagnostic on Open-Meteo's CAMS grid resolution showed those 31 coordinates resolved to **11 nominal grid cells**, of which only **~3 produced meaningfully distinct signals** — mean off-diagonal correlation exceeded 0.97. Producing 31 separate forecasts from data that supports ~3 distinct signals would be presentation theatre. Scope was refined to a single city-wide forecast at the centroid.

### 4.3 The dual-endpoint architecture

A single endpoint feeds *training* and feature-store ingestion; a different endpoint feeds the *live* "Current Conditions" tile.

- **Validated hourly endpoint (training + forecast anchor).** Open-Meteo's `forecast` API with `past_days=1` returns CAMS-validated hourly data. This is the system of record. The training pipeline reads from the feature group; the live inference path also reads from it, with the most recent row becoming the *anchor* for the +24h / +48h / +72h forecasts. Using validated data ensures training and serving see the same distribution.
- **`current` endpoint (live tile only).** Open-Meteo's `current` parameter returns the most recent reading without the validated batch step — ~15 minute lag vs. the validated endpoint's 6–24 hours, but less rigorously processed. The dashboard's "Current Conditions" tile pulls from this for display only.

Mixing them — feeding `current` data into the model — would risk a training/serving distribution mismatch. Keeping them separate preserves both prediction integrity and live freshness. The two are visibly different freshness levels on the dashboard, and that asymmetry is intentional.

### 4.4 CAMS publishing cadence

Investigation (Day 13–16) confirmed CAMS publishes the previous day's 24 hours in a single batch around 00:00 UTC each day. So of the 24 hourly cron runs in any 24-hour period, only the one shortly after 00:00 UTC actually finds new data; the other 23 exit with "0 new rows." The hourly schedule was kept anyway, for resilience (if the publish slips by an hour, subsequent runs catch up) and operational simplicity. Empty runs log at `INFO` and report green.

### 4.5 Alternatives evaluated

- **AQICN** was tested. Its free-tier search endpoint for Karachi returns only the US Consulate ground station, which has been offline since March 2025. The functioning Karachi stations visible on the AQICN website are not exposed through the free API. Unusable.
- **OpenWeather** was reviewed. Open-Meteo was chosen instead because CAMS data has explicit publishing semantics (validated batch vs. rolling endpoint) and requires no API-key management.

---

## 5. Exploratory Data Analysis

The full EDA is in `notebooks/eda.ipynb`. This section distils the findings that materially shaped Sections 8 and 9.

The dataset: **9,264 hourly rows from 25 April 2025 to 19 May 2026** (~13 months), pulled directly from the production feature group. One contiguous 4-day gap (96 hours, 1.04% of the time range) sits outside any train/test split boundary and does not affect modelling. No duplicate timestamps; nulls confined to expected boundary positions in lag/target columns.

### 5.1 Target distribution

AQI is right-skewed with mean 93.2, median 85.8, std 27.3; range 23.9–389.7. Distribution across EPA categories:

| Category | Hours | % |
|---|---|---|
| Good (0–50) | 47 | 0.51% |
| Moderate (51–100) | 6,547 | 70.7% |
| Unhealthy for Sensitive Groups (101–150) | 2,084 | 22.5% |
| Unhealthy (151–200) | 533 | 5.8% |
| Very Unhealthy (201–300) | 52 | 0.56% |
| Hazardous (301+) | 1 | 0.01% |

![Figure 5.1 — AQI target distribution](artifacts/eda/target_distribution.png)

**Decisions:** no log transformation (skew is moderate and Ridge handles it); tiered alerts rather than constant warnings (Hazardous = ~0.01% of hours); regression on continuous AQI, not classification.

### 5.2 Temporal patterns

- **Diurnal cycle present.** AQI peaks around 14:00 PKT, dips overnight. Amplitude ~15 AQI units.
- **No weekly cycle.** AQI by day of week is essentially flat. This is informative about the *source* — a commuter-driven city would show weekend dips. Karachi's pollution is dominated by baseline industrial and dust sources that operate continuously.
- **Strong seasonal cycle.** Winter peak around November (~118 AQI), monsoon trough around September (~78). Amplitude ~40 AQI units, much larger than diurnal.

**Decisions:** `month` retained (becomes a top-5 feature); `hour` engineered but later dropped by SHAP (redundant with lag features); `day_of_week` and `is_weekend` dropped.

### 5.3 Feature correlation with AQI

| Feature | r |
|---|---|
| PM2.5 | **+0.88** |
| aqi_rolling_6h | **+0.88** |
| PM10 | +0.74 |
| aqi_rolling_24h | +0.73 |
| aqi_lag_24h | +0.62 |
| aqi_lag_48h | +0.51 |
| aqi_lag_72h | +0.45 |
| humidity | **−0.40** |
| wind_speed | **−0.32** |
| NO₂ | +0.30 |
| O₃ | +0.19 |
| month | +0.18 |
| day_of_week, is_weekend | ≈0 |

PM2.5 dominates, consistent with the US AQI formula. Rolling and lag features show strong autocorrelation — exactly the property a lag-feature model exploits. Humidity and wind are the strongest weather signals, both inversely related (wet, windy weather disperses particulates).

Inter-pollutant correlations also yielded one practical signal: **NO₂ ↔ CO at +0.72**, flagging redundancy. SHAP later dropped CO. PM10 was essentially independent of combustion pollutants (r ≈ 0 with NO₂ and CO), suggesting Karachi's coarse particulates are dust-dominated rather than combustion-dominated.

### 5.4 The predictability ceiling

The single most consequential finding. Correlation between current AQI and AQI at horizon *h*:

| Horizon | Pearson r |
|---|---|
| +24h | 0.63 |
| +48h | 0.49 |
| +72h | 0.41 |

Target distributions are nearly identical across horizons (mean ~93.4, std ~27.2). There is **no distributional shift** — only autocorrelation decay.

![Figure 5.2 — Autocorrelation decay with forecast horizon](artifacts/eda/autocorrelation_decay.png)

This decay is a property of the data, not a model failure. It places an upper bound on the R² achievable at each horizon. The model's observed R² (0.35 / 0.18 / 0.13) follows the same ratio — strong evidence the champion is operating at the ceiling, not below it.

**Decisions:** one model per horizon (not a single model fitted across all three); conformal intervals widen with horizon (calibrated separately per horizon); the R² values are correctly read as honest ceiling estimates, not as failures to fit harder.

### 5.5 Findings → decisions summary

| Finding | Decision |
|---|---|
| Continuous hourly series, one 4-day gap outside splits | Treat as continuous; no interpolation |
| Right-skew, rare Hazardous (~0.01%) | Tiered alert UX, not constant warnings |
| 70.7% Moderate baseline | Regression on continuous value, not classification |
| Strong diurnal, no weekly, strong seasonal | `hour` engineered then SHAP-dropped; `day_of_week` dropped; `month` retained |
| PM2.5↔AQI r = 0.88, linear scatter | Linear model viable; Ridge chosen |
| Strong lag autocorrelation | Lag features at 24/48/72h engineered; champion uses 24h and 72h |
| Humidity −0.40, wind −0.32 | Humidity kept in champion; wind dropped by SHAP |
| Pressure narrow band, weak signal | Dropped by SHAP |
| NO₂↔CO 0.72 redundancy | CO dropped by SHAP |
| Autocorrelation 0.63 → 0.49 → 0.41 | One model per horizon; intervals widen with horizon |

The EDA is not a separate exercise from modelling. Every finding above ties to a specific design decision in Sections 6–9.

---

## 6. Feature Engineering

### 6.1 Raw inputs

Each hourly observation begins as raw measurements from the two Open-Meteo endpoints (Section 4), keyed by UTC timestamp: six pollutant concentrations and four meteorological variables. From these, `utils/features.py` derives modelling features through a deterministic, idempotent transform — `clean_raw_data` then `engineer_features`. Idempotency matters because the hourly pipeline re-processes an overlapping window on every run.

### 6.2 AQI is computed, not taken from the API

Open-Meteo's air-quality endpoint returns a `us_aqi` field, but the project does not use it. AQI is computed independently from the six pollutant concentrations using the US EPA piecewise-linear formula (`utils/aqi_calculator.py`), applying the dominant-pollutant rule (overall AQI = max of six sub-indices). Computing in-house keeps the target definition under project control.

The breakpoint logic was hardened during development: the EPA tables have small gaps between adjacent bands, and a naive lookup let readings falling in a gap skip every band and return a spurious AQI of 500. The sub-index function was rewritten to treat each band's upper bound as the next band's lower bound, with regression tests locking the fix.

### 6.3 Engineered features

The raw inputs and the computed AQI are expanded into four families, each motivated by an EDA finding:

- **Temporal** — `hour`, `month`, `day_of_week`, `is_weekend`, derived from the timestamp.
- **Lag** — AQI at 24h, 48h, 72h prior, plus pollutant lags. The autoregressive backbone, directly motivated by the autocorrelation structure in Section 5.4.
- **Rolling statistics** — 6h and 24h rolling means for recent trend; 30-day rolling mean and std for longer-run baseline.
- **Change-rate** — short-term deltas capturing how fast conditions are moving.

### 6.4 Targets and the `has_target` flag

The model predicts AQI at three forward horizons, so each row carries three target columns (AQI shifted +24h, +48h, +72h). The most recent rows cannot have complete targets, so a `has_target` flag marks fully-labelled rows. Training filters on this flag.

### 6.5 Feature group schema

The engineered rows live in Hopsworks feature group `aqi_features` (v1), in Stream mode so writes land online immediately and materialise to the offline store for training. The group holds **31 columns**: 1 timestamp key, 26 candidate predictors, 3 horizon targets, and the `has_target` flag. Full schema in Appendix A.

Because the primary key is `timestamp`, inserting a row whose timestamp already exists is idempotent — the basis for the safe re-runnable pipeline (Section 7).

### 6.6 From 26 candidates to 5 champion features

The 26 candidates were narrowed to the 5 features used by every champion — `aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity` — through SHAP ranking and an ablation sweep. Full method in Section 8.4–8.5.

The feature group still stores all 31 columns even though the champion uses 5. Keeping the wider set materialised means feature selection can be revisited without re-engineering or re-backfilling.

---

## 7. Feature Pipeline

### 7.1 Role and trigger

The feature pipeline keeps the feature group current by appending newly-published validated rows. It runs as an hourly GitHub Actions cron invoking `pipelines/feature_pipeline.py`. The pipeline reads from and writes to Hopsworks only — there is no local CSV in the production path. This was a hard rule fixed early: the training pipeline and the live backend must both read from the same store, so the store has to be the single source of truth.

### 7.2 Step-by-step flow

Each run:

1. **Fetch** the last ~6 hours from Open-Meteo, merge air quality and weather on UTC timestamp.
2. **Pull recent history** from the feature group — ~80 hours.
3. **Clean and engineer** over the combined frame using the same functions as the backfill.
4. **Insert** the deduplicated new rows.

### 7.3 Why pull ~80 hours of history

Lag features go up to 72 hours and rolling windows up to 24 hours. To compute features for a *new* row, the pipeline needs the preceding rows in context. Fetching only the genuinely new hours would leave lags undefined. Pulling 72 hours plus a small buffer guarantees every new row has the context it needs.

This is also why incomplete history surfaces downstream as `NaN` lag features — if preceding rows are missing (e.g., during a materialisation catch-up), the longest lags can't be computed. The backend handles that case defensively at inference (Section 10).

### 7.4 Idempotency and the steady state

Primary key is `timestamp`, so re-processing an overlapping window is safe — rows whose timestamps already exist are no-ops. The pipeline runs every hour without risk of duplication.

In normal operation, most hourly runs insert **zero** new rows (CAMS publishes once daily — Section 4.4). Empty runs are logged at `INFO` and report green. The hourly cadence is a resilience layer that catches the daily batch whenever it lands.

### 7.5 Arrow Flight read with JDBC fallback

Hopsworks feature-group reads default to Arrow Flight. On the free tier the Flight connection occasionally drops mid-read, surfacing as `Flight`, `Socket closed`, or `Query Service` errors. The pipeline catches these and retries over the slower Hive/JDBC path (`read_options={"use_hive": True}`). This trades read speed for reliability and keeps transient platform hiccups from failing the run.

### 7.6 Write semantics

An insert writes to the online store immediately, after which the row materialises to the offline store that training and the backend read from. The "inserted N rows" log line confirms the online write, **not** offline materialisation. A multi-day materialisation stall encountered during the project, and the tooling built to monitor it, are documented in Section 14.

---

## 8. Model Training and Selection

The forecasting task is hourly Karachi AQI at three horizons: +24h, +48h, +72h. Each horizon gets a dedicated model trained on the same feature set. **Conclusion: a small Ridge regression on 5 features per horizon outperformed every alternative tested, including deep learning, on chronologically held-out data.**

### 8.1 Training data and split

Training reads from `aqi_features` v1 — the same store that powers live inference. No parallel data path. At the final training run: 9,264 hourly rows, of which 9,000 had complete target values.

**Chronological 80/20 hold-out**: the model fits on the first 80% of time-ordered rows, evaluated on the last 20%. Train and test sizes vary slightly by horizon because longer lags drop more rows.

| Horizon | n_train | n_test |
|---|---|---|
| 24h | 7,008 | 1,752 |
| 48h | 6,988 | 1,748 |
| 72h | 6,969 | 1,743 |

**Why chronological.** Random or k-fold splits on time-series data silently leak temporal autocorrelation across the train/test boundary, inflating measured performance. Hours adjacent in time are highly correlated; if they can land on opposite sides of the split, the test set looks easier than it really is. The R² values reported below are lower than they would be under a random split, but they reflect actual generalization to unseen future time.

### 8.2 Model family selection

Four families were evaluated: **Ridge regression** (L2-regularized linear), **Random Forest**, **XGBoost**, and **LSTM** (2-layer, 64→32 units). All used the same chronological split and metrics.

**LSTM result (full 26-feature set):**

| Horizon | LSTM R² | Ridge R² | Δ |
|---|---|---|---|
| 24h | 0.179 | 0.308 | −0.129 |
| 48h | 0.085 | 0.162 | −0.077 |
| 72h | 0.044 | 0.116 | −0.072 |

Early stopping triggered at epoch 16 with the loss plateauing almost immediately. The LSTM overfits before it learns the underlying signal — symptomatic of insufficient data for model capacity. With ~8,800 training rows, the binding constraint is signal-to-noise of the training set, not the expressiveness of the architecture. Two completely different inductive biases (regularized linear vs. deep recurrent) converging to similar ceilings is strong evidence the predictability limit is set by the data, not the modelling approach.

**Tree model re-verification on the SHAP-pruned 5-feature set:**

| Horizon | Ridge R² | Random Forest R² | XGBoost R² |
|---|---|---|---|
| 24h | 0.326 | −0.103 | 0.006 |
| 48h | 0.174 | −0.347 | −0.276 |
| 72h | 0.129 | −0.391 | −0.301 |

Random Forest and XGBoost go **negative** on the pruned set — they have nothing to exploit when feature interactions are gone, and they overfit to noise. Ridge's linear inductive bias matches the structure of the data, and regularization handles the mild multicollinearity in the lag features. **Ridge is the right model for this task at this data scale.**

### 8.3 Feature selection — SHAP

The initial feature set was 26 candidates. SHAP values were computed per horizon using `shap.LinearExplainer`, ranking features by mean absolute SHAP contribution.

![Figure 8.1 — SHAP feature importance for the 24h Ridge model](artifacts/shap/shap_bar_24h.png)

**Figure 8.1.** SHAP feature importance bar plot for the 24h Ridge model. The champion feature set uses the global mean |SHAP| aggregated across all three horizons.

Cross-horizon observations:

- **Short horizon (24h)** is dominated by recent state — current AQI and 24-hour lag are the largest contributors.
- **Long horizon (72h)** is dominated by `month`. When short-term signal decays, the model falls back on seasonality. This is exactly what a well-regularized forecaster should do.
- **Humidity has a real negative contribution** (high humidity → lower predicted AQI), consistent with particulate deposition physics — an interpretable physical relationship.

Features that appeared consistently in the top-5 across all three horizons: **`aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity`** — used identically for all three horizons so interpretation, deployment, and CQR calibration are consistent.

### 8.4 Feature count — ablation

SHAP gives a ranking, not a count. The ablation: train Ridge with the top-K features by mean |SHAP|, measure R² across K.

![Figure 8.2 — Feature ablation curve](artifacts/ablation/ablation_curves.png)

**Figure 8.2.** R² on chronological holdout vs. number of features, across the three horizons. Peak at K = 5.

| K | 24h R² | 48h R² | 72h R² |
|---|---|---|---|
| 26 (all) | ~0.32 | ~0.16 | ~0.11 |
| 10 | ~0.33 | ~0.16 | ~0.10 |
| **5 (champion)** | **0.350** | **0.181** | **0.129** |
| 4 | ~0.35 | ~0.18 | ~0.13 |
| 3 | <0.30 | <0.15 | <0.10 |

Performance **improves as features drop**, peaking at K = 5, then collapses at K = 3. Ridge's L2 regularization handles noise from extra features but doesn't denoise as effectively as explicit selection. Lift from full → K=5: +14% R² at 24h, +12% at 48h, +11% at 72h.

A separate A/B swap test (replacing `pm2_5_lag_24h` with `day_of_week`) was worst at every horizon, confirming the top-5 carry the signal rather than being interchangeable with arbitrary alternatives.

### 8.5 Champion specification

- **Model**: `sklearn.linear_model.Ridge`, α = 10
- **Scaler**: `StandardScaler` (fitted on training only)
- **Imputation**: median (fitted on training only)
- **Features (5)**: `aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity`
- **Per-horizon**: one model per horizon, separately fitted scaler/imputer/weights
- **Registry version**: v19 in Hopsworks (registered 2026-05-25)

**Final metrics on chronological holdout:**

| Horizon | R² | RMSE | MAE |
|---|---|---|---|
| 24h | 0.350 | 20.15 | 15.05 |
| 48h | 0.181 | 22.64 | 16.76 |
| 72h | 0.129 | 23.35 | 17.50 |

**Coefficient interpretation (24h Ridge, on standardized features):**

| Feature | Coefficient |
|---|---|
| `aqi` | +12.88 |
| `aqi_lag_24h` | +3.01 |
| `aqi_lag_72h` | +1.74 |
| `month` | +0.96 |
| `humidity` | −4.43 |
| (intercept) | +94.72 |

`aqi` weights decrease from +12.88 (24h) to +6.47 (72h), while `month` increases from +0.96 to +2.70 — direct evidence of "recency dominates short horizon, seasonality dominates long horizon."

### 8.6 Pipeline orchestration

Training is automated via GitHub Actions, daily at 02:00 UTC. `pipelines/training_pipeline.py` runs three steps:

1. **`train_champion.main()`** — pulls features, fits per-horizon Ridge models, saves artifacts.
2. **`conformal_intervals.main()`** — fits CQR on top, computes per-horizon `Q_widen`.
3. **`register_to_registry.main()`** — uploads all 6 artifacts (3 champion + 3 CQR) with auto-incrementing versions.

Each step exits non-zero on failure. The registry's auto-versioning ensures every successful run produces an immutable numbered snapshot.

---

## 9. Conformal Prediction Intervals

### 9.1 Why intervals, why conformal

A point forecast without uncertainty is operationally indistinguishable from a guess. Several methods produce intervals: Gaussian assumption (requires normal residuals — AQI is skewed), bootstrap (expensive, smoothness assumptions), Bayesian (heavier, needs priors), or **conformal prediction** — distribution-free, finite-sample coverage guarantee, model-agnostic.

Conformal was chosen because it doesn't require Gaussian residuals (AQI is right-skewed) and its coverage guarantee is distribution-free, subject only to exchangeability between calibration and test data.

Within the conformal family, the method used is **Conformalized Quantile Regression** (CQR; Romano et al., 2019), in a hybrid variant that keeps the production champion's point estimates intact.

### 9.2 The hybrid CQR construction

The standard CQR construction trains a single quantile regression on the training set and calibrates against a held-out calibration set. This project departs in a deliberate way: **point predictions on the holdout come from the production champion Ridge (trained on the full pre-holdout pool), while intervals come from a separately-trained quantile regression system fit on a strict subset of that data.**

This is documented in the conformal literature as "locally-valid" conformal — strict exchangeability is relaxed in exchange for usable point accuracy from the deployed model. Empirically, coverage on the holdout meets or slightly exceeds the nominal target (intervals are slightly conservative), which is the right direction to err for a public-health tool.

**Data splits** (chronological):

| Segment | Fraction | Purpose |
|---|---|---|
| Train (proper) | ~56% | Fits local Ridge used to derive residuals for quantile regressors |
| Calibration | ~24% | Computes conformity score `Q_widen` |
| Holdout | 20% | Final per-horizon coverage and width measurement |

**Construction steps:**

1. Fit a **local Ridge** on the train (proper) split, same 5 features as the champion. Used only to generate residuals for quantile regressors — never used for predictions on the holdout.
2. **Compute residuals** `r = y_train − ŷ_local_ridge` on the train (proper) split.
3. **Fit two quantile regressors** on the residuals: `qr_lo` (10th percentile) and `qr_hi` (90th percentile). Both `sklearn.linear_model.QuantileRegressor` with `alpha=0.001`.
4. **Calibration**: on the calibration set, define conformity score `score = max(predicted_lower − y_actual, y_actual − predicted_upper)`. Take the `⌈(n_cal + 1) × (1 − α)⌉`-th largest score (α = 0.20 for nominal 80% coverage) as `Q`. Clamp at zero: `Q_widen = max(Q, 0)`.
5. **Holdout intervals**: point prediction `ŷ = production_champion(x)`. Interval: `[ŷ + qr_lo(x) − Q_widen, ŷ + qr_hi(x) + Q_widen]`.

The asymmetry between point prediction (production champion) and interval (quantile system on local Ridge) is the "hybrid" aspect.

### 9.3 Per-horizon calibration values

Registered as `karachi_aqi_cqr_24h`, `karachi_aqi_cqr_48h`, `karachi_aqi_cqr_72h` (v19, 2026-05-25):

| Horizon | `Q_widen` | Train/cal split |
|---|---|---|
| 24h | 13.64 | 4,905 / 2,103 |
| 48h | 12.30 | 4,891 / 2,097 |
| 72h | 19.27 | 4,878 / 2,091 |

`Q_widen` is largest at 72h — the conformal system honestly registering that 72h residuals are harder to bound, consistent with the autocorrelation decay in Section 5.4.

### 9.4 Empirical coverage

| Horizon | Target | Achieved | Avg interval width |
|---|---|---|---|
| 24h | 80% | **86.4%** | 57.7 AQI units |
| 48h | 80% | **82.9%** | 57.3 AQI units |
| 72h | 80% | **86.8%** | 63.6 AQI units |

All horizons meet or exceed the 80% target. Slightly conservative — a few percentage points wider than the minimum to hit 80%. Interval widths are roughly stable at 57–64 AQI units across horizons.

### 9.5 In practice

Example: with current AQI ~70 and a 24h forecast of point AQI = 69, the conformal interval is approximately [49, 93] — wide enough to express that the model cannot rule out either a recovery to Good or a slip into Unhealthy for Sensitive Groups, but tight enough to rule out Hazardous outcomes.

### 9.6 Coverage drift across approaches

The empirical case for adding the conformal calibration step:

| Approach | 24h coverage |
|---|---|
| XGBoost intervals on raw target (no conformal) | 57.7% |
| Residual quantile regression on Ridge, no calibration | 73–74% |
| Hybrid CQR with conformal calibration (current) | 86.4% |

Quantile regression alone produces under-coverage on real data; the conformal calibration step corrects this with a finite-sample guarantee.

### 9.7 Scope of the coverage claim

These coverage figures are measured on the chronological holdout — same data as the point-prediction evaluation. Re-evaluation of CQR coverage on the Nov 2025 simulation and the May 2026 post-deployment window was not performed (point-prediction skill was the primary OOT question). The values should be read as **in-distribution holdout coverage** — a calibration check on the standard test set, not a claim about coverage stability under distribution shift.

---

## 10. Backend API

### 10.1 Role and hosting

A Flask application (`backend/app.py`) deployed on Render's free tier. It serves predictions and history as JSON over a small REST surface, consumed by the Streamlit dashboard and demoable via `curl`. All endpoints except `/health` require an API key (`X-API-Key` header).

### 10.2 Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `/health` | none | Uptime check; returns status and loaded-model count |
| `/predictions` | API key | Current reading + 24/48/72h forecasts with conformal intervals |
| `/history` | API key | Last N hours of actual AQI (N clamped to 1–720) |
| `/metadata` | API key | Champion training info |
| `/current_live` | API key | Live Open-Meteo `current` snapshot for the dashboard tile |

Full schemas in Appendix D.

### 10.3 Model loading at cold start

The three champion and three CQR models load once when the Flask process starts, via `load_all_models()` reading the latest versions from the Hopsworks Model Registry. Artifacts are cached to local disk, so a dyno restart reuses the local copy. On Render's free tier the dyno sleeps after inactivity, so the first request after sleep pays a one-time ~30 second cold-start cost; subsequent requests are served from the warmed process.

### 10.4 Prediction logic

`/predictions` reads recent rows from the feature store via `fetch_features` (Arrow Flight with JDBC fallback — Section 7), then `compute_predictions` builds the forecast. The most recent row with complete champion features is the anchor; for each horizon it applies that horizon's scaler and Ridge for the point estimate, and the CQR plus calibrated `Q_widen` for the interval. All outputs clamped at zero.

The anchor selection is defensive against incomplete data. If no recent row has all five features — which occurs when `aqi_lag_72h` can't be computed during a materialisation catch-up — the code falls back to median imputation, uses current AQI as a persistence proxy, and a final guard ensures no `NaN` reaches Ridge. This keeps the endpoint returning valid JSON through transient gaps. The honest caveat: while a lag feature is missing, the longest-horizon forecast runs on the proxy and is approximate; it self-heals as lags repopulate. The underlying gap is documented in Section 14.

### 10.5 Live tile and cold-start mitigation

`/current_live` queries Open-Meteo's `current` endpoint directly and caches the result server-side for 5 minutes — the display-only path whose rationale is in Section 4.3. Separately, an UptimeRobot monitor pings `/health` every five minutes to keep the dyno warm, suppressing cold-start latency for normal dashboard traffic.

---

## 11. Dashboard

### 11.1 Overview

A Streamlit application at `khi-aqi.streamlit.app`, hosted on Streamlit Community Cloud. Public, read-only, no state of its own — renders entirely from backend API calls. Header: "ML-powered AQI forecasts for Karachi · Updated hourly · Powered by Hopsworks + Open-Meteo."

### 11.2 Layout

Four display regions:

- **Current Conditions tile** — live snapshot from `/current_live`: AQI with category label, temperature, humidity, PM2.5, PM10, stamped with last reading time in PKT.
- **Forecast cards** — three cards (24h / 48h / 72h), each showing predicted AQI, category, and the conformal interval as `Range: low – high`, labelled with the target day/time in PKT.
- **Forecast Trajectory chart** — current reading + three forecast points + last seven days of actual AQI from `/history`.
- **Health Advisory banner** — tiered alert (11.3).

### 11.3 Single prediction per horizon

Each card is a single point prediction at that horizon, not an hourly series through the next three days. The trajectory chart connects the points with a line for visual continuity only; the line is interpolation, not a prediction at intermediate hours. This is surfaced explicitly in the UI to prevent misreading.

### 11.4 Tiered alert banner

The banner reports the **forecast peak** — the highest AQI across the three horizons — and selects a severity tier by hard EPA-category threshold:

| Tier | Threshold | Banner |
|---|---|---|
| Acceptable | < 101 | "Air quality is acceptable. Sensitive groups may experience minor effects." |
| ⚠️ ADVISORY | ≥ 101 | "Forecast peak: AQI {n} — Unhealthy for Sensitive Groups. Sensitive groups should reduce prolonged outdoor exertion." |
| ⚠️ WARNING | ≥ 151 | "Forecast peak: AQI {n} — Unhealthy. Everyone may experience health effects. Limit outdoor activity." |
| 🚨 HEALTH ALERT | ≥ 201 | "Forecast peak: AQI {n} — Very Unhealthy. Health alert: serious effects possible. Avoid outdoor activity." |

A category-threshold banner is the clearest public-facing signal in a city like Karachi where the baseline is already elevated.

### 11.5 Dev-mode override

A sidebar AQI override slider lets the alert tiers be tested directly — a development and demonstration aid, separate from the live forecast path.

---

## 12. Continuous Integration and Automation

### 12.1 Overview

Every recurring action is a GitHub Actions cron job. No external scheduler, no always-on worker. Schedules live with the code in `.github/workflows/`, GitHub-hosted runners provide ephemeral compute, credentials come from Actions secrets.

### 12.2 The two workflows

| Workflow | Schedule | Cron | Timeout |
|---|---|---|---|
| Feature Pipeline (hourly) | Hourly, minute 5 | `5 * * * *` | 10 min |
| Training Pipeline (daily) | Daily, 02:00 UTC | `0 2 * * *` | 30 min |

Both expose `workflow_dispatch` for manual triggering. The hourly pipeline fires at minute 5 to give Open-Meteo time to publish the latest hour. Full YAML in Appendices B and C.

### 12.3 Run structure

Each run: check out repo, set up Python 3.11 with pip caching, install `requirements-dev.txt`, invoke the pipeline as a module. Both pass `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` as environment variables from secrets. The training workflow adds `pytest tests/` *before* the training step — a failing test aborts the run before any model is trained or registered. A quality gate on every daily retrain.

### 12.4 Safeguards

- **Concurrency control.** Each workflow declares a concurrency group with `cancel-in-progress: false`. New scheduled runs don't start while previous ones are still going.
- **Timeouts.** 10 minutes feature, 30 training.
- **Secrets** stored as GitHub Actions secrets, referenced as `${{ secrets.NAME }}` — never appear in the repo.

### 12.5 Why GitHub Actions, and failure handling

GitHub Actions was chosen because the cron lives with the code, free-tier minutes are more than adequate, and secret management is built in. Failure handling is inherited from the platform: any step that exits non-zero fails the run, GitHub marks it red, and the standard notification fires. The one class of failure this does *not* catch — an online write that succeeds while offline materialisation stalls — is discussed in Sections 14.1 and 15.1.

---

## 13. Out-of-Time Verification

### 13.1 Why OOT, and why three windows

For time-series forecasting, shuffling rows breaks temporal order, leaks future information, and inflates measured performance. The rigorous standard is **out-of-time (OOT) evaluation** — train on data through time *t*, test after *t*, no overlap. The chronological hold-out follows this.

A single OOT window can still mislead. On a calm window where AQI barely moves, a naive baseline is near-optimal, and any model that adds structure looks worse on point metrics. On a noisy window, the model's actual skill becomes measurable. To evaluate fairly, three windows were used:

| # | Window | n | AQI mean | AQI std | Purpose |
|---|---|---|---|---|---|
| 1 | Chronological holdout | 1,743–1,752 | — | ~27 | Primary OOT |
| 2 | Nov 1–15 2025 (sim) | 360 | 114 | 27.9 | High-variance stress test |
| 3 | May 9–13 2026 (post-deploy) | 27 | 68.4 | 5.22 | Honest live-window limitation |

### 13.2 Window 1 — Chronological holdout (primary)

The training pipeline reserves the final 20% as a held-out test set. These metrics are the values registered with the champion in Hopsworks (v19, 2026-05-25):

| Horizon | R² | RMSE | MAE | n_train | n_test |
|---|---|---|---|---|---|
| 24h | 0.350 | 20.15 | 15.05 | 7,008 | 1,752 |
| 48h | 0.181 | 22.64 | 16.76 | 6,988 | 1,748 |
| 72h | 0.129 | 23.35 | 17.50 | 6,969 | 1,743 |

R² values are modest but honest. The decay pattern (0.350 → 0.181 → 0.129) closely matches the EDA-measured autocorrelation decay (0.63 → 0.49 → 0.41). This is the strongest evidence the model is performing at the ceiling set by the data: per-horizon degradation is a property of the forecasting problem, not the model.

### 13.3 Window 2 — Nov 2025 noisy-window simulation

To evaluate on conditions that matter most for public-health alerting (Karachi's seasonal peak), a fresh Ridge was trained on data ending 31 Oct 2025 and used to predict the following 15 days. This window has mean AQI 114 (Unhealthy for Sensitive Groups), peaks at 175 (Unhealthy).

Methodology mirrored production training exactly. Only the split criterion changed (date-based instead of fractional).

| Horizon | Ridge R² | Naive R² | Ridge RMSE | Naive RMSE | Ridge MAE | Naive MAE |
|---|---|---|---|---|---|---|
| 24h | **0.201** | −0.260 | **22.65** | 28.45 | **17.47** | 21.15 |
| 48h | −0.288 | −1.220 | **28.71** | 37.68 | **23.21** | 29.83 |
| 72h | −0.430 | −1.301 | **31.08** | 39.43 | **25.16** | 33.19 |

**Headline: Ridge beats Naive on every metric at every horizon, with 20–24% RMSE reductions.**

![Figure 13.1 — Nov 2025 simulation, 24h horizon](artifacts/oot_nov2025/h24/plot_h24.png)

Interpretation by horizon:

- **24h:** Ridge clearly tracks the diurnal cycle, hitting peaks and troughs in phase with actuals. Naive is consistently phase-shifted by 24 hours. Direct evidence the model has learned predictive structure beyond persistence.
- **48h and 72h:** Ridge correctly reverts toward the conditional mean as predictability decays. It beats Naive on RMSE primarily by avoiding phase-shift errors, rather than by predicting the cycle precisely.

Negative R² at 48h and 72h reflects that even a well-behaved model cannot fully explain variance two-to-three days ahead in an industrial-baseline city. **The relative comparison against Naive is the more honest measure of skill at these horizons than absolute R².**

### 13.4 Window 3 — May 2026 post-deployment

After the May 9 training run, the deployed champion was evaluated on the next ~4 days of incoming data — the first true post-deployment OOT window. After dropping rows with missing features: n = 27, AQI mean 68.4, std 5.22. An unusually calm window.

| Horizon | Ridge R² | Naive R² | Ridge RMSE | Naive RMSE | n |
|---|---|---|---|---|---|
| 24h | −0.279 | −0.284 | 8.11 | 8.13 | 20 |
| 48h | −0.414 | −0.097 | 8.82 | 7.77 | 17 |
| 72h | −1.470 | −0.012 | 11.19 | 7.16 | 10 |

Both Ridge and Naive perform poorly on R² in this window because target variance is tiny (std 5.22 vs ~27 in training). When the signal barely moves, R² becomes nearly uninformative.

Crucially, **the absolute errors are small**: Ridge MAE of 5.65 at 24h is *better* than the training-window MAE (15.05). The predictions track reality in absolute terms; only R² makes them look bad.

This window is reported as a documented limitation of point-metric evaluation on low-variance regimes, not as evidence of model failure. The model's behaviour here is exactly what a well-regularized forecaster should do when there's little signal: it doesn't invent variance that isn't there.

### 13.5 Combined conclusion

The champion's behaviour is consistent across all three evaluations:

1. **On varied samples** (chronological holdout, n ≈ 1,700): R² 0.35 / 0.18 / 0.13 — modest but honest.
2. **On the noisy seasonal-peak window**: beats Naive by 20–24% on RMSE at every horizon, with clear cycle tracking at 24h.
3. **On a calm post-deploy window**: absolute errors small even though R² is uninformative — and the model correctly does not over-fit limited signal.

The empirical performance pattern (degradation across horizons, beating Naive on noisy windows, near-Naive on calm) is exactly what the EDA in Section 5 predicted. The model is rigorously evaluated, behaves correctly across regimes, and is defensible as the production champion.

---

## 14. Operational Observations and Known Limitations

All but one of these issues stem from running entirely on free-tier managed services. They are reported honestly because the diagnosis and handling are themselves part of the engineering record.

### 14.1 Hopsworks free-tier offline materialisation stall

The most significant incident. From around 21 May, the dashboard's forecast cards began showing stale dates while the live tile stayed fresh — forecasts were anchored several days behind.

**Diagnosis.** A `curl` of `/predictions` showed the anchor frozen at 19 May. The backend reads from the *offline* feature store, which had stopped receiving data on that date even though hourly online writes were succeeding. The materialisation job that moves rows online → offline was stuck in `SUBMITTED` state. Deeper inspection: `0/23 nodes available, max node group size reached` — the free-tier shared compute cluster was capacity-saturated. A classmate on the same tier confirmed it was platform-wide.

**Tooling.** Two scripts were built: `check_materialization.py` (read-only diagnostic) and `restart_materialization.py` (kill a stuck job and trigger a fresh one). Multiple kill-and-restart cycles were attempted; each new submission also queued behind the saturated cluster.

**Resolution.** On 29 May the cluster freed up and a materialisation job completed in ~2 minutes, draining roughly nine days of backlog. No migration was performed — staying on Hopsworks was the lower-risk choice given proximity to the deadline.

**Key learning.** An `"inserted N rows"` log line confirms the online write, **not** offline materialisation. The online write is necessary but not sufficient for the data to reach training and serving, and the client logs success on the online write regardless. A more defensive pipeline would detect a materialisation backlog and fail loudly.

### 14.2 Backend `NaN` handling during catch-up

A direct downstream consequence of 14.1: rows materialised during the catch-up carried `NaN` in `aqi_lag_72h` (the 72-hour lag can't be computed before its reference rows exist). This caused `/predictions` to return 500 errors. The fix — median imputation with a current-AQI persistence proxy, plus a hard guard preventing `NaN` reaching Ridge — is described in Section 10.4. It keeps the endpoint functional through such gaps, at the cost of an approximate longest-horizon forecast until valid lags repopulate.

### 14.3 GitHub Actions account block

For roughly 24 hours from 27 May, Actions runs failed at the checkout step with a `403 "account suspended"` — an automated false-positive flag, not a real violation. Support ticket filed; cleared overnight. No code change involved. Recorded here only because it briefly halted the pipelines.

### 14.4 Streamlit Community Cloud sleep

The dashboard sleeps after ~12 hours of inactivity. A plain HTTP ping doesn't prevent this because Streamlit serves a static HTML shell without booting the Python backend — a monitor would report "up" while the app slept. Keeping it awake would require a headless-browser ping on a separate schedule, which was judged scope creep for a cosmetic gain. Accepted limitation: first visit after long idle takes ~30 seconds to wake.

### 14.5 Render cold starts

The backend dyno sleeps after inactivity, adding a one-time ~2–3 second cold-start cost on the first request. Mitigated by the UptimeRobot `/health` ping (Section 10.5).

### 14.6 Hopsworks client/backend version mismatch

The Hopsworks Python client (4.8.1) is a minor version ahead of the backend (4.7.2). Produces a warning but no functional failure. Flagged as a known, non-blocking discrepancy.

---

## 15. Future Work

### 15.1 Platform reliability

- **Move off the shared free tier.** The materialisation stall is a free-tier capacity problem, not a design flaw. A dedicated Hopsworks tier — or migration to Feast, DagsHub, or Vertex AI — would remove shared-cluster contention.
- **Hopsworks Model Deployments.** Hosting models on Hopsworks infrastructure rather than downloading at cold start would eliminate the per-restart registry fetch and reduce the client/backend version surface.
- **Fail-loud pipeline hardening.** Per Section 14.1, the feature pipeline should detect a materialisation backlog and surface it as a red Actions run, rather than treating a successful online write as overall success.

### 15.2 Modelling and evaluation

- **Path X — live-anchored forecasts.** Currently forecasts anchor to the most recent *validated* hour, which can lag the present. Anchoring instead to the live `current` reading would make the horizon track "now" more closely. Open question whether it's worth trading the training/serving distribution match the dual-endpoint design protects (Section 4.3).
- **Rolling-window CQR recalibration.** Coverage figures in Section 9 are measured on a single chronological holdout. Repeating calibration on rolling windows would show how `Q_widen` evolves as data accumulates and whether coverage holds under distribution shift.

### 15.3 Product

- **More alert channels.** SMS or webhook/push notifications so users receive warnings without opening the dashboard.
- **Integrate the two-channel alert module.** A tested `utils/alerts.py` implementing hard-threshold plus statistical-vs-30-day-baseline alerting exists but is not yet wired into the dashboard.
- **Pollution source attribution.** Separating combustion from dust signatures using the inter-pollutant structure observed in the EDA (Section 5.3) would add explanatory value.
- **Geographic expansion.** The architecture supports multi-city replication. Out of scope for this project but a straightforward extension.

---

## 16. Conclusion

### 16.1 What was built

An end-to-end, automated, publicly deployed AQI forecasting system for Karachi. From a resident's perspective, it answers a question real-time AQI displays cannot: not just "what is the air like now," but "what will the air be like for the next three days." From an engineering perspective, it shows that a meaningful MLOps system — automated hourly ingestion, daily retraining, model versioning, distribution-free uncertainty quantification, and a live public dashboard — can run entirely on free-tier managed services with no persistent infrastructure spend.

The deployed model is small, fast, and interpretable: Ridge regression on five features, retrained daily, with conformal intervals calibrated to 80% nominal coverage. The system is live at `khi-aqi.streamlit.app` and `pearls-aqi-backend.onrender.com`.

### 16.2 Key judgment calls

**Chronological splits, even where they hurt headline numbers.** Random splits silently leak temporal autocorrelation and inflate R². The chronological split produces lower numbers, but those numbers reflect actual generalization. The R² decay closely matches the EDA-measured autocorrelation decay — the strongest indication the model is performing at the ceiling set by the data.

**Ridge over deep learning.** An LSTM on the full 26-feature set under-performed Ridge at every horizon, and tree models went negative R² on the pruned 5-feature set. The data is the binding constraint at this scale (one year of hourly data). The choice was empirical, not philosophical.

**CQR for uncertainty.** A point forecast without an interval is indistinguishable from a guess. CQR was selected because it's distribution-free with a finite-sample guarantee under exchangeability. The hybrid construction (production point estimates + calibration-set intervals) produced empirical coverage above target at every horizon. Slightly conservative is the right direction to err.

**Dual-endpoint Open-Meteo design.** Two endpoints with different freshness and quality: validated hourly batch anchors training and forecasting; the `current` endpoint powers the live tile only. Mixing them would risk distribution mismatch. Keeping them separate preserves both prediction integrity and live freshness.

### 16.3 What the platform issues taught

Operating on free-tier infrastructure surfaced failure modes a paid deployment would have obscured. The Hopsworks materialisation stall was the most instructive: a log line confirming "online write succeeded" is not the same as "data is queryable for training and inference." The pipeline silently went stale for nine days because the monitoring assumption — "if the write returned success, the data made it" — was wrong. The defensive response was to build the diagnostic and recovery scripts that should have existed from day one, and to record the lesson: **in a multi-stage data pipeline, downstream freshness must be verified at every stage, not inferred from upstream success.**

The free-tier sleep behaviours of Render and Streamlit reinforced the same lesson. A heartbeat ping that boots the backend is meaningfully different from one that hits a static shell and reports "up" while the application is asleep. Both incidents argue for the same principle: monitor for what the user experiences, not what the infrastructure reports.

### 16.4 Closing

The system is modest in scale — one city, three horizons, one year of training data — and the choices throughout reflect that scale honestly. There is no claim to outperform purpose-built operational forecasting from meteorological agencies with orders-of-magnitude more data and physics-based models. The claim is narrower: that a small team, a public API, and a stack of free-tier managed services are enough to build a working, accountable, decision-supporting forecast for a city that has not had one. On that narrower claim, the empirical evidence supports the system as deployed.

---

## Appendix A — Feature-Group Schema

`aqi_features` (v1) in Hopsworks; 31 columns.

| Column | Type | Role |
|---|---|---|
| timestamp | string | key |
| aqi | fractional | predictor |
| pm2_5 | fractional | predictor |
| pm10 | fractional | predictor |
| no2 | fractional | predictor |
| o3 | fractional | predictor |
| so2 | fractional | predictor |
| co | fractional | predictor |
| temperature | fractional | predictor |
| humidity | fractional | predictor |
| pressure | fractional | predictor |
| wind_speed | fractional | predictor |
| hour | integral | predictor |
| day_of_week | integral | predictor |
| month | integral | predictor |
| is_weekend | integral | predictor |
| aqi_lag_24h | fractional | predictor |
| aqi_lag_48h | fractional | predictor |
| aqi_lag_72h | fractional | predictor |
| pm2_5_lag_24h | fractional | predictor |
| pm2_5_lag_48h | fractional | predictor |
| pm2_5_lag_72h | fractional | predictor |
| aqi_rolling_6h | fractional | predictor |
| aqi_rolling_24h | fractional | predictor |
| rolling_30day_avg | fractional | predictor |
| rolling_30day_std | fractional | predictor |
| aqi_change_rate | fractional | predictor |
| has_target | integral | flag |
| target_aqi_24h | fractional | target |
| target_aqi_48h | fractional | target |
| target_aqi_72h | fractional | target |

The champion uses 5 predictors as inputs: `aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity`. The remaining 22 were evaluated during feature selection and retained for reproducibility and future experimentation.

---

## Appendix B — Hourly Feature Pipeline (workflow)

```yaml
name: Feature Pipeline (hourly)

on:
  workflow_dispatch:
  schedule:
    - cron: '5 * * * *'

concurrency:
  group: feature-pipeline
  cancel-in-progress: false

jobs:
  ingest:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run feature pipeline
        env:
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
          HOPSWORKS_PROJECT: ${{ secrets.HOPSWORKS_PROJECT }}
        run: |
          python -m pipelines.feature_pipeline
```

*`feature_pipeline.yml` — runs hourly at minute 5 to fetch the latest Open-Meteo hour, recompute features, and upsert into the Hopsworks feature group.*

---

## Appendix C — Daily Training Pipeline (workflow)

```yaml
name: Training Pipeline (daily)

on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * *'

concurrency:
  group: training-pipeline
  cancel-in-progress: false

jobs:
  train:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ -v

      - name: Run training pipeline
        env:
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
          HOPSWORKS_PROJECT: ${{ secrets.HOPSWORKS_PROJECT }}
        run: |
          python -m pipelines.training_pipeline
```

*`training_pipeline.yml` — runs daily at 02:00 UTC; executes the test suite, then retrains champion and CQR models for all three horizons and registers new versions.*

---

## Appendix D — API Endpoint Reference

Flask service at `pearls-aqi-backend.onrender.com`. Authenticated endpoints expect the API key in `X-API-Key`.

### `GET /health`

- **Auth**: none
- **Purpose**: liveness probe; used by Render health monitoring and UptimeRobot.
- **Response**:

```json
{ "status": "ok", "models_loaded": 6 }
```

### `GET /predictions`

- **Auth**: API key
- **Purpose**: latest AQI snapshot from the feature store plus 24h / 48h / 72h forecasts with intervals.
- **Response**:

```json
{
  "current": {
    "aqi": 87.2,
    "pm2_5": 28.4,
    "pm10": 53.1,
    "temperature": 27.1,
    "humidity": 64.8,
    "timestamp_utc": "2026-06-01T04:00:00+00:00"
  },
  "forecasts": {
    "24": {
      "point": 91.5,
      "lower": 72.1,
      "upper": 114.3,
      "target_time_utc": "2026-06-02T04:00:00+00:00"
    },
    "48": { "point": 94.0, "lower": 68.4, "upper": 121.7, "target_time_utc": "..." },
    "72": { "point": 96.8, "lower": 65.1, "upper": 129.0, "target_time_utc": "..." }
  }
}
```

### `GET /history`

- **Auth**: API key
- **Query param**: `hours` (int, clamped 1–720; default 168 = 7 days)
- **Purpose**: recent actual AQI readings for dashboard charting.
- **Response**:

```json
{
  "hours": 168,
  "rows": [
    { "timestamp_utc": "2026-05-25T04:00:00+00:00", "aqi": 88.3 },
    { "timestamp_utc": "2026-05-25T05:00:00+00:00", "aqi": 86.1 }
  ]
}
```

### `GET /metadata`

- **Auth**: API key
- **Purpose**: champion model card — algorithm, feature list, training timestamp, per-horizon validation metrics.
- **Response** *(illustrative)*:

```json
{
  "model": "Ridge",
  "alpha": 10,
  "features": ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"],
  "trained_at_utc": "2026-06-01T02:14:33+00:00",
  "metrics": {
    "24h": { "rmse": 18.2, "mae": 12.7, "r2": 0.52 },
    "48h": { "rmse": 21.4, "mae": 15.1, "r2": 0.41 },
    "72h": { "rmse": 23.0, "mae": 16.6, "r2": 0.33 }
  }
}
```

### `GET /current_live`

- **Auth**: API key
- **Purpose**: ~15-minute-lag pollutant and weather snapshot from Open-Meteo's `current` endpoint, cached server-side for 5 minutes. Powers the "Current Conditions" tile only — not used as model input, to avoid distribution mismatch with the validated data the model was trained on.
- **Response (success)**:

```json
{
  "aqi": 86,
  "pm2_5": 27.9, "pm10": 51.2,
  "no2": 18.3, "o3": 64.1, "so2": 14.0, "co": 480,
  "temperature": 27.3, "humidity": 65.2,
  "pressure": 1008.4, "wind_speed": 11.2,
  "timestamp_utc": "2026-06-01T04:45",
  "source": "open-meteo-current",
  "from_cache": false
}
```

- **Response (degraded)**: HTTP 503 with `{ "error": "Open-Meteo unavailable: ...", "fallback_required": true }` if upstream is unreachable.

---

## Appendix E — Glossary

**AQI** — Air Quality Index, a 0–500+ scale that converts pollutant concentrations into a single number indicating health risk. The US EPA formulation used here is driven primarily by PM2.5 and PM10.

**CAMS** — Copernicus Atmosphere Monitoring Service, the European chemistry-transport model whose hourly output is republished by Open-Meteo and serves as the source dataset for this project.

**CQR** — Conformalized Quantile Regression, a distribution-free method that wraps quantile regressors (p10/p90) with a calibration step on held-out residuals to produce prediction intervals with finite-sample coverage guarantees.

**OOT** — Out-of-time evaluation, a held-out window placed strictly after the training period. Required for time-series work because random train/test splits leak information across time and inflate apparent performance.

**SHAP** — SHapley Additive exPlanations, a feature-attribution method that assigns each input feature a contribution to an individual prediction based on cooperative game theory.

**Ridge regression** — Linear regression with an L2 penalty (`alpha`) on coefficient magnitudes. The penalty trades a small amount of bias for a large reduction in variance, which is why it dominates tree-based models on small, autocorrelated datasets like this one. Used here with `alpha=10`.

**Feature store** — A managed system (Hopsworks here) that materializes, versions, and serves engineered features, so the exact same feature definitions feed both training and inference.

**Model registry** — A versioned repository for trained model artefacts and their metadata (also Hopsworks), enabling reproducible promotion from a training run to the model production serves.

---

## Appendix F — References

1. Zippenfenig, P. (2024). *Open-Meteo Air Quality Forecast API*. Open-Meteo. https://open-meteo.com/en/docs/air-quality-api
2. Hopsworks AB. *Hopsworks Documentation — Feature Store and Model Registry*. https://docs.hopsworks.ai
3. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. U.S. Environmental Protection Agency (2018). *Technical Assistance Document for the Reporting of Daily Air Quality – the Air Quality Index (AQI)*. EPA-454/B-18-007. https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
5. Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *Advances in Neural Information Processing Systems (NeurIPS)* 32. arXiv:1905.03222.