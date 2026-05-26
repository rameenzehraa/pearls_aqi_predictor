# Section 2 — Problem statement and motivation

## 2.1 Context

Karachi consistently ranks among the most polluted major cities in the world. Pakistan's average PM2.5 concentration in 2025 was 67.3 µg/m³, equivalent to an AQI of approximately 156 (classified as "unhealthy") — nearly 14 times higher than the WHO annual guideline of 5 µg/m³, making Pakistan the most polluted country measured that year. Karachi has repeatedly been ranked among the top five most polluted cities globally in 2025, and the city's 2024 PM2.5 average of 47.1 µg/m³ corresponded to an AQI of 130, 9.4 times the WHO guideline.

The drivers are a mixture of structural and meteorological factors. Traffic emissions, industrial activity, dust from the surrounding region, and seasonal temperature inversions all contribute. The exploratory data analysis (Section 5) confirms what the public health record implies: AQI in Karachi rarely drops to "Good"; the modal hour sits in the "Moderate" range; episodic spikes into "Unhealthy" and "Very Unhealthy" are not rare events.

For residents — particularly children, the elderly, and people with respiratory or cardiovascular conditions — air quality is not an abstract environmental concern. It is a daily decision input: whether to schedule outdoor exercise, whether to send a child to school with a mask, whether to keep windows closed and the HVAC on recirculation. **Real-time AQI reporting answers "what is the air like now." It does not answer "what will the air be like tomorrow, or the day after, or the day after that"** — which is the question that actually shapes behaviour, because human plans are forward-looking, not instantaneous.

## 2.2 The forecasting gap

A real-time AQI reading is a measurement; a forecast is a model output. Pakistan has reasonable real-time air quality monitoring infrastructure — public stations, news media reporting, and several mobile applications that surface current AQI. What it largely lacks is **publicly accessible, multi-day AQI forecasting at the city level** that residents can use to plan ahead.

Closing that gap — even at the modest scale of one city, three horizons (24h / 48h / 72h), with honest uncertainty intervals — has concrete utility:

- **Outdoor activity planning:** schools, sports clubs, outdoor workers can schedule around predicted bad-air windows
- **Health management:** sensitive groups can adjust medication, mask use, and venue choices in advance
- **Indoor preparation:** households can pre-cool indoor air, change HVAC filters, prepare purifiers before an episode rather than after

A forecasting system that combines real-time inference, automated retraining, and a public dashboard is also a non-trivial engineering exercise — which is the project's secondary purpose, beyond the public utility argument.

## 2.3 Project scope

This project builds a public, hourly-updating, 3-day-ahead AQI forecasting system for Karachi. The system is end-to-end automated:

- **Data ingestion** runs hourly from public APIs (Open-Meteo air quality and weather endpoints) into a feature store
- **Model training** runs daily, retraining the production champion against the latest data
- **Inference** runs on demand when a user opens the dashboard, against the live feature store
- **Storage** of trained models is handled through a model registry that auto-versions every successful training run

There is no manual intervention in the production loop. Data lands hourly, models are retrained daily, and predictions update accordingly — all driven by scheduled cron workflows.

### 2.3.1 What the system predicts

The single most important point of clarity, because it is easy to misread: the system predicts **one AQI value per forecast horizon, anchored to the present moment**. It does *not* produce an hourly trajectory through the next 72 hours.

Specifically, at any inference time *t*:

- The **+24h** forecast is a point prediction of AQI at *t + 24 hours*, with a conformal prediction interval
- The **+48h** forecast is a point prediction of AQI at *t + 48 hours*, with a conformal prediction interval
- The **+72h** forecast is a point prediction of AQI at *t + 72 hours*, with a conformal prediction interval

Three predictions total per inference call, one per horizon. This framing is consistent throughout the dashboard (three forecast cards corresponding to the three horizons), through the model architecture (three separate Ridge models, one per horizon, see Section 8), and through the conformal interval calibration (three separate `Q_widen` values, one per horizon, see Section 9). The horizon-decay analysis in Section 5.9 — and the corresponding R² degradation across horizons in Section 8.7 — is meaningful precisely because each horizon is a separate forecasting task with its own intrinsic difficulty.

The dashboard's Forecast Trajectory chart visualises these three predictions plus the current AQI as four connected points, which can give the impression of a continuous forecast trajectory. Strictly, the line connecting them is interpolation for visual continuity, not a separate prediction at intermediate hours.

### 2.3.2 What the system does not predict

To be explicit about scope:

- **No pollution source attribution.** The model does not separate "this AQI is high because of dust" from "this AQI is high because of vehicular emissions."
- **No multi-pollutant separate forecasts.** AQI is a composite index; the project forecasts the index, not its individual pollutant components.
- **No sub-city geographic resolution.** Karachi is forecast at a single representative location (the city centroid). Spatial heterogeneity within Karachi is not modelled — early scoping work (Day 2 of the project, noted in the working log) determined that the Open-Meteo CAMS data resolution does not support meaningfully distinct sub-city forecasts.
- **No intra-day forecast updates within the dashboard's current architecture.** The "Current Conditions" tile updates with each page load (sub-15-minute freshness via Open-Meteo's current endpoint), but the +24/+48/+72h forecasts are anchored to the most recent hour in the feature store, which updates as the daily data pipeline materialises new rows.
- **No multi-city expansion.** The system is single-city by design. The architecture supports multi-city replication, but instantiating a second city was not in scope.

### 2.3.3 Engineering scope (project-level requirements)

Beyond the forecasting task itself, the project specification required demonstrating several MLOps practices:

- Data fetched from a third-party API (Open-Meteo air quality and weather endpoints)
- Features computed and stored in a managed feature store (Hopsworks)
- Models stored in a managed model registry (Hopsworks Model Registry)
- Hourly feature pipeline and daily training pipeline, both automated via cron (GitHub Actions)
- Multiple model families compared (Ridge, Random Forest, XGBoost, LSTM — see Section 8)
- Feature importance analysis (SHAP, see Section 8.4)
- Web dashboard for end users (Streamlit, see Section 11)
- HTTP API for prediction serving (Flask, see Section 10)
- Alerts for hazardous AQI levels (tiered banner system, see Section 11)
- Out-of-time evaluation (see Section 13)

The system meets all of the above. The architecture chosen is fully serverless: GitHub Actions for compute, Hopsworks for state, Render for the backend API, Streamlit Community Cloud for the dashboard. No persistently-running infrastructure is provisioned or paid for — every component runs on free-tier managed services.

## 2.4 Audience and intended use

The dashboard at `khi-aqi.streamlit.app` is publicly accessible. The intended audience is two-fold:

1. **General Karachi residents** who want a glanceable forecast and current air quality reading
2. **Project evaluators and engineers** who want to inspect the system architecture, model methodology, and code

For (1), the dashboard emphasises tiered alerts, plain-language AQI category labels, and PKT-local timestamps. For (2), the system exposes a public REST API (with key-based authentication), a fully version-controlled codebase, and a separately-published GitHub repository.

This report documents the design decisions, empirical evidence, and operational observations that produced the deployed system.