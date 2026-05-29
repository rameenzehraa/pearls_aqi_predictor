# Section 3 — System architecture

## 3.1 Overview

This section is the map for the rest of the report: it shows how the components fit together and which later section documents each one in detail. The system is a fully serverless, end-to-end automated forecasting pipeline. Data flows in one direction — from a third-party API, through an automated feature pipeline into a managed feature store, into a daily training loop that publishes versioned models to a registry, and out through a prediction API to a public dashboard. No component runs on infrastructure the project operates or pays for; every piece is either a free-tier managed service or an ephemeral scheduled job.

![Figure 3.1 — System architecture](../artifacts/architecture_diagram.png)

**Figure 3.1.** System architecture. Solid arrows trace the validated-data and model-production path; dashed arrows trace the live-display and keep-warm paths.

The remainder of the report drills into each component: data sources (Section 4), feature engineering and the feature pipeline (Sections 6–7), model training and conformal intervals (Sections 8–9), the backend API (Section 10), the dashboard (Section 11), and the automation that drives all of it (Section 12).

## 3.2 Component responsibilities

| Component | Role | Hosted on |
|---|---|---|
| Open-Meteo API | External data source — CAMS air quality + ECMWF weather | Third-party (no auth) |
| Feature pipeline (`pipelines/feature_pipeline.py`) | Hourly ingest: fetch, clean, engineer, insert | GitHub Actions (cron) |
| Training pipeline (`pipelines/training_pipeline.py`) | Daily retrain, calibrate intervals, register models | GitHub Actions (cron) |
| Hopsworks Feature Store | System of record for features (`aqi_features` v1) | Hopsworks (free tier) |
| Hopsworks Model Registry | Versioned champion + CQR artifacts | Hopsworks (free tier) |
| Flask backend (`backend/app.py`) | Prediction / serving REST API | Render (free tier) |
| Streamlit dashboard (`frontend/app.py`) | Public end-user UI | Streamlit Community Cloud |
| UptimeRobot | Health pings to suppress Render cold starts | Third-party (free tier) |

Each component has a single clear responsibility, and durable state lives only in Hopsworks — neither the backend nor the dashboard holds persistent state of its own. That is what makes the serving tier disposable: either service can be restarted or redeployed at any time with no data loss, because it reloads everything it needs from Hopsworks on startup.

## 3.3 Data flow

Three independent flows run on different cadences.

**Ingest (hourly).** A GitHub Actions cron triggers the feature pipeline, which fetches the most recent validated hourly readings from Open-Meteo's CAMS air-quality and ECMWF weather endpoints, merges them on UTC timestamp, runs cleaning and feature engineering, and inserts deduplicated new rows into the Hopsworks feature group. Because CAMS publishes its validated data once daily (Section 4.4), most hourly runs find no new rows and exit cleanly — the hourly schedule is a resilience layer that catches the daily batch whenever it lands, not a sign that new data arrives every hour. Full detail in Section 7.

**Train (daily).** A second GitHub Actions cron triggers the training pipeline, which reads the current feature group, fits one Ridge champion per horizon (Section 8), fits the conformal quantile-regression intervals on top of them (Section 9), and registers all six artifacts — three champion plus three CQR — to the Hopsworks Model Registry with auto-incrementing versions. The loop is fully unattended; a failure in any step exits non-zero and surfaces as a red GitHub Actions run (Section 12).

**Serve (on demand).** When a user opens the dashboard, Streamlit calls the Flask backend. On a cold start the backend loads the latest champion and CQR models from the Model Registry and caches them to disk (Section 10). The `/predictions` endpoint reads the most recent validated row from the feature store, uses it as the forecast anchor, and computes the +24h / +48h / +72h point forecasts and conformal intervals. The dashboard renders these as forecast cards, a trajectory chart, and tiered alert banners (Section 11). Separately, the "Current Conditions" tile calls `/current_live`, which queries Open-Meteo's live `current` endpoint directly and never touches the feature store.

## 3.4 The dual-endpoint design

The architecture deliberately reads from two different Open-Meteo endpoints for two different purposes, visible as the two inbound paths to the backend in Figure 3.1. The validated hourly batch is the system of record: it feeds the feature store, the training data, and the forecast anchor, so the model is always trained and served on the same signal. The live `current` endpoint feeds only the display tile, giving the user a sub-15-minute "now" reading without contaminating the forecast path. Keeping these separate is what prevents a training/serving distribution mismatch; the full rationale and the freshness trade-off are documented in Section 4.3.

## 3.5 Automation and the serverless claim

Every recurring action in the system is a scheduled GitHub Actions job — there is no always-on scheduler, worker, or server that the project maintains. Compute is ephemeral: GitHub-hosted runners exist only for the duration of a cron run, and the Render dyno wakes on request rather than running continuously. All durable state lives in Hopsworks. The backend's only standing concern is cold-start latency, handled by a free UptimeRobot monitor pinging `/health`. The result is a system that runs continuously in production with zero provisioned infrastructure and zero recurring spend — the "serverless" claim is literal, not aspirational. The CI/CD wiring that makes this work is documented in Section 12.