# Section 1 — Executive summary

## 1.1 What was built

This project delivers **Pearls AQI Predictor**, a publicly accessible system that forecasts Karachi's Air Quality Index at three horizons — 24, 48, and 72 hours ahead — alongside a live current-conditions reading. The forecasts are produced by an end-to-end automated pipeline: data is ingested hourly from a public meteorological API, models are retrained daily on the latest data, and predictions update on the dashboard with each page load. No manual intervention is required to keep the system running.

The intended audience is two-fold. For residents of Karachi, the dashboard provides a glanceable forecast with plain-language AQI categories, tiered health alerts for hazardous readings, and PKT-local timestamps. For evaluators and engineers, the system exposes a versioned codebase, a documented REST API, and a full MLOps stack assembled entirely on free-tier managed services.

## 1.2 Architecture

The system is fully serverless. Every component runs on a free-tier managed service; no persistently provisioned infrastructure is required.

| Layer | Service |
|---|---|
| Data source | Open-Meteo air quality and weather APIs |
| Feature store and model registry | Hopsworks |
| Scheduled compute | GitHub Actions (hourly ingestion, daily training) |
| Prediction API | Flask on Render |
| Dashboard | Streamlit Community Cloud |

Data flows in one direction: Open-Meteo → Hopsworks feature store → daily training pipeline → Hopsworks model registry → Flask backend → Streamlit dashboard. The production model is a small regularized linear regression (Ridge) on five features, selected through systematic comparison against tree-based and recurrent-neural-network alternatives.

## 1.3 Headline outcomes

On the chronological hold-out — the final 20% of the time-ordered dataset, never seen during training (n ≈ 1,750) — the deployed model achieves:

| Horizon | R² | RMSE | MAE |
|---|---|---|---|
| 24h | 0.350 | 20.15 | 15.05 |
| 48h | 0.181 | 22.64 | 16.76 |
| 72h | 0.129 | 23.35 | 17.50 |

These figures are modest but defensible. The R² decay across horizons (0.35 → 0.18 → 0.13) closely matches the underlying autocorrelation decay in Karachi's AQI signal (0.63 → 0.49 → 0.41), indicating the model is operating at the ceiling set by the data rather than under-fitting.

On a high-variance stress test — a 15-day simulation across Karachi's November 2025 seasonal pollution peak — the model beats a naive "tomorrow equals today" baseline by **20–24% on RMSE at every horizon**, with the 24h forecast clearly tracking the diurnal cycle in phase with the actual signal.

Every point forecast is accompanied by an 80%-target conformal prediction interval. Empirical coverage on the hold-out is 86.4% / 82.9% / 86.8% across the three horizons — slightly conservative, which is the appropriate direction to err for a public-health-relevant tool.

The dashboard is live at `khi-aqi.streamlit.app`; the prediction API is live at `pearls-aqi-backend.onrender.com`.

## 1.4 Honest limitations

Two limitations stem directly from running entirely on free-tier infrastructure. The Hopsworks offline materialisation stalled for nine days in late May because the shared free-tier compute cluster was capacity-saturated, freezing the dashboard's forecast anchor before catching up on its own; the diagnostic and recovery work is documented in Section 14. The Streamlit dashboard and Render backend both sleep after inactivity on their free tiers, producing a one-time ~30-second wake-up delay on the first visit after a long idle period — accepted as a cosmetic cost of zero infrastructure spend.