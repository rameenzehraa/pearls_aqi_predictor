# Pearls AQI Predictor

End-to-end MLOps system that forecasts Karachi's Air Quality Index (AQI) up to 72 hours ahead, with hourly automated data ingestion, daily model retraining, and a live dashboard with prediction intervals and hazard alerts.

## Live URLs

- **Dashboard:** https://khi-aqi.streamlit.app
- **Backend API:** https://pearls-aqi-backend.onrender.com
- **Model Registry:** Hopsworks (6 models — 3 Ridge champions + 3 CQR quantile regressors)

## Features

- 24h / 48h / 72h AQI forecasts with conformal prediction intervals (p10/p90)
- Hourly feature pipeline (GitHub Actions → Open-Meteo → Hopsworks Feature Store)
- Daily training pipeline (Ridge, Random Forest, XGBoost, and LSTM evaluated; Ridge selected as champion across all horizons)
- SHAP-based feature selection (5 features selected from 26 candidates via SHAP + ablation)
- Hazard alerts: tiered banner on the dashboard (advisory/unhealthy/hazardous) plus a tested dual-channel alert utility (hard threshold + statistical spike detection) ready for integration
- Streamlit frontend + Flask backend (model serving from Hopsworks Model Registry)

## Architecture

```
Open-Meteo API
     ↓ (hourly)
GitHub Actions feature_pipeline.yml
     ↓
Hopsworks Feature Store (aqi_features v1)
     ↓ (daily)
GitHub Actions training_pipeline.yml
     ↓
Hopsworks Model Registry (Ridge × 3 + CQR × 3)
     ↓
Flask backend (Render) ←→ Streamlit frontend (Streamlit Cloud)
```

## Tech Stack

Python 3.11, scikit-learn, TensorFlow (LSTM), XGBoost, SHAP, Hopsworks, GitHub Actions, Flask, Streamlit, Plotly, Pandas, NumPy.

## Setup (Local Development)

```bash
conda create -n pearls_aqi python=3.11 -y
conda activate pearls_aqi
pip install -r requirements.txt
```

Create a `.env` file with:
```
HOPSWORKS_API_KEY=<your_key>
HOPSWORKS_PROJECT=<your_project>
BACKEND_API_KEY=<shared_secret>
```

## Project Structure

```
pipelines/           # Feature & training pipelines (run on GitHub Actions)
models/              # Training scripts (Ridge, RF, XGBoost, LSTM, CQR, SHAP)
backend/             # Flask API serving predictions from Hopsworks Registry
frontend/            # Streamlit dashboard
utils/               # Shared helpers (alerts, metrics, CV)
notebooks/           # EDA notebook
tests/               # Unit tests (78 tests across AQI calc, features, alerts)
.github/workflows/   # Hourly feature + daily training automation
```

## Tests

```bash
pytest tests/ -v
```

## Status

Deployed and fully operational.
