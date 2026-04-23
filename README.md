# Pearls AQI Predictor

Karachi AQI prediction and early warning system across 31 administrative subdivisions.

- 3-day AQI forecast per subdivision
- Automated hourly feature pipeline + daily training pipeline
- Streamlit dashboard with heatmap and alerts
- Built on Open-Meteo, Hopsworks, GitHub Actions, Flask, Streamlit

Status: In development.

## Environment

- Python 3.11 (via conda)
- Core: pandas 2.2.x, numpy 1.26.x
- See `requirements.txt` for full pinned versions (generated after Phase 1 completes)

## Setup

```
conda create -n pearls_aqi python=3.11 -y
conda activate pearls_aqi
pip install -r requirements.txt
```
