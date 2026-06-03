# Appendices

---

## Appendix A — Feature-Group Schema

`aqi_features` (v1) in Hopsworks; 31 columns. Role definitions:

- **key** — primary key / event time
- **target** — prediction label (one per horizon)
- **flag** — row-level metadata
- **predictor** — candidate input feature (raw or engineered)

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

The champion model uses 5 of the 27 predictors as inputs: `aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity`. The remaining 22 predictors were evaluated during feature selection and retained in the feature group for reproducibility and future experimentation.

---

## Appendix B — Hourly Feature Pipeline

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

*GitHub Actions workflow `feature_pipeline.yml` — runs hourly at minute 5 to fetch the latest Open-Meteo hour, recompute features, and upsert into the Hopsworks feature group.*

---

## Appendix C — Daily Training Pipeline

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

*GitHub Actions workflow `training_pipeline.yml` — runs daily at 02:00 UTC; executes the test suite, then retrains champion and CQR models for all three horizons and registers new versions in the Hopsworks model registry.*

---

## Appendix D — API Endpoint Reference

Flask service deployed on Render at `pearls-aqi-backend.onrender.com`. All authenticated endpoints expect the API key in the `X-API-Key` request header.

### `GET /health`

- **Auth**: none
- **Purpose**: liveness probe; used by Render health monitoring and UptimeRobot to prevent cold starts.
- **Response**:

```json
{ "status": "ok", "models_loaded": 6 }
```

### `GET /predictions`

- **Auth**: API key
- **Purpose**: latest AQI snapshot from the feature store plus 24h / 48h / 72h forecasts with prediction intervals.
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
- **Query parameter**: `hours` (int, clamped 1–720; default 168 = 7 days)
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
- **Purpose**: champion model card — algorithm, feature list, training timestamp, and per-horizon validation metrics.
- **Response** *(illustrative; actual values produced at training time)*:

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
- **Purpose**: ~15-minute-lag pollutant and weather snapshot from Open-Meteo's `current` endpoint, cached server-side for 5 minutes. Powers the dashboard's "Current Conditions" tile only — not used as model input, to avoid distribution mismatch with the validated hourly data the model was trained on.
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

- **Response (degraded)**: HTTP 503 with `{ "error": "Open-Meteo unavailable: ...", "fallback_required": true }` if the upstream is unreachable.

---

## Appendix E — Glossary

**AQI** — Air Quality Index, a 0–500+ scale that converts pollutant concentrations into a single number indicating health risk. The US EPA formulation used here is driven primarily by PM2.5 and PM10.

**CAMS** — Copernicus Atmosphere Monitoring Service, the European chemistry-transport model whose hourly output is republished by Open-Meteo and serves as the source dataset for this project.

**CQR** — Conformalized Quantile Regression, a distribution-free method that wraps quantile regressors (e.g. p10/p90) with a calibration step on held-out residuals to produce prediction intervals with finite-sample coverage guarantees.

**OOT** — Out-of-time evaluation, a held-out window placed strictly after the training period. Required for time-series work because random train/test splits leak information across time and inflate apparent performance.

**SHAP** — SHapley Additive exPlanations, a feature-attribution method that assigns each input feature a contribution to an individual prediction based on cooperative game theory.

**Ridge regression** — Linear regression with an L2 penalty (`alpha`) on coefficient magnitudes. The penalty trades a small amount of bias for a large reduction in variance, which is why it dominates tree-based models on small, autocorrelated datasets like this one. Used here with `alpha=10`.

**Feature store** — A managed system (Hopsworks, in this project) that materializes, versions, and serves engineered features, so the exact same feature definitions feed both training and inference.

**Model registry** — A versioned repository for trained model artefacts and their metadata (also Hopsworks here), enabling reproducible promotion from a training run to the model that production serves.

---

## Appendix F — References

1. Zippenfenig, P. (2024). *Open-Meteo Air Quality Forecast API*. Open-Meteo. https://open-meteo.com/en/docs/air-quality-api
2. Hopsworks AB. *Hopsworks Documentation — Feature Store and Model Registry*. https://docs.hopsworks.ai
3. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. U.S. Environmental Protection Agency (2018). *Technical Assistance Document for the Reporting of Daily Air Quality – the Air Quality Index (AQI)*. EPA-454/B-18-007. https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
5. Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *Advances in Neural Information Processing Systems (NeurIPS)* 32. arXiv:1905.03222.