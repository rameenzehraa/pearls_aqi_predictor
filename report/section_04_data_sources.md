# Section 4 — Data sources and ingestion

## 4.1 Data source — Open-Meteo

The system uses Open-Meteo's free, no-authentication-required public APIs as the sole external data source. Two distinct endpoints are queried:

- **Air quality endpoint** — provides hourly concentrations of six pollutants: PM2.5, PM10, NO₂, O₃, SO₂, CO. Backed by the Copernicus Atmosphere Monitoring Service (CAMS) global atmospheric composition model.
- **Weather forecast endpoint** — provides hourly meteorological variables: temperature at 2m, relative humidity at 2m, wind speed at 10m, surface pressure. Backed by ECMWF-derived weather models.

Both endpoints are queried for the same Karachi coordinate (centroid: 24.8607°N, 67.0011°E) and merged on UTC timestamp to form the unified hourly observation row that flows into the feature pipeline. Open-Meteo was selected after an empirical evaluation of alternatives in the early phase of the project (documented further in Section 4.5).

## 4.2 Why a single representative coordinate, not a city grid

An initial design considered fetching pollution data for 31 separate coordinates corresponding to administrative subdivisions of Karachi. A diagnostic on Open-Meteo's CAMS grid resolution showed that the 31 subdivision coordinates resolved to **11 nominal grid cells**, of which only **~3 produced meaningfully distinct signals** — the mean off-diagonal correlation across grid cells exceeded 0.97. The CAMS resolution does not support sub-city forecast differentiation at the scale of Karachi's administrative boundaries.

Producing 31 separate forecasts from data that supports ~3 distinct signals would be presentation theater, not real information. The scope was refined to a single city-wide forecast at the centroid coordinate. This is consistent with the project specification ("predict the AQI in your city") and matches the data's actual spatial resolution.

## 4.3 The dual-endpoint architecture

A single endpoint is queried for *training* and feature-store ingestion; a different endpoint is queried for the *live* "Current Conditions" tile on the dashboard. This is a deliberate design decision, not a redundancy.

### 4.3.1 The hourly historical endpoint (training and forecast anchor)

The air-quality `forecast` API (queried with `past_days=1` to retrieve the most recent 24 hours) returns Open-Meteo's CAMS-validated hourly data. This is the **system of record** — every row in the feature group originates here. The training pipeline reads from the feature group, and the live inference path also reads from it: the most recent feature-group row becomes the *anchor* from which the +24h / +48h / +72h forecasts are computed.

Using validated CAMS data ensures that the model's training distribution and serving distribution are the same. If the training pipeline fits on validated data while the inference path uses real-time, semi-processed data, the model would be evaluated on one signal type and asked to predict on another — a classic distribution mismatch that silently degrades production performance.

### 4.3.2 The `current` endpoint (dashboard live tile only)

Open-Meteo also exposes a `current` parameter on the air-quality and weather endpoints that returns the most recent available reading without the validated batch step. This data has a much shorter publishing lag (typically ~15 minutes vs. the validated endpoint's 6–24 hours) but is less rigorously processed.

The dashboard's "Current Conditions" tile pulls from this endpoint for display purposes only. The values shown — current AQI, temperature, humidity, PM2.5, PM10 — give the user a freshness-cued sense of present air quality without affecting the forecast computation. This tile is rendered through the backend's `/current_live` endpoint, with a 5-minute server-side cache to avoid hammering Open-Meteo on every dashboard refresh.

### 4.3.3 Why separate the two

The split protects the system from distribution mismatch while still giving the user a "now" reading that doesn't lag by a quarter-day. Specifically:

- **Forecasts** (+24h, +48h, +72h cards) anchor to the most recent *validated* feature-store row. They reflect the same data the model was trained on.
- **The "Current Conditions" tile** displays the most recent *real-time* CAMS reading, refreshed every ~15 minutes by Open-Meteo upstream and cached for 5 minutes downstream.

The two are visibly different freshness levels on the dashboard — the live tile shows a recent reading while the forecast cards may be anchored to a few hours earlier — and that asymmetry is intentional, not a bug. A user looking at the dashboard at 6 PM gets the actual current air quality alongside forecasts grounded in the validated historical signal.

## 4.4 CAMS publishing cadence

The validated CAMS air-quality data does not publish in real time. Investigation of the publishing pattern (Day 13–16 of the project) confirmed: **CAMS publishes the previous day's 24 hours in a single batch around 00:00 UTC each day.**

The practical consequence: of the 24 hourly cron runs in any 24-hour period, only the run shortly after 00:00 UTC actually finds new data to ingest. The other 23 runs query Open-Meteo, find that the latest available hour is already in the feature group, and exit with "0 new rows to insert." This is documented in pipeline logs and is the upstream API characteristic, not a defect in the ingestion code.

The hourly cron schedule was kept rather than reduced to daily-only for two reasons:

1. **Resilience.** If the 00:00 UTC publish is delayed by Open-Meteo's upstream chain, subsequent hourly runs catch up automatically when the data does land. A single daily cron that ran at, say, 02:00 UTC would miss the data if publishing slipped to 03:00 UTC and would have to wait 24 hours for the next attempt.

2. **Operational simplicity.** Hourly schedules are easier to reason about than carefully-timed daily ones, and the cost of empty runs is negligible (~10 seconds each, against a generous GitHub Actions minute budget).

The 0-row runs are reported as `INFO` log messages, not warnings or errors. GitHub Actions reports them as green (successful) because they completed without exception — the absence of new data is the expected steady state, not a failure.

## 4.5 Alternative data sources evaluated

Before settling on Open-Meteo as the sole source, two alternatives were investigated:

**AQICN** (`aqicn.org`) was tested as a potential real-time data source to address Open-Meteo's batch publishing lag. The free-tier AQICN API exposes a "Karachi" search endpoint that returns the US Consulate ground station — which has been offline since March 2025 and reports no current data. While AQICN's public website displays four functioning Karachi ground stations (University of Karachi, Aga Khan, Urban Resource Center, G3 Engineering), these stations are not exposed through the free API. The conclusion: AQICN free-tier provides nothing useful for Karachi at this time.

**OpenWeather** (`openweathermap.org`) was reviewed but not tested in depth. Its air pollution API covers PM2.5, PM10, NO₂, O₃, SO₂, CO, and NH₃ — overlapping with Open-Meteo's coverage but at a coarser resolution. Open-Meteo's CAMS-backed data was selected over OpenWeather for two reasons: it includes a validated batch with explicit publishing semantics (rather than a single rolling endpoint), and it does not require API key registration or rate-limit management for the project's volume.

The selection of Open-Meteo was therefore not "we used the most convenient option" — it was the empirically validated choice after considering the available alternatives.

## 4.6 Karachi coordinate

The single coordinate used throughout: **24.8607°N, 67.0011°E**. This is approximately Karachi's geographic centroid. All Open-Meteo queries pass `latitude=24.8607, longitude=67.0011` and `timezone=UTC`. Timestamp handling stores everything in UTC throughout the pipeline; the dashboard converts to Pakistan Standard Time (UTC+5, no daylight saving) at the presentation layer.

## 4.7 Summary

Data flows from Open-Meteo's CAMS-backed air-quality endpoint and ECMWF-backed weather endpoint into the feature group, hourly, via an automated GitHub Actions cron. A separate live-tile endpoint provides freshness for the dashboard's "Current Conditions" panel without contaminating the training-serving distribution match. The CAMS publishing cadence (daily batch around 00:00 UTC) defines the system's effective data refresh rhythm, with the hourly cron acting as a resilience layer that picks up the validated batch whenever it lands.

The dual-endpoint architecture, the single-coordinate scope decision, and the choice of Open-Meteo over alternatives are all empirically grounded design choices rather than defaults.