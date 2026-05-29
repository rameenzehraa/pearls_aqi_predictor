# Section 7 — Feature pipeline

## 7.1 Role and trigger

The feature pipeline keeps the feature group current by appending newly-published validated rows. It runs as an hourly GitHub Actions cron that invokes `pipelines/feature_pipeline.py`. The pipeline reads from and writes to Hopsworks only — there is no local CSV or parquet in the production path. This was a hard rule fixed early in the project: the training pipeline and the live backend must both read from the same feature store, so the feature store has to be the single source of truth, populated exclusively through this pipeline.

## 7.2 Step-by-step flow

Each run executes four steps:

1. **Fetch** the most recent window from Open-Meteo — roughly the last 6 hours of air-quality and weather data — and merge the two endpoints on UTC timestamp.
2. **Pull recent history** from the feature group — roughly the last 80 hours of stored rows.
3. **Clean and engineer** over the combined frame using the same `clean_raw_data` and `engineer_features` functions as the backfill (Section 6), so live rows and historical rows are produced identically.
4. **Insert** the deduplicated new rows into the feature group.

## 7.3 Why pull ~80 hours of history

The engineered features include lags up to 72 hours and rolling windows up to 24 hours. To compute those features correctly for a *new* row, the pipeline needs the preceding rows in context — the 72-hour lag is the binding constraint. Fetching only the handful of genuinely new hours from Open-Meteo would leave the lag and rolling features undefined, because the lookback rows wouldn't be present in the frame being engineered. Pulling ~80 hours of stored history (72h plus a small buffer) guarantees every new row has the context it needs.

This dependency is also why incomplete history shows up downstream as `NaN` lag features: if the preceding rows are missing — for example during a materialisation catch-up — the longest lags cannot be computed and arrive null. The backend handles that case defensively at inference time (Section 10); the pipeline's job is simply to provide as much valid context as the store contains.

## 7.4 Idempotency and the steady state

Because the feature group's primary key is `timestamp`, re-processing an overlapping window is safe: rows whose timestamps already exist are no-ops, and only genuinely new timestamps add data. The pipeline can therefore run every hour without risk of duplication or corruption.

In normal operation, most hourly runs insert **zero** new rows. The validated CAMS data publishes once daily (Section 4.4), so only the run shortly after the daily batch lands actually finds new hours to append; the other runs fetch, find nothing newer than what they already have, and exit. These empty runs are logged at `INFO` level and reported by GitHub Actions as green — the absence of new data is the expected steady state, not a failure. The hourly cadence functions as a resilience layer that catches the daily batch whenever upstream publishing lands it.

## 7.5 Operational note — Arrow Flight read with JDBC fallback

Hopsworks feature-group reads default to Arrow Flight, a fast columnar transport. On the free tier the Flight / Query Service connection occasionally drops mid-read, surfacing as `Flight`, `Socket closed`, or `Query Service` errors. The pipeline (and the backend's reader, Section 10) catches these and retries the read over the slower Hive/JDBC path (`read_options={"use_hive": True}`). This trades read speed for reliability whenever Flight is flaky, and keeps an otherwise-transient platform hiccup from failing the run.

## 7.6 Write semantics

An insert writes to the online store immediately, after which the row materialises to the offline store that training and the backend read from. On the free tier this materialisation is asynchronous and normally completes quickly, but it is a distinct step from the online write — the "inserted N rows" log line confirms the online write, not offline materialisation. A multi-day materialisation stall encountered during the project, and the diagnostic tooling built to monitor it, are documented in Section 14.