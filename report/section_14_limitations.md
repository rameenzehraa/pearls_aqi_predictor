# Section 14 — Operational observations and known limitations

This section documents the operational issues encountered in production. All but one stem from running entirely on free-tier managed services; they are reported honestly because the diagnosis and handling are themselves part of the engineering record.

## 14.1 Hopsworks free-tier offline materialisation stall

The most significant incident. From around 21 May, the dashboard's forecast cards began showing stale dates while the live "Current Conditions" tile stayed fresh — the forecasts were anchored several days behind.

**Diagnosis.** A `curl` of `/predictions` showed the anchor row's timestamp frozen at 19 May. The backend reads from the *offline* feature store, and the offline store had stopped receiving data on that date even though the hourly online writes were still succeeding. The materialisation job that moves rows from the online to the offline store was stuck in a `SUBMITTED` state. Deeper inspection of the Hopsworks job monitoring showed the underlying cause: `0/23 nodes available, max node group size reached` — the free-tier compute cluster, which is shared across users, was capacity-saturated and could not schedule the job. A classmate on the same tier confirmed it was a platform-wide free-tier issue, not specific to this project.

**Tooling.** Two scripts were built to manage the incident: `check_materialization.py` (a read-only diagnostic reporting job state and the offline store's latest timestamp) and `restart_materialization.py` (kills a stuck job and triggers a fresh one). Multiple kill-and-restart cycles were attempted; each new submission also queued behind the saturated cluster.

**Resolution.** On 29 May the cluster freed up and a materialisation job completed in about two minutes, draining roughly nine days of backlog and catching the offline store up to the current hour. No migration was performed — staying on Hopsworks was the lower-risk choice given proximity to the deadline (alternatives are discussed in Section 15).

**Key learning.** An `"inserted N rows"` log line confirms the online write, not offline materialisation. The online write is necessary but not sufficient for the data to reach training and serving, and the client logs success on the online write regardless of whether materialisation is keeping up. A more defensive pipeline would detect a materialisation backlog and fail loudly.

## 14.2 Backend `NaN` handling during catch-up

A direct downstream consequence of 14.1: rows materialised during the catch-up carried `NaN` in `aqi_lag_72h` (the 72-hour lag could not be computed before its reference rows existed), and every recent row was affected. This caused `/predictions` to return 500 errors, since Ridge rejects `NaN` natively. The fix — median imputation with a current-AQI persistence proxy and a hard guard that prevents any `NaN` reaching the model — is described in Section 10.4. It keeps the endpoint functional through such gaps, at the cost of an approximate longest-horizon forecast until valid lags repopulate.

## 14.3 GitHub Actions account block

For roughly 24 hours from 27 May, GitHub Actions runs failed at the checkout step with a `403 "account suspended"` error — an automated false-positive flag on the account rather than any real violation. A support ticket was filed; the block cleared on its own overnight and Actions resumed normally. No code or configuration change was involved. It is recorded here only because it briefly halted the automated pipelines.

## 14.4 Streamlit Community Cloud sleep

The dashboard is hosted on Streamlit Community Cloud's free tier, which sleeps an app after about 12 hours of inactivity. A plain HTTP ping does not prevent this, because Streamlit serves a static HTML shell without booting the Python backend — a monitor would report the app "up" while it slept. Keeping it awake would require a headless-browser ping on a separate schedule, which was judged scope creep for a cosmetic gain. The accepted limitation: the first visit after a long idle period takes roughly 30 seconds to wake the app.

## 14.5 Render cold starts

The backend's Render free-tier dyno sleeps after inactivity, adding a one-time ~2–3 second cold-start cost on the first request. This is mitigated rather than eliminated, by the UptimeRobot `/health` ping described in Section 10.5.

## 14.6 Hopsworks client/backend version mismatch

The Hopsworks Python client (4.8.1) is a minor version ahead of the backend it connects to (4.7.2). This produces a version-mismatch warning but no functional failure, and is flagged here as a known, non-blocking discrepancy.