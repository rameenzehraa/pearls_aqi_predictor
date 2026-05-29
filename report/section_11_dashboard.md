# Section 11 — Dashboard

## 11.1 Overview

The end-user interface is a Streamlit application at `khi-aqi.streamlit.app`, hosted on Streamlit Community Cloud. It is public and read-only: it holds no state of its own and renders entirely from calls to the backend API (Section 10). The header frames the system in one line — "ML-powered AQI forecasts for Karachi · Updated hourly · Powered by Hopsworks + Open-Meteo."

## 11.2 Layout

The dashboard has four display regions:

- **Current Conditions tile** — the live snapshot from `/current_live`: current AQI with its category label, temperature, humidity, PM2.5, and PM10, stamped with the last reading time in PKT. This is the fresh "now" reading (Section 4.3).
- **Forecast cards** — three cards for the +24h / +48h / +72h horizons, each showing the predicted AQI, its category, and the conformal interval as a `Range: low – high`, labelled with the target day and time in PKT.
- **Forecast Trajectory chart** — the current reading plus the three forecast points plotted together, alongside the last seven days of actual AQI from `/history`.
- **Health Advisory banner** — the tiered alert described in 11.3.

## 11.3 Single prediction per horizon

Consistent with Section 2.3.1, each forecast card is a single point prediction at that horizon, not an hourly series through the next three days. The trajectory chart connects the current reading and the three forecast points with a line for visual continuity only; the line is interpolation, not a prediction at intermediate hours. This framing is surfaced explicitly in the UI to prevent the cards being misread as an hour-by-hour forecast.

## 11.4 Tiered alert banner

The advisory banner reports the **forecast peak** — the highest AQI across the three horizons — and selects a severity tier by hard AQI threshold. Hard thresholds (rather than a purely statistical spike detector) are used because Karachi's baseline air quality is often already elevated; an alert keyed only to deviation from baseline could miss air that is objectively dangerous. The tiers and their exact wording:

| Tier | Threshold | Banner |
|---|---|---|
| Acceptable | < 101 | "Air quality is acceptable. Sensitive groups may experience minor effects." |
| ⚠️ ADVISORY | ≥ 101 | "Forecast peak: AQI {n} — Unhealthy for Sensitive Groups. Sensitive groups should reduce prolonged outdoor exertion." |
| ⚠️ WARNING | ≥ 151 | "Forecast peak: AQI {n} — Unhealthy. Everyone may experience health effects. Limit outdoor activity." |
| 🚨 HEALTH ALERT | ≥ 201 | "Forecast peak: AQI {n} — Very Unhealthy. Health alert: serious effects possible. Avoid outdoor activity." |

Each banner names the triggering AQI value and its category, so the user sees both the severity and the number behind it.

## 11.5 Dev-mode override

A sidebar AQI override slider lets the alert tiers be tested directly: setting an override value drives the banner through ADVISORY, WARNING, and HEALTH ALERT without waiting for live conditions to cross a threshold. It is a development and demonstration aid, separate from the live forecast path.