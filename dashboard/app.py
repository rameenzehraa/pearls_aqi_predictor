"""
PEARLS AQI Predictor — Streamlit Dashboard.

Live air quality forecasts for Karachi using an ML pipeline backed by
Hopsworks Feature Store and Model Registry.

Run locally:
    streamlit run dashboard/app.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make project-root imports work when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.utils import (
    HORIZONS,
    aqi_to_category,
    fetch_current_conditions,
    fetch_recent_features,
    format_pkt,
    load_champion_metadata,
    load_champion_models,
    load_cqr_models,
    make_predictions,
    to_pkt,
)


# ========================================================================
# Page config
# ========================================================================

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ========================================================================
# Cached loaders (avoid re-loading models / re-querying Hopsworks every
# rerun; Streamlit reruns on every interaction by design)
# ========================================================================

@st.cache_resource
def get_models():
    return load_champion_models(), load_cqr_models()


@st.cache_resource
def get_metadata():
    return load_champion_metadata()


@st.cache_data(ttl=600)  # refresh from Hopsworks every 10 min
def get_features():
    return fetch_recent_features(hours=168)


# ========================================================================
# Header + refresh button
# ========================================================================

header_col, refresh_col = st.columns([5, 1])

with header_col:
    st.title("🌫️ Karachi Air Quality Predictor")
    st.caption(
        "ML-powered AQI forecasts for Karachi · Updated hourly · "
        "Powered by Hopsworks + Open-Meteo"
    )

with refresh_col:
    st.write("")  # spacer for vertical alignment
    if st.button("🔄 Refresh", help="Force re-fetch from Hopsworks"):
        get_features.clear()
        st.rerun()

# Loading spinner while we hit Hopsworks
load_msg = (
    "Connecting to Hopsworks Feature Store and loading models… "
    "(first load can take ~60 seconds while data syncs)"
)
with st.spinner(load_msg):
    champion, cqr = get_models()
    features = get_features()
    metadata = get_metadata()

if features.empty:
    st.error("Could not load feature data from Hopsworks. Try again later.")
    st.stop()

predictions = make_predictions(features, champion, cqr)
latest = fetch_current_conditions()
current_aqi = latest["aqi"]
current_cat = aqi_to_category(current_aqi)

# ========================================================================
# Data freshness warning
# ========================================================================

# now_utc = pd.Timestamp.now(tz="UTC")
# data_age_hours = (now_utc - latest["timestamp"]).total_seconds() / 3600

# if data_age_hours > 3:
#     hours_str = f"{int(data_age_hours)}h" if data_age_hours < 48 else f"{int(data_age_hours / 24)}d"
#     st.warning(
#         f"⚠️ **Data is {hours_str} old.** "
#         f"The most recent reading is from {format_pkt(latest['timestamp'])}. "
#         f"This usually means Open-Meteo hasn't published fresh data yet, or "
#         f"the hourly ingestion pipeline is catching up. Forecasts below are "
#         f"computed from this last available reading."
#     )


# ========================================================================
# Section 1: Current AQI snapshot
# ========================================================================

st.markdown("### Current Conditions")

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.markdown(
        f"""
        <div style="
            background: {current_cat['color']}22;
            border-left: 6px solid {current_cat['color']};
            padding: 20px;
            border-radius: 8px;
        ">
            <div style="font-size: 14px; opacity: 0.7;">CURRENT AQI</div>
            <div style="font-size: 56px; font-weight: 700; color: {current_cat['color']};">
                {current_cat['value']}
            </div>
            <div style="font-size: 18px; font-weight: 600;">{current_cat['name']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.metric("Temperature", f"{latest['temperature']:.1f} °C")
    st.metric("Humidity", f"{latest['humidity']:.0f}%")

with col3:
    st.metric("PM2.5", f"{latest['pm2_5']:.1f} µg/m³")
    st.metric("PM10", f"{latest['pm10']:.1f} µg/m³")

st.caption(f"Last reading: {format_pkt(latest['timestamp'])}")


# ========================================================================
# Section 2: Forecast cards
# ========================================================================

st.markdown("### Forecast")

cols = st.columns(3)
for col, h in zip(cols, HORIZONS):
    pred = predictions[h]
    cat = aqi_to_category(pred["point"])
    target_str = format_pkt(pred["target_time"], fmt="%a · %H:%M")
    with col:
        st.markdown(
            f"""
            <div style="
                background: {cat['color']}15;
                border: 1px solid {cat['color']}55;
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 12px; opacity: 0.7;">+{h}H · {target_str}</div>
                <div style="font-size: 40px; font-weight: 700; color: {cat['color']};">
                    {int(round(pred['point']))}
                </div>
                <div style="font-size: 14px; font-weight: 600;">{cat['name']}</div>
                <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
                    Range: {int(round(pred['lower']))} – {int(round(pred['upper']))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ========================================================================
# Section 3: Forecast chart with confidence bands
# ========================================================================

st.markdown("### Forecast Trajectory")

# Build a small dataframe of forecast points (timestamps in PKT for display)
forecast_df = pd.DataFrame([
    {"time": to_pkt(latest["timestamp"]), "point": current_aqi,
     "lower": current_aqi, "upper": current_aqi},
] + [
    {"time": to_pkt(predictions[h]["target_time"]),
     "point": predictions[h]["point"],
     "lower": predictions[h]["lower"],
     "upper": predictions[h]["upper"]}
    for h in HORIZONS
])

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=forecast_df["time"], y=forecast_df["upper"],
    mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
))
fig_forecast.add_trace(go.Scatter(
    x=forecast_df["time"], y=forecast_df["lower"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(100, 149, 237, 0.2)",
    name="80% confidence",
))
fig_forecast.add_trace(go.Scatter(
    x=forecast_df["time"], y=forecast_df["point"],
    mode="lines+markers",
    line=dict(color="#4a90e2", width=3),
    marker=dict(size=10),
    name="Predicted AQI",
))

fig_forecast.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Time (PKT)",
    yaxis_title="AQI",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_forecast, use_container_width=True)


# ========================================================================
# Section 4: Last 7 days historical
# ========================================================================

st.markdown("### Last 7 Days")

# Convert timestamps to PKT for display
features_display = features.copy()
features_display["time_pkt"] = features_display["timestamp"].apply(to_pkt)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=features_display["time_pkt"],
    y=features_display["aqi"],
    mode="lines",
    line=dict(color="#888", width=2),
    name="Actual AQI",
))

# Add colored category bands
for low, high, color in [
    (0, 50, "rgba(0,228,0,0.08)"),
    (51, 100, "rgba(255,255,0,0.08)"),
    (101, 150, "rgba(255,126,0,0.08)"),
    (151, 200, "rgba(255,0,0,0.08)"),
    (201, 300, "rgba(143,63,151,0.08)"),
]:
    fig_hist.add_hrect(y0=low, y1=high, fillcolor=color, line_width=0)

fig_hist.update_layout(
    height=350,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Time (PKT)",
    yaxis_title="AQI",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_hist, use_container_width=True)


# ========================================================================
# Section 5: Health advisory
# ========================================================================

worst = max(predictions.values(), key=lambda p: p["point"])
worst_cat = aqi_to_category(worst["point"])

st.markdown("### Health Advisory")
st.markdown(
    f"""
    <div style="
        background: {worst_cat['color']}15;
        border-left: 4px solid {worst_cat['color']};
        padding: 14px 18px;
        border-radius: 6px;
    ">
        <div style="font-weight: 600; color: {worst_cat['color']};">
            Peak forecast: {worst_cat['name']} (AQI {int(round(worst['point']))})
        </div>
        <div style="margin-top: 6px; font-size: 14px;">
            {worst_cat['advice']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================================================================
# Section 6: Footer / model info
# ========================================================================

with st.expander("ℹ️ About this dashboard"):
    st.markdown(f"""
    **Model:** Ridge Regression with 5 features, selected via SHAP analysis +
    feature ablation study (reduced from 26 candidate features).

    **Training metrics (R² / RMSE / MAE) on chronological holdout:**
    - 24h: {metadata['horizons'][0]['metrics']['r2']:.3f} · {metadata['horizons'][0]['metrics']['rmse']:.1f} · {metadata['horizons'][0]['metrics']['mae']:.1f}
    - 48h: {metadata['horizons'][1]['metrics']['r2']:.3f} · {metadata['horizons'][1]['metrics']['rmse']:.1f} · {metadata['horizons'][1]['metrics']['mae']:.1f}
    - 72h: {metadata['horizons'][2]['metrics']['r2']:.3f} · {metadata['horizons'][2]['metrics']['rmse']:.1f} · {metadata['horizons'][2]['metrics']['mae']:.1f}

    **Confidence intervals:** Hybrid Conformalized Quantile Regression
    (Romano et al. 2019), calibrated to ~80% empirical coverage on holdout.

    **MLOps pipeline:**
    - **Hourly feature pipeline** — pulls fresh data from Open-Meteo,
      engineers features, writes to Hopsworks Feature Store. Runs on
      GitHub Actions every hour.
    - **Daily training pipeline** — validates feature store access and
      data quality. Runs on GitHub Actions daily at 02:00 UTC.
    - **Model Registry** — 6 models versioned in Hopsworks
      (3 Ridge champions + 3 CQR interval systems).

    **Data source:** Open-Meteo Air Quality + Weather APIs.

    **Last training:** {metadata['trained_at_utc'][:10]} ·
    **Repo:** [github.com/rameenzehraa/pearls_aqi_predictor](https://github.com/rameenzehraa/pearls_aqi_predictor)
    """)