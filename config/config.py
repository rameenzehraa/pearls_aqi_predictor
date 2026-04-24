"""
Central configuration for Pearls AQI Predictor.

Single source of truth for all constants. Every pipeline imports from here.
Do NOT hardcode values in other files — add them here and import.
"""

# ========================================================================
# Location — Karachi centroid
# ========================================================================
# City-wide forecast. Spatial diagnostic (see notes.md Day 2) showed that
# CAMS grid resolution collapses 31 subdivisions into ~3 distinct signals,
# so per-subdivision forecasting is not supported by the data. Legacy
# subdivision list preserved at the bottom of this file for audit trail.

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# ========================================================================
# Rolling windows (hours)
# ========================================================================
ROLLING_WINDOW_SHORT    = 6
ROLLING_WINDOW_LONG     = 24
ROLLING_WINDOW_BASELINE = 720   # 30 days

# ========================================================================
# Lag features (hours)
# ========================================================================
LAG_HOURS = [24, 48, 72]

# ========================================================================
# Alert thresholds
# ========================================================================
# Statistical alert (unusual spike relative to recent baseline)
ALERT_STD_MULTIPLIER = 1.5
ALERT_STD_FLOOR      = 20
ALERT_SPIKE_RATIO    = 1.30
ALERT_MINIMUM_JUMP   = 30

# Hard thresholds — fire regardless of baseline.
# Karachi's baseline is often elevated; a purely statistical alert could
# miss genuinely dangerous air that's "normal" for the month. These catch
# the absolute danger levels defined by US EPA.
ALERT_HARD_THRESHOLDS = [151, 301]   # Unhealthy, Hazardous

# ========================================================================
# Model training
# ========================================================================
TRAIN_TEST_SPLIT      = 0.8   # final holdout fraction
CV_N_SPLITS           = 5     # walk-forward CV folds within training pool
XGBOOST_QUANTILE_LOW  = 0.10
XGBOOST_QUANTILE_HIGH = 0.90

# ========================================================================
# LSTM
# ========================================================================
LSTM_LOOKBACK_HOURS = 48
LSTM_HIDDEN_UNITS   = 64
LSTM_EPOCHS         = 30
LSTM_BATCH_SIZE     = 64

# ========================================================================
# Hyperparameter tuning (XGBoost only)
# ========================================================================
HPT_N_ITER   = 20
HPT_CV_FOLDS = 3

# ========================================================================
# Multi-model dashboard
# ========================================================================
NUM_COMPARISON_MODELS = 3   # champion + 2 others shown side by side

# ========================================================================
# API / Pipeline
# ========================================================================
API_RETRY_COUNT      = 3
API_RETRY_BASE_DELAY = 5   # seconds, exponential backoff base

# ========================================================================
# Drift detection
# ========================================================================
DRIFT_Z_SCORE_THRESHOLD = 3

# ========================================================================
# Streamlit
# ========================================================================
CACHE_TTL_SECONDS = 300   # 5 minutes

# ========================================================================
# Backend API
# ========================================================================
# Overridden via environment variable in Streamlit Cloud secrets.
BACKEND_API_URL = "https://pearls-aqi-api.onrender.com"

# ========================================================================
# Backfill
# ========================================================================
BACKFILL_START_DATE = "2025-04-25"   # 1 year back from project start

# ========================================================================
# AQI categories (US EPA standard)
# Range (inclusive low, inclusive high) → (category_id, label)
# ========================================================================
AQI_CATEGORIES = {
    (0,   50):  (1, "Good"),
    (51,  100): (2, "Moderate"),
    (101, 150): (3, "Unhealthy for Sensitive Groups"),
    (151, 200): (4, "Unhealthy"),
    (201, 300): (5, "Very Unhealthy"),
    (301, 500): (6, "Hazardous"),
}

# ========================================================================
# Timezone
# ========================================================================
TIMEZONE_STORAGE = "UTC"
TIMEZONE_DISPLAY = "Asia/Karachi"


# ========================================================================
# LEGACY — preserved for audit trail only. Do NOT import.
# ========================================================================
# Initial plan called for per-subdivision forecasts across Karachi's 31
# official administrative subdivisions. Spatial diagnostic on Open-Meteo's
# CAMS global pollution data (April 2026) showed:
#   - 31 coordinates resolve to 11 nominal grid cells
#   - Those 11 cells produce only ~3 meaningfully distinct signals
#   - Mean off-diagonal correlation across cells: 0.97
# Scope refined to a single city-wide forecast. See notes.md Day 2 for the
# full decision trail and diagnostic scripts in pipelines/.
#
# _SUBDIVISIONS_LEGACY = {
#     # District Central (5)
#     "Gulberg":           (24.9200, 67.0700),
#     "Liaquatabad":       (24.9050, 67.0400),
#     "Nazimabad":         (24.9120, 67.0330),
#     "New Karachi":       (24.999886, 67.064827),
#     "New Nazimabad":     (24.957439, 67.043921),
#     # District East (4)
#     "Ferozabad":         (24.872202, 67.066136),
#     "Gulshan-e-Iqbal":   (24.9200, 67.0850),
#     "Gulzar-e-Hijri":    (24.9400, 67.1500),
#     "Jamshed Quarter":   (24.883527, 67.042922),
#     # Karachi South (5)
#     "Aram Bagh":         (24.856334, 67.013423),
#     "Civil Lines":       (24.8500, 67.0230),
#     "Garden":            (24.872915, 67.014370),
#     "Lyari":             (24.8700, 66.9950),
#     "Saddar":            (24.8600, 67.0300),
#     # Karachi West (3)
#     "Orangi":            (24.949446, 67.004710),
#     "Mominabad":         (24.929671, 66.991663),
#     "Manghopir":         (24.998720, 67.018685),
#     # Keamari (4)
#     "Baldia":            (24.928667, 66.960371),
#     "Harbour":           (24.8400, 66.9800),
#     "Mauripur":          (24.870536, 66.956652),
#     "SITE":              (24.909595, 67.006064),
#     # Korangi (4)
#     "Korangi":           (24.8400, 67.1500),
#     "Landhi":            (24.8500, 67.2100),
#     "Model Colony":      (24.904835, 67.188046),
#     "Shah Faisal":       (24.8900, 67.1400),
#     # Malir (6)
#     "Airport":           (24.9000, 67.1600),
#     "Bin Qasim":         (24.7900, 67.3500),
#     "Gadap":             (25.0500, 67.1700),
#     "Ibrahim Hyderi":    (24.794110, 67.145783),
#     "Murad Memon":       (24.924854, 67.248761),
#     "Shah Murad":        (25.001321, 67.190570),
# }