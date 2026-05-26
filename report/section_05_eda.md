# Section 5 — Exploratory Data Analysis

## 5.1 Purpose and scope

The full EDA is documented in `notebooks/eda.ipynb`, executed against the production feature group. This section distils the analysis into the findings that materially shaped the modelling decisions in Section 8 (model training) and Section 9 (conformal intervals). The format throughout: each subsection identifies the question being asked, presents the empirical result, and states the design choice that followed.

The dataset analysed comprises **9,264 hourly rows from 25 April 2025 to 19 May 2026** (~13 months), pulled directly from the `aqi_features` feature group in Hopsworks. The data has 31 columns: raw pollutants (PM2.5, PM10, NO₂, O₃, SO₂, CO), weather variables (temperature, humidity, wind speed, pressure), engineered time features (hour, day of week, month, is_weekend), engineered statistical features (rolling means over 6h/24h/30d windows, lag features at 24/48/72h offsets), and forecast targets at three horizons.

## 5.2 Data quality and continuity

**Question:** Is the time series clean and continuous enough to support hourly forecasting and lag-feature engineering?

**Findings:**

- 9,264 hourly rows spanning ~13 months
- No duplicate timestamps
- A single 4-day gap (25 April 2026 23:00 UTC → 30 April 2026 00:00 UTC), accounting for 96 missing hours (1.04% of the time range)
- Null values restricted to lag/target columns at the expected boundary positions (the first 72 hours have no 72-hour lag value; the last 72 hours have no 72-hour-ahead target)

The 4-day gap is contiguous (one block, not scattered missing hours), occurs well after the 72-hour lag-feature warmup completes, and falls outside any train/validation/test split boundary. It does not affect modelling or evaluation.

**Decision:** The dataset is treated as a continuous hourly time series. No interpolation or imputation is applied at the EDA stage; missing lag values are dropped during training (handled by `dropna` on the per-horizon target column).

## 5.3 Target distribution

**Question:** What is the shape of the AQI distribution? Does it suggest transformations, alert thresholds, or class-imbalance considerations?

**Findings:**

- AQI is right-skewed with mean 93.2, median 85.8, standard deviation 27.3
- Range: 23.9 minimum, 389.7 maximum
- 75th percentile at 106 — most hours sit below the "Unhealthy for Sensitive Groups" boundary (151)
- A long upper tail of extreme outliers indicates episodic pollution events (dust storms, smog episodes, winter inversion peaks)

Distribution across EPA AQI categories:

| Category | Hours | % of total |
|---|---|---|
| Good (0–50) | 47 | 0.51% |
| Moderate (51–100) | 6,547 | 70.7% |
| Unhealthy for Sensitive Groups (101–150) | 2,084 | 22.5% |
| Unhealthy (151–200) | 533 | 5.8% |
| Very Unhealthy (201–300) | 52 | 0.56% |
| Hazardous (301+) | 1 | 0.01% |

**Decisions:**

1. **No log transformation.** The right-skew is moderate (not multiple-orders-of-magnitude), and downstream Ridge regression with the SHAP-pruned feature set handles the distribution adequately without transformation. The 24h/48h/72h targets share the same distributional shape, so a transform would have to be applied consistently — adding complexity for no measured benefit.

2. **Alert-based UX, not constant warnings.** Hazardous events occurring once in 9,264 hours (~0.01%) justify a tiered alert system that fires only on elevated AQI rather than displaying persistent warnings. This design decision is implemented in the dashboard's tiered banner system (ADVISORY at 101+, WARNING at 151+, HEALTH ALERT at 201+).

3. **Class imbalance noted.** Moderate hours dominate (70.7%). Model evaluation is conducted with continuous metrics (R², RMSE, MAE) rather than category-level accuracy, since the regression target is the AQI value itself, not the EPA category.

## 5.4 Temporal patterns

**Question:** Which calendar-based features carry signal? Hourly, daily, weekly, or monthly cycles?

**Findings:**

- **Diurnal cycle is present and strong.** AQI peaks around 09:00 UTC (14:00 PKT — early afternoon) and dips overnight. Amplitude ~15 AQI units around the daily mean.
- **No weekly cycle.** AQI averaged by day of week is essentially flat. This is the most consequential finding in this subsection.
- **Strong monthly/seasonal cycle.** Winter peak around November (mean ~118 AQI), monsoon trough around September (mean ~78). The amplitude (~40 AQI units month-over-month) is much larger than the diurnal amplitude.

The absence of a weekly cycle is informative about the *source* of Karachi's air pollution. A commuter-driven city would show weekend dips (less traffic, less industrial activity). Karachi's flat weekly pattern suggests pollution is dominated by **baseline industrial and dust sources** that operate continuously, not by Monday-to-Friday commuter behaviour.

**Decisions:**

1. **`month` feature retained**, weighted heavily by the model (see Section 8 coefficient analysis).
2. **`hour` feature engineered** (`hour_sin`, `hour_cos` in some explorations), but ultimately dropped by SHAP pruning — the diurnal cycle is largely already encoded by recent-lag features (`aqi_lag_24h` captures yesterday's same-hour AQI).
3. **`day_of_week` and `is_weekend` dropped** by the SHAP-based feature selection in Section 8. Their independent contribution was near zero.

## 5.5 Pollutant distributions

**Question:** How are the individual pollutant concentrations distributed? Does any pollutant suggest transformation or special treatment?

**Findings:**

- **PM2.5, PM10, NO₂, SO₂, CO are all right-skewed** with long tails — typical of pollution data where most hours are moderate but episodic spikes occur.
- **O₃ shows a bimodal distribution** with peaks near 50 µg/m³ and 125 µg/m³ — a signature of day/night photochemistry. Ozone forms in sunlight via NOx + VOC reactions, peaks in the afternoon, and is scavenged by NO at night.
- **CO has the most extreme range** (mean ~536, max ~4,000 µg/m³), suggesting traffic/combustion spike events.

**Decision:** No log transformation applied to pollutant features. The bimodal O₃ distribution would be distorted by a transform, and the subsequent SHAP analysis (Section 8.4) and ablation (Section 8.5) demonstrated that raw values combined with feature selection produced better R² than alternatives. The decision was to keep raw values and let SHAP pruning remove features that didn't earn their place.

## 5.6 Weather distributions

**Question:** Do the weather variables carry enough variation to be predictive?

**Findings:**

- **Temperature**: range 10–40°C, mean 26.5°C. Reflects Karachi's seasonal swing.
- **Humidity**: left-skewed, concentrated 70–90%. Coastal city dominated by humid days, but the spread is meaningful.
- **Wind speed**: near-normal around 10.7 m/s. Consistent Arabian Sea breeze patterns.
- **Pressure**: nearly uniform in a tight 995–1025 hPa band. Typical for sea-level coastal locations.

**Decision:** Pressure's narrow band predicted weak discriminative power, which the correlation analysis (Section 5.7) and SHAP analysis (Section 8.4) confirmed. Pressure was kept in the initial 26-feature candidate set but dropped by SHAP-based selection. Humidity and wind speed were retained based on their measurable correlation with AQI (next subsection).

## 5.7 Feature correlation with AQI

**Question:** Which features have the strongest linear relationship with the target? This is a first-pass filter on the candidate feature set.

**Findings (Pearson correlation with current AQI):**

| Feature | r |
|---|---|
| PM2.5 | **+0.88** |
| aqi_rolling_6h | **+0.88** |
| PM10 | +0.74 |
| aqi_rolling_24h | +0.73 |
| aqi_lag_24h | +0.62 |
| aqi_lag_48h | +0.51 |
| aqi_lag_72h | +0.45 |
| humidity | **−0.40** |
| wind_speed | **−0.32** |
| NO₂ | +0.30 |
| O₃ | +0.19 |
| month | +0.18 |
| day_of_week | ≈0 |
| is_weekend | ≈0 |

Key observations:

- **PM2.5 dominates** — consistent with the US AQI formula, which is a PM2.5-driven index for the typical concentration ranges seen in Karachi.
- **Rolling and lag features show strong correlations** — `aqi_rolling_6h` matches PM2.5 at r = 0.88, confirming AQI has strong temporal autocorrelation. Recent history is highly predictive of current values, which is exactly the property a lag-feature model exploits.
- **Humidity (−0.40) and wind speed (−0.32) are the strongest weather signals**, both inversely related to AQI. Wet weather and wind disperse particulates.
- **Day-of-week and is_weekend (~0)** corroborate the temporal-pattern finding in Section 5.4: no weekly cycle to exploit.

**Decision:** This shortlist — PM2.5, lag/rolling AQI features, humidity, with secondary signal from wind, NO₂, and PM10 — directly informed the candidate feature set passed to the SHAP-based selection in Section 8. The final 5-feature champion set (`aqi`, `month`, `aqi_lag_24h`, `aqi_lag_72h`, `humidity`) is a subset of features that appeared meaningfully here.

## 5.8 Inter-pollutant correlations and Karachi's pollution profile

**Question:** Beyond each pollutant's relationship with AQI, what do the relationships *between* pollutants reveal about the underlying physical sources?

**Findings (Pearson correlation between pollutant pairs):**

- **NO₂ ↔ CO: +0.72** — both are traffic and combustion byproducts; they move together.
- **NO₂ ↔ O₃: −0.56** — classic photochemical anticorrelation. NO scavenges O₃ at night; sunlight converts NO₂ to O₃ during the day.
- **PM10 ↔ NO₂: ≈0** and **PM10 ↔ CO: ≈0** — PM10 is essentially independent of combustion pollutants in Karachi.
- **O₃ ↔ AQI: +0.19** (weak) — Karachi's AQI is rarely ozone-driven.

The PM10-independent-of-combustion finding is the most informative: it suggests Karachi's coarse particulate matter (PM10) is dominated by **dust and aerosol sources** (Arabian Sea coastal aerosol, Thar desert dust, construction dust) rather than combustion. PM2.5 (the finer fraction) is more aligned with combustion sources, but the US AQI calculation weights PM2.5 strongly enough that even Karachi's dust-driven PM10 doesn't dominate the AQI index.

**Decisions:**

1. **High NO₂↔CO correlation flagged potential feature redundancy.** Keeping both adds little incremental signal. SHAP-based selection (Section 8.4) ultimately dropped CO, retaining the strictly more predictive features.

2. **Confirmed PM2.5 vs PM10 capture different physical processes.** Both could have been retained if SHAP favoured them; in practice the champion relied on AQI lag features, which already encode the particulate history implicitly.

3. **No special handling of O₃ in the model.** Its weak AQI correlation made it unattractive for inclusion despite the interesting photochemistry.

## 5.9 Forecast horizon decay — the predictability ceiling

**Question:** How much information about future AQI is recoverable from the present at each horizon (24h, 48h, 72h)?

**Findings:**

Correlation between current AQI and AQI at horizon *h*:

| Horizon | Pearson r |
|---|---|
| +24h | 0.63 |
| +48h | 0.49 |
| +72h | 0.41 |

Target distributions are nearly identical across horizons (mean ~93.4, std ~27.2). There is **no distributional shift** — only a decay in the temporal autocorrelation between present and future.

This decay is a **property of the data**, not a model failure. It places an upper bound on the R² achievable by any model at each horizon. A model can only exploit signal that exists; once present-to-future correlation drops to 0.41, no algorithm can produce miraculous 72-hour forecasts.

**Decisions (the most consequential decisions in the project):**

1. **One model per horizon.** A single model fitted across all three horizons would either be dominated by the easiest (24h) and underperform on 72h, or be regularized into the conditional mean and underperform on all three. Three separate Ridge models, each fitted to its own horizon, was the chosen design.

2. **Conformal prediction intervals widen with horizon.** Residual uncertainty grows with horizon by definition — if the model can predict less variance, the unpredicted residuals are larger. The conformal interval system (Section 9) is calibrated per-horizon so intervals at 72h are wider than at 24h, honestly reflecting the larger uncertainty.

3. **The observed model R² values** (0.350 / 0.181 / 0.129 at 24h/48h/72h, per Section 8.7) **follow the data-property prediction.** The empirical R² ratio mirrors the autocorrelation ratio, providing strong evidence that the champion model is operating at the ceiling set by the data, not below it.

## 5.10 Top feature scatter visualization — does correlation imply usability?

**Question:** Correlation is a single number. Are the underlying relationships actually linear enough that a linear model can exploit them?

**Findings:**

- **PM2.5 vs AQI**: nearly linear, slope ≈ 1.9. This confirms the US AQI formula's PM2.5-dominance and indicates a linear model can capture the relationship without polynomial expansion.
- **aqi_rolling_24h vs AQI**: the tightest scatter among engineered features. Recent average AQI is a strong baseline predictor with low residual variance.
- **Humidity vs AQI**: clear negative trend with wide scatter. Useful directional signal, not deterministic on its own.
- **Wind speed vs AQI**: similar — clear negative slope, wide spread.
- **CO vs AQI**: a long tail of high-CO outliers maps to moderate AQI values, indicating CO spikes don't always translate to AQI spikes. PM2.5 must also be elevated for AQI to rise.

**Decisions:**

1. **Linear model is viable.** The dominantly linear PM2.5↔AQI shape was direct evidence in favour of Ridge regression over kernel methods, polynomial features, or tree-based non-linear models. This is corroborated empirically in Section 8.3, where tree models (Random Forest, XGBoost) went negative R² on the pruned 5-feature set.

2. **No single weather feature can carry the model.** The wide scatter on humidity and wind required combining them with lag features rather than building weather-only baselines.

3. **CO dropped by SHAP analysis.** High CO values aren't reliable AQI signal — the scatter confirmed what the SHAP-based selection later concluded.

## 5.11 Findings to decisions — summary

The EDA produced findings that converged on a set of design choices for the rest of the project. The mapping is intentionally explicit:

| EDA Finding | Modelling Decision |
|---|---|
| Continuous hourly time series with one 4-day gap | Treat as continuous; no interpolation; gap doesn't cross train/test boundaries |
| Right-skewed AQI with rare Hazardous events (~0.01%) | Alert-based dashboard UX (tiered banners), not constant warnings |
| 70.7% Moderate baseline | Regression on continuous AQI value, not classification by category |
| Strong diurnal cycle | `hour` engineered; later dropped by SHAP (redundant with lag features) |
| No weekly cycle | `day_of_week` and `is_weekend` dropped by SHAP |
| Strong monthly seasonality | `month` retained; ends up as a top-5 feature |
| PM2.5 dominates AQI (r = 0.88) | Confirms linear-model viability; raw current AQI features chosen as strongest predictor |
| Strong autocorrelation in lag/rolling features | Lag-feature engineering at 24h, 48h, 72h; champion uses `aqi_lag_24h` and `aqi_lag_72h` |
| Humidity (−0.40), wind (−0.32) as best weather signals | Humidity kept in champion; wind in the candidate set but dropped by SHAP |
| Pressure narrow band, weak signal | Dropped by SHAP |
| Nearly linear PM2.5↔AQI scatter | Ridge regression chosen; tree models tested and rejected |
| O₃ bimodal day/night, weak AQI link | No log transform applied; O₃ not in final feature set |
| NO₂↔CO correlation 0.72 | Redundancy flagged; CO dropped by SHAP |
| Forecast autocorrelation decay (0.63 → 0.49 → 0.41) | One model per horizon; conformal intervals widen with horizon; sets the ceiling for achievable R² |

The EDA is not a separate exercise from modelling — every finding above ties to a specific design decision elsewhere in the report. The model (Section 8) and its conformal intervals (Section 9) are the empirical answer to the questions raised here; the OOT verification (Section 13) confirms the model behaves as the EDA predicted.