# Section 16 — Conclusion

## 16.1 What was built

This project produced an end-to-end, automated, publicly deployed AQI forecasting system for Karachi. From a resident's perspective, it answers a question that real-time AQI displays cannot: not just "what is the air like now," but "what will the air be like for the next three days." From an engineering perspective, it demonstrates that a meaningful MLOps system — automated hourly ingestion, daily retraining, model versioning, distribution-free uncertainty quantification, and a live public dashboard — can be assembled and operated entirely on free-tier managed services, with no persistent infrastructure spend.

The deployed model is small, fast, and interpretable: a Ridge regression on five features, retrained automatically each day, with conformal prediction intervals calibrated to 80% nominal coverage. The system is live at `khi-aqi.streamlit.app` and `pearls-aqi-backend.onrender.com`, and has been operating autonomously since deployment.

## 16.2 Key judgment calls

Several decisions defined the project's character, and each was made with empirical justification rather than convention.

**Chronological splits, even where they hurt the headline numbers.** Random or k-fold splits on time-series data silently leak temporal autocorrelation across the train/test boundary and inflate measured R². The chronological split used throughout produces lower headline numbers than a random split would, but those numbers reflect actual generalization to unseen future time. The R² decay across horizons closely matches the EDA-measured autocorrelation decay — the strongest indication that the model is performing at the ceiling set by the data, not under a methodological flaw.

**Ridge over deep learning.** An LSTM on the full 26-feature set under-performed Ridge at every horizon, and Random Forest and XGBoost went negative on the SHAP-pruned 5-feature set. The data is the binding constraint at this scale (one year of hourly observations), and Ridge's regularization and linear inductive bias match the structure of the target better than higher-capacity alternatives. The choice was empirical, not philosophical.

**Conformalized Quantile Regression for uncertainty.** A point forecast without an interval is operationally indistinguishable from a guess. CQR was selected over Gaussian-residual or Bayesian alternatives because it is distribution-free and offers a finite-sample coverage guarantee under exchangeability. The hybrid construction — production point estimates paired with calibration-set-derived intervals — produced empirical hold-out coverage above target at every horizon. Slightly conservative is the correct direction to err for a public-health-relevant tool.

**Dual-endpoint Open-Meteo design.** The system reads from two Open-Meteo endpoints with different freshness and quality characteristics: the validated hourly batch anchors training and forecasting, while the lower-latency `current` endpoint powers the dashboard's live tile only. Mixing them — feeding the `current` reading into the model — would risk a distribution mismatch between training data and inference inputs. Keeping them separated preserves both the integrity of the predictions and the freshness of the live tile.

## 16.3 What the platform issues taught

Operating entirely on free-tier infrastructure surfaced failure modes that a fully-managed paid deployment would have obscured. The Hopsworks offline materialisation stall (Section 14.1) was the most instructive: a log line confirming "online write succeeded" is not the same as "data is queryable for training and inference." The pipeline silently went stale for nine days because the monitoring assumption — "if the write returned success, the data made it" — was wrong. The defensive response, after recovery, was to build the diagnostic and recovery scripts that should have existed from day one, and to record the lesson: in a multi-stage data pipeline, downstream freshness must be verified at every stage, not inferred from upstream success.

The free-tier sleep behaviours of Render and Streamlit were less dramatic but reinforced the same lesson. A heartbeat ping that boots the backend is meaningfully different from one that hits a static shell and reports "up" while the application is asleep. Both incidents argue for the same principle: monitor for what the user experiences, not for what the infrastructure reports.

## 16.4 Closing

The system is modest in scale — one city, three horizons, one year of training data — and the choices throughout reflect that scale honestly. There is no claim to outperform purpose-built operational forecasting from the meteorological agencies that have orders-of-magnitude more data and physics-based atmospheric models behind them. The claim is narrower: that a small team, a public API, and a stack of free-tier managed services are enough to build a working, accountable, decision-supporting forecast for a city that has not had one. On that narrower claim, the empirical evidence supports the system as deployed.