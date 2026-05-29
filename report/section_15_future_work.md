# Section 15 — Future work

The system is complete and operational; the items below are the natural next steps, grouped by the problem they address.

## 15.1 Platform reliability

- **Move off the shared free tier.** The materialisation stall in Section 14.1 is a free-tier capacity problem, not a design flaw. A dedicated Hopsworks tier, or a migration to an alternative stack (Feast, DagsHub, or a managed option such as Vertex AI), would remove the shared-cluster contention that caused the multi-day backlog.
- **Hopsworks Model Deployments.** Hosting the models on Hopsworks infrastructure rather than downloading them at cold start (Section 10.3) would eliminate the per-restart registry fetch and reduce the client/backend version surface noted in Section 14.6.
- **Fail-loud pipeline hardening.** Per the key learning in Section 14.1, the feature pipeline should detect a materialisation backlog — rather than treating a successful online write as overall success — and surface it as a red GitHub Actions run instead of a silent gap.

## 15.2 Modelling and evaluation

- **Path X — live-anchored forecasts.** Currently forecasts anchor to the most recent *validated* hour, which can lag the user's present hour. Anchoring instead to the live `current` reading would make the forecast horizon track "now" more closely. The open question is whether it is worth the trade-off: doing so trades the training/serving distribution match the dual-endpoint design protects (Section 4.3), and would require validating that a model trained on validated data generalises to `current`-endpoint inputs.
- **Rolling-window CQR recalibration.** The conformal coverage figures in Section 9 are measured on a single chronological holdout. Repeating the calibration on rolling windows would show how `Q_widen` evolves as data accumulates and whether coverage holds under distribution shift, rather than only in-distribution (Section 9.7).

## 15.3 Product

- **More alert channels.** The tiered banner (Section 11) could be extended to SMS or webhook/push notifications so users receive hazardous-air warnings without opening the dashboard.
- **Pollution source attribution.** Separating combustion from dust signatures using the inter-pollutant structure observed in the EDA (Section 5) would add explanatory value beyond the composite AQI.
- **Geographic expansion.** The architecture supports multi-city replication; instantiating additional cities was out of scope (Section 2.3.2) but is a straightforward extension given a per-city feature group and model set.