# Section 12 — Continuous integration and automation

## 12.1 Overview

Every recurring action in the system is a GitHub Actions cron job. There is no external scheduler and no always-on worker: the schedules live alongside the code in `.github/workflows/`, GitHub-hosted runners provide ephemeral compute, and credentials are injected from GitHub Actions secrets. Two workflows drive the system.

## 12.2 The two workflows

| Workflow | Schedule | Cron | Timeout |
|---|---|---|---|
| Feature Pipeline (hourly) | Hourly, at minute 5 | `5 * * * *` | 10 min |
| Training Pipeline (daily) | Daily, 02:00 UTC (07:00 PKT) | `0 2 * * *` | 30 min |

Both also expose `workflow_dispatch` for manual triggering, used for first-run testing before the cron was enabled. The hourly pipeline fires at minute 5 rather than the top of the hour to give Open-Meteo time to publish the latest hour before the run fetches it.

## 12.3 Run structure

Each run uses an `ubuntu-latest` runner and follows the same shape: check out the repo, set up Python 3.11 with pip caching, install dependencies from `requirements-dev.txt`, then invoke the pipeline as a module (`python -m pipelines.feature_pipeline` / `python -m pipelines.training_pipeline`). Both pass `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` as environment variables sourced from secrets. The training workflow adds one step the feature workflow does not: it runs the full `pytest tests/` suite *before* the training step, so a failing test aborts the run before any model is trained or registered — a quality gate on every daily retrain.

## 12.4 Safeguards

- **Concurrency control.** Each workflow declares a concurrency group with `cancel-in-progress: false`. A new scheduled run will not start while a previous run of the same pipeline is still going, and an in-flight run is never cancelled mid-execution. This prevents overlapping writes and half-completed runs.
- **Timeouts.** 10 minutes for the feature pipeline, 30 for training, bounding any runaway run.
- **Secrets.** Hopsworks credentials are stored as GitHub Actions secrets and referenced as `${{ secrets.NAME }}`; they never appear in the repository.

## 12.5 Why GitHub Actions, and failure handling

GitHub Actions was chosen because the cron schedule lives with the code, free-tier minutes are more than adequate for roughly two dozen short hourly runs plus one daily training run, and secret management is built in — no separate scheduling service to provision or secure. Failure handling is inherited from the platform: any step that exits non-zero fails the run, GitHub marks it red, and the standard Actions notification fires. Because the pipelines raise on error rather than swallowing exceptions, a genuine problem surfaces as a visible red run rather than a silent gap — the observability backstop for the otherwise unattended system. (The one class of failure this does *not* catch — an online write that succeeds while offline materialisation stalls — is discussed in Sections 14.1 and 15.1.)