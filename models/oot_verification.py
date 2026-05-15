"""
OOT Verification - Day 17

Tests the May 9 champion models against truly unseen data
(May 9 18:00 - May 13 23:00, 102 hours).

Outputs:
  - Console metrics table (Ridge vs Naive baseline)
  - artifacts/oot/oot_metrics.json
  - artifacts/oot/oot_predictions.csv
  - artifacts/oot/oot_plot_h24.png, h48.png, h72.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

import hopsworks

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "models" / "champion" / "champion_metadata.json"
OUT_DIR = ROOT / "artifacts" / "oot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START = pd.Timestamp("2026-05-09 18:00:00", tz="UTC")
FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]
HORIZONS = [24, 48, 72]


def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)


def fetch_data():
    load_dotenv(ROOT / ".env")
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def predict_ridge(X_scaled, coefs, intercept):
    w = np.array([coefs[f] for f in FEATURES])
    return X_scaled[FEATURES].values @ w + intercept


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return None
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": float(r2), "rmse": rmse, "mae": mae, "n": int(len(y_true))}


def main():
    print("=" * 70)
    print("OOT Verification — May 9 champion vs post-training actuals")
    print("=" * 70)

    meta = load_metadata()
    print(f"Training cutoff: {meta['trained_at_utc']}")
    print(f"OOT window starts: {OOT_START.isoformat()}\n")

    df = fetch_data()
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}\n")

    train_df = df[df["timestamp"] < OOT_START].dropna(subset=FEATURES).reset_index(drop=True)
    oot_df = df[df["timestamp"] >= OOT_START].dropna(subset=FEATURES).reset_index(drop=True)

    print(f"Training-era rows: {len(train_df)}")
    print(f"OOT rows: {len(oot_df)}")
    print(f"OOT AQI stats: mean={oot_df['aqi'].mean():.2f}, std={oot_df['aqi'].std():.2f}\n")

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    oot_scaled = pd.DataFrame(scaler.transform(oot_df[FEATURES]), columns=FEATURES)

    print(f"{'Horizon':<10}{'Train R²':<10}{'OOT R²':<10}{'Naive R²':<11}"
          f"{'OOT RMSE':<11}{'Naive RMSE':<12}{'OOT MAE':<10}{'n':<5}")
    print("-" * 78)

    all_results = {"window_start": OOT_START.isoformat(), "horizons": []}
    pred_rows = []

    for h_meta in meta["horizons"]:
        h = h_meta["horizon"]
        coefs = h_meta["coefficients"]
        intercept = h_meta["intercept"]

        y_pred_ridge = predict_ridge(oot_scaled, coefs, intercept)

        actuals, ridge_aligned, naive_preds, anchors = [], [], [], []
        for i, ts in enumerate(oot_df["timestamp"]):
            target_time = ts + pd.Timedelta(hours=h)
            match = df[df["timestamp"] == target_time]
            if len(match) > 0:
                actuals.append(float(match["aqi"].iloc[0]))
                ridge_aligned.append(float(y_pred_ridge[i]))
                naive_preds.append(float(oot_df["aqi"].iloc[i]))  # naive = current AQI
                anchors.append(target_time)

        ridge_metrics = compute_metrics(actuals, ridge_aligned)
        naive_metrics = compute_metrics(actuals, naive_preds)
        train_r2 = h_meta["metrics"]["r2"]

        print(
            f"{h}h        "
            f"{train_r2:<10.3f}{ridge_metrics['r2']:<10.3f}{naive_metrics['r2']:<11.3f}"
            f"{ridge_metrics['rmse']:<11.2f}{naive_metrics['rmse']:<12.2f}"
            f"{ridge_metrics['mae']:<10.2f}{ridge_metrics['n']:<5}"
        )

        all_results["horizons"].append({
            "horizon": h,
            "train_r2": train_r2,
            "ridge": ridge_metrics,
            "naive": naive_metrics,
        })

        for ts, a, r, n in zip(anchors, actuals, ridge_aligned, naive_preds):
            pred_rows.append({
                "target_time": ts.isoformat(),
                "horizon": h,
                "actual": a,
                "ridge_pred": r,
                "naive_pred": n,
            })

        # Plot
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(anchors, actuals, "o-", label="Actual", color="black", linewidth=2)
        ax.plot(anchors, ridge_aligned, "s--", label=f"Ridge (R²={ridge_metrics['r2']:.2f})",
                color="steelblue")
        ax.plot(anchors, naive_preds, "^:", label=f"Naive (R²={naive_metrics['r2']:.2f})",
                color="coral", alpha=0.7)
        ax.set_xlabel("Target time (UTC)")
        ax.set_ylabel("AQI")
        ax.set_title(f"OOT Verification — {h}h horizon (n={ridge_metrics['n']})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plot_path = OUT_DIR / f"oot_plot_h{h}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  → saved {plot_path.name}")

    # Save artifacts
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "oot_predictions.csv", index=False)
    with open(OUT_DIR / "oot_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nArtifacts saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()