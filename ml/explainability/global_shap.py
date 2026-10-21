"""Global SHAP importance of the deployed risk model on the SYNTHETIC cohort (PRD §17, §51 check 6).

Run from the repo root (venv active, database up), after ml/risk/train_risk_model.py:
    python ml/explainability/global_shap.py                 # risk_v1.0.0
    python ml/explainability/global_shap.py --version 1.0.0 --part test_seen

It rebuilds exactly the training matrix (same cohort cache, same origins, same splits) and
explains the SAVED artifact with the same code the API uses (app/ml/explain.py), so the
reported importances describe the model patients are served, not a retrained copy.

Reported, per evaluation part and per disease:
  - mean |SHAP| per feature and per patient-facing factor, in calibrated log-odds
  - the share of total |SHAP| per factor
  - mean signed SHAP (direction on average)
  - an additivity check: max |sigmoid(base + Σφ) − calibrated probability|
Output: docs/experiments/risk_<version>_shap.json

Synthetic labels: importances show what the simulated cohort encoded (e.g. the diary
carries the signal), not clinical knowledge.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "ml" / "risk"))

from app.core.logging import configure_logging  # noqa: E402
from app.db.session import get_sessionmaker  # noqa: E402
from app.ml.cohort import CohortConfig  # noqa: E402
from app.ml.dataset import load_observations, load_weather, station_coordinates  # noqa: E402
from app.ml.explain import (  # noqa: E402
    EXPLANATION_VERSION,
    FACTORS,
    calibration_scale,
    explain_margin,
)
from app.ml.features import build_station_frames  # noqa: E402
from app.ml.forecasting import dataset_fingerprint  # noqa: E402
from app.ml.risk import load_risk_artifact, matrix_fingerprint  # noqa: E402
from app.ml.risk_dataset import build_cohort_matrix, sample_origins  # noqa: E402
from train_risk_model import DATA_START, assign_parts, load_or_generate  # noqa: E402

PARTS = ("validation", "test_seen", "test_unseen")


def summarise(phi: np.ndarray, names: list[str]) -> dict:
    abs_mean = np.abs(phi).mean(0)
    signed = phi.mean(0)
    per_feature = {
        n: {"mean_abs": round(float(a), 6), "mean_signed": round(float(s), 6)}
        for n, a, s in sorted(zip(names, abs_mean, signed, strict=True), key=lambda t: -t[1])
    }
    idx = {n: i for i, n in enumerate(names)}
    # Factor importance = mean |Σ φ over the factor's features| (contributions inside a factor
    # can cancel, exactly as in the per-patient explanation).
    factor_abs = {
        key: float(np.abs(phi[:, [idx[f] for f in feats if f in idx]].sum(1)).mean())
        for key, (_, feats) in FACTORS.items()
    }
    total = sum(factor_abs.values()) or 1.0
    per_factor = {
        key: {
            "label": FACTORS[key][0],
            "mean_abs": round(v, 6),
            "share": round(v / total, 4),
        }
        for key, v in sorted(factor_abs.items(), key=lambda t: -t[1])
    }
    return {"n": int(len(phi)), "factors": per_factor, "features": per_feature}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    configure_logging("WARNING")

    directory = REPO / "ml" / "artifacts" / "risk" / args.version
    model, calibrators, meta = load_risk_artifact(directory)

    with get_sessionmaker()() as db:
        obs = load_observations(db, start=pd.Timestamp(DATA_START, tz="UTC").to_pydatetime())
        weather = load_weather(db)
        coords = station_coordinates(db)
    if obs.empty or weather.empty:
        print("No observations/weather in the database. Run the backfills first.")
        return 1
    obs = obs[obs["pollutant"].isin(["pm25", "pm10", "o3"])]
    frames = build_station_frames(obs)
    start, end = obs["timestamp"].min(), obs["timestamp"].max()

    saved = meta.cohort_config
    # Built exactly as the training run did, so the cohort cache key matches.
    cfg = CohortConfig(
        n_patients=saved["n_patients"],
        seed=saved["seed"],
        aq_effect_scale=saved["aq_effect_scale"],
    )
    if json.loads(json.dumps(cfg.to_dict(), default=str)) != json.loads(
        json.dumps(saved, default=str)
    ):
        print("cohort config differs from the one the artifact was trained on")
        return 1
    cohort = load_or_generate(cfg, frames, weather, coords, start, end, dataset_fingerprint(obs))
    X, y_s, row_meta = build_cohort_matrix(cohort, sample_origins(cohort))
    dataset_id = matrix_fingerprint(X, y_s)
    if dataset_id != meta.dataset_id:
        print(f"matrix {dataset_id} differs from the artifact's {meta.dataset_id}: data changed")
        return 1
    d = row_meta["disease_type"].to_numpy()
    parts = assign_parts(row_meta, len(cohort.patients), cfg.seed).to_numpy()

    report = {
        "model_version": meta.model_version,
        "calibration_version": calibrators.version,
        "explanation_version": EXPLANATION_VERSION,
        "dataset_id": dataset_id,
        "synthetic_labels": True,
        "space": "calibrated_log_odds",
        "created_at": datetime.now(UTC).isoformat(),
        "parts": {},
    }
    for part in PARTS:
        m = parts == part
        if not m.any():
            continue
        Xp, dp = X[m].reset_index(drop=True), d[m]
        contrib, base = explain_margin(model, Xp, dp)
        scaled = np.empty_like(contrib)
        max_err = 0.0
        p_cal = calibrators.calibrate(model.predict_proba(Xp, dp), dp)
        out = {"all": None, "by_disease": {}}
        for dis in np.unique(dp):
            k = dp == dis
            a, b = calibration_scale(calibrators.by_disease[str(dis)])
            scaled[k] = a * contrib[k]
            recon = 1 / (1 + np.exp(-(a * (contrib[k].sum(1) + base[k]) + b)))
            max_err = max(max_err, float(np.abs(recon - p_cal[k]).max()))
            out["by_disease"][str(dis)] = summarise(scaled[k], model.feature_names)
        out["all"] = summarise(scaled, model.feature_names)
        out["additivity_max_abs_error"] = max_err
        report["parts"][part] = out
        top = list(out["all"]["factors"].items())[:6]
        print(f"{part} (n={m.sum():,}, additivity err {max_err:.1e}):")
        for _key, v in top:
            print(f"  {v['label']:52} mean|φ| {v['mean_abs']:.4f}  share {v['share']:.1%}")

    path = REPO / "docs" / "experiments" / f"risk_{args.version}_shap.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
