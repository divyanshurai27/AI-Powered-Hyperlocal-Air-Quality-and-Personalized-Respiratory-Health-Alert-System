"""Train and evaluate 24 h AQ forecasters; freeze the artifact the API will serve.

Run from the repo root (venv active):
    python ml/forecasting/train_aq_forecast.py --pollutant pm25
    python ml/forecasting/train_aq_forecast.py --pollutant pm25 --version 1.0.0

Protocol (identical for every model, PRD §51 check 4):
  - chronological split: train < validation < test, never shuffled
  - 24 h embargo at the end of train and validation, so no target leaks across a boundary
  - validation is used only for LightGBM early stopping; the test period is touched once
  - scored only against observed (not imputed) readings
Outputs:
  ml/artifacts/aq_forecast/<pollutant>/<version>/{model.joblib, metadata.json}
  docs/experiments/aq_forecast_<pollutant>_<version>.json
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.core.logging import configure_logging  # noqa: E402
from app.db.session import get_sessionmaker  # noqa: E402
from app.domain.pollutants import Pollutant  # noqa: E402
from app.ml.dataset import load_observations, load_weather, station_coordinates  # noqa: E402
from app.ml.features import (  # noqa: E402
    FEATURE_VERSION,
    HORIZONS,
    build_forecasting_matrix,
    build_station_frames,
    feature_names,
)
from app.ml.forecasting import (  # noqa: E402
    DirectGBMForecaster,
    ForecastArtifactMetadata,
    GBMConfig,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    compare_forecast_models,
    dataset_fingerprint,
    save_model_artifact,
    validate_forecast,
)

EMBARGO = pd.Timedelta(hours=max(HORIZONS))
DEFAULT_SPLITS = {
    "train": ("2025-02-18", "2026-01-31"),
    "validation": ("2026-02-01", "2026-04-30"),
    "test": ("2026-05-01", "2026-09-30"),
}


def _ts(day: str, end: bool = False) -> pd.Timestamp:
    t = pd.Timestamp(day, tz="UTC")
    return t + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) if end else t


def split(frames: tuple[pd.DataFrame, ...], start: str, end: str, embargo: bool):
    t = frames[0].index.get_level_values("t")
    hi = _ts(end, end=True) - (EMBARGO if embargo else pd.Timedelta(0))
    mask = (t >= _ts(start)) & (t <= hi)
    return tuple(f[mask] for f in frames)


def print_table(title: str, table: list[dict]) -> None:
    print(f"\n{title}")
    cols = ("MAE", "RMSE", "R²", "MAE h1", "MAE h6", "MAE h24", "skill", "n")
    print(f"{'model':18} " + " ".join(f"{c:>8}" for c in cols))
    for r in table:
        s = r["skill_vs_persistence"]
        vals = (r["mae"], r["rmse"], r["r2"], r["mae_h1"], r["mae_h6"], r["mae_h24"])
        cells = [f"{v:8.2f}" for v in vals] + [f"{s:+8.1%}" if s is not None else f"{'—':>8}"]
        print(f"{r['model']:18} " + " ".join(cells) + f" {r['n']:>8}")


def evaluate_common(preds: dict[str, pd.DataFrame], Y: pd.DataFrame) -> dict:
    """Score every model on the SAME (origin, horizon) pairs: those where all models produced
    a forecast and an observation exists. Otherwise a baseline that abstains on hard cases
    would look better than it is."""
    common = Y.notna()
    for p in preds.values():
        common &= p.notna()
    target = Y.where(common)
    return {name: validate_forecast(p, target) for name, p in preds.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pollutant", default="pm25", choices=[p.value for p in Pollutant])
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()
    configure_logging("WARNING")
    pollutant = Pollutant(args.pollutant)
    version = f"aq_{pollutant.value}_v{args.version}"

    with get_sessionmaker()() as db:
        obs = load_observations(db)
        weather = load_weather(db)
        coords = station_coordinates(db)
    if obs.empty or weather.empty:
        print("No observations/weather in the database. Run the backfills first.")
        return 1
    dataset_id = dataset_fingerprint(obs)
    print(
        f"dataset {dataset_id}: {len(obs):,} observations, {obs['station_id'].nunique()} stations"
    )

    frames = build_station_frames(obs)
    X, Y, S, YA = build_forecasting_matrix(frames, weather, pollutant, coords)
    print(f"matrix: {len(X):,} origins × {X.shape[1]} features ({FEATURE_VERSION})")

    parts = {
        name: split((X, Y, S, YA), lo, hi, embargo=(name != "test"))
        for name, (lo, hi) in DEFAULT_SPLITS.items()
    }
    for name, (x, *_) in parts.items():
        t = x.index.get_level_values("t")
        print(f"  {name:10} {len(x):>8,} rows  {t.min():%Y-%m-%d} → {t.max():%Y-%m-%d}")
    (Xtr, Ytr, _, _), (Xva, Yva, Sva, _) = parts["train"], parts["validation"]
    Xte, Yte, Ste, YAte = parts["test"]

    names = feature_names(pollutant)
    config = GBMConfig()
    gbm = DirectGBMForecaster(names, config)
    print("training 24 LightGBM horizon models ...")
    gbm.fit(Xtr, Ytr, Xva, Yva)

    persistence = PersistenceForecaster(f"{pollutant.value}_current")
    seasonal = SeasonalNaiveForecaster()
    results_val = evaluate_common(
        {
            "persistence": persistence.predict(Xva),
            "seasonal_naive": seasonal.predict(Sva),
            gbm.name: gbm.predict(Xva),
        },
        Yva,
    )
    test_preds = {
        "persistence": persistence.predict(Xte),
        "seasonal_naive": seasonal.predict(Ste),
        gbm.name: gbm.predict(Xte),
    }
    results_test = evaluate_common(test_preds, Yte)
    results_test_all = evaluate_common(test_preds, YAte)
    table = compare_forecast_models(results_test)
    table_all = compare_forecast_models(results_test_all)
    period = f"{DEFAULT_SPLITS['test'][0]} → {DEFAULT_SPLITS['test'][1]}"
    print_table(f"TEST {period}, {pollutant.value} µg/m³ — VALID hours (primary)", table)
    print_table(
        f"TEST {period}, {pollutant.value} µg/m³ — ALL observed hours (secondary)",
        table_all,
    )

    importances = pd.Series(gbm.models[6].feature_importances_, index=names).sort_values(
        ascending=False
    )
    print("\ntop features (h6 model, split count):")
    for n, v in importances.head(8).items():
        print(f"  {n:28} {v}")

    t = lambda x: x.index.get_level_values("t")  # noqa: E731
    meta = ForecastArtifactMetadata(
        model_name=gbm.name,
        model_version=version,
        pollutant=pollutant.value,
        feature_version=FEATURE_VERSION,
        feature_names=names,
        horizons=list(HORIZONS),
        training_dataset=dataset_id,
        training_start=str(t(Xtr).min()),
        training_end=str(t(Xtr).max()),
        validation_start=str(t(Xva).min()),
        validation_end=str(t(Xva).max()),
        test_start=str(t(Xte).min()),
        test_end=str(t(Xte).max()),
        hyperparameters={**config.__dict__, "best_iterations": gbm.best_iterations},
        validation_metrics=results_val[gbm.name]["overall"],
        test_metrics=results_test[gbm.name]["overall"],
    )
    out_dir = save_model_artifact(
        gbm,
        meta,
        REPO / "ml" / "artifacts" / "aq_forecast" / pollutant.value / args.version,
    )

    report = {
        "experiment_id": f"aq_forecast_{pollutant.value}_{args.version}",
        "run_at": datetime.now(UTC).isoformat(),
        "pollutant": pollutant.value,
        "dataset": dataset_id,
        "feature_version": FEATURE_VERSION,
        "features": names,
        "splits": DEFAULT_SPLITS,
        "embargo_hours": int(EMBARGO.total_seconds() // 3600),
        "rows": {k: len(v[0]) for k, v in parts.items()},
        "model_version": version,
        "hyperparameters": meta.hyperparameters,
        "target_definition": "observed hours flagged VALID (>=75% coverage, not provider-flagged)",
        "comparison_test": table,
        "comparison_test_all_observed": table_all,
        "test_all_observed": results_test_all,
        "validation": results_val,
        "test": results_test,
        "artifact": str(out_dir.relative_to(REPO)),
        "model_sha256": meta.model_sha256,
    }
    exp_dir = REPO / "docs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_path = exp_dir / f"aq_forecast_{pollutant.value}_{args.version}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nartifact → {out_dir.relative_to(REPO)}\nreport   → {report_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
