"""Train, calibrate and evaluate the respiratory-risk model on the SYNTHETIC cohort.

Run from the repo root (venv active, database up):
    python ml/risk/train_risk_model.py                          # main run, saves the artifact
    python ml/risk/train_risk_model.py --aq-effect-scale 0 --no-save   # negative control
    python ml/risk/train_risk_model.py --skip-sensitivity       # faster, no exposure scenarios

Protocol (fixed before looking at any test result):
  - cohort: real Bengaluru AQ + weather 2025-02-18 → last observation, simulated patients
  - patients: 80% "seen" (train/validation/test over time), 20% "unseen" (test only)
  - time: train ≤ 2026-01-31 < validation ≤ 2026-04-30 < test; origins whose 24 h label
    window crosses a boundary are dropped (embargo)
  - validation: LightGBM early stopping, calibration fit, policy threshold derivation
  - test: touched once; seen patients are the primary result, unseen patients show how the
    model does for people it never saw
  - 95% CIs: patient-level cluster bootstrap
Outputs:
  ml/artifacts/risk/<version>/{model.joblib, calibration.joblib, metadata.json}
  docs/experiments/risk_<version>[_aq<scale>].json
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.core.logging import configure_logging  # noqa: E402
from app.db.session import get_sessionmaker  # noqa: E402
from app.domain.exposure import sensitivity_factors  # noqa: E402
from app.domain.risk_policy import (  # noqa: E402
    RISK_POLICY_VERSION,
    RISK_THRESHOLDS,
    apply_personal_offset,
    estimate_personal_baseline,
)
from app.ml.calibration import (  # noqa: E402
    CALIBRATION_VERSION,
    best_f1_threshold,
    cluster_bootstrap,
    fit_disease_calibrators,
    precision_threshold,
    reliability_table,
    threshold_metrics,
    validate_risk_prediction,
)
from app.ml.cohort import COHORT_VERSION, CohortConfig, generate_cohort  # noqa: E402
from app.ml.dataset import load_observations, load_weather, station_coordinates  # noqa: E402
from app.ml.features import build_station_frames  # noqa: E402
from app.ml.forecasting import dataset_fingerprint  # noqa: E402
from app.ml.risk import (  # noqa: E402
    LogisticRisk,
    PooledGBM,
    RiskArtifactMetadata,
    RiskGBMConfig,
    SeparateGBM,
    SharedHeadRiskModel,
    StratumPrior,
    matrix_fingerprint,
    save_risk_artifact,
)
from app.ml.risk_dataset import (  # noqa: E402
    LABEL_HOURS,
    build_cohort_matrix,
    patient_health,
    sample_origins,
)
from app.ml.risk_features import (  # noqa: E402
    EXPOSURE_FEATURES,
    FEATURE_GROUPS,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_VERSION,
    personal_history_many,
)

DATA_START = "2025-02-18"
SPLITS = {"train_end": "2026-01-31", "validation_end": "2026-04-30"}
SEEN_FRACTION = 0.8
CACHE = REPO / "data" / "processed"


def ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)  # end of that day


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def load_or_generate(cfg: CohortConfig, frames, weather, coords, start, end, obs_id: str):
    key = hashlib.sha256(
        json.dumps(
            [COHORT_VERSION, obs_id, cfg.to_dict(), str(start), str(end)],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:12]
    path = CACHE / f"cohort_{key}.joblib"
    if path.exists():
        print(f"cohort: cached {path.name}")
        return joblib.load(path)  # noqa: S301  (our own cache file)
    t0 = time.time()
    cohort = generate_cohort(frames, weather, coords, start, end, cfg)
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(cohort, path, compress=3)
    print(f"cohort: generated in {time.time() - t0:.0f}s → {path.name}")
    return cohort


def assign_parts(meta: pd.DataFrame, n_patients: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    seen = rng.permutation(n_patients)[: int(round(SEEN_FRACTION * n_patients))]
    is_seen = meta["patient_idx"].isin(seen).to_numpy()
    t = meta["t"]
    embargo = pd.Timedelta(hours=LABEL_HOURS)
    tr_end, va_end = ts(SPLITS["train_end"]), ts(SPLITS["validation_end"])
    part = np.full(len(meta), "", dtype=object)
    part[is_seen & (t <= tr_end - embargo)] = "train"
    part[is_seen & (t > tr_end) & (t <= va_end - embargo)] = "validation"
    part[is_seen & (t > va_end)] = "test_seen"
    part[~is_seen & (t > va_end)] = "test_unseen"
    return pd.Series(part, index=meta.index)


def stratum_rates(cohort, meta, part) -> dict[str, float]:
    """Reported exacerbations per diary day, per disease|severity, train patients/period."""
    d = cohort.diary
    train_patients = set(meta.loc[part == "train", "patient_idx"])
    d = d[d["patient_idx"].isin(train_patients) & (d["timestamp"] <= ts(SPLITS["train_end"]))]
    d = d[d["exacerbation"].notna()].merge(
        cohort.patients[["disease_type", "severity"]], left_on="patient_idx", right_index=True
    )
    rates = d.groupby(["disease_type", "severity"])["exacerbation"].mean()
    return {f"{a}|{b}": round(float(v), 6) for (a, b), v in rates.items()}


def metrics_with_ci(p, y, groups, n_boot) -> dict:
    out = validate_risk_prediction(p, y)
    if n_boot:
        from sklearn.metrics import average_precision_score, roc_auc_score

        for name, fn in (
            ("auroc", lambda i: roc_auc_score(y[i], p[i]) if 0 < y[i].sum() < len(i) else np.nan),
            ("auprc", lambda i: average_precision_score(y[i], p[i]) if y[i].sum() else np.nan),
            ("brier", lambda i: float(np.mean((p[i] - y[i]) ** 2))),
        ):
            out[f"{name}_ci95"] = cluster_bootstrap(groups, fn, n_boot=n_boot)
    return out


def train_and_score(factory, X, y, d, parts, feats=None) -> tuple[dict, object, object]:
    tr, va = parts == "train", parts == "validation"
    model = factory()
    model.fit(X[tr], y[tr], d[tr], X[va], y[va], d[va])
    cal = fit_disease_calibrators(model.predict_proba(X[va], d[va]), y[va], d[va])
    scores = {}
    for name in ("validation", "test_seen", "test_unseen"):
        m = parts == name
        scores[name] = cal.calibrate(model.predict_proba(X[m], d[m]), d[m])
    return scores, model, cal


def derive_policy(p_val, y_val, d_val) -> dict:
    """The rule documented in domain/risk_policy.py, applied to validation data only."""
    out = {}
    for disease in np.unique(d_val):
        m = d_val == disease
        base = float(y_val[m].mean())
        high = best_f1_threshold(p_val[m], y_val[m])
        precise = precision_threshold(p_val[m], y_val[m], 0.5)
        out[str(disease)] = {
            "base_rate": round(base, 5),
            "moderate_1.5x_base": round(1.5 * base, 4),
            "high_best_f1": round(high, 4),
            "severe_top1pct": round(float(np.quantile(p_val[m], 0.99)), 4),
            "precision_0.5_threshold": round(precise, 4) if precise is not None else None,
        }
    return out


def level_table(p, y, d) -> dict:
    out = {}
    for disease in np.unique(d):
        m = d == disease
        moderate, high, severe = RISK_THRESHOLDS[str(disease)]
        cuts = [0, moderate, high, severe, 1.01]
        rows = {}
        names = ("Low", "Moderate", "High", "Severe")
        for name, lo, hi in zip(names, cuts[:-1], cuts[1:], strict=True):
            k = m & (p >= lo) & (p < hi)
            rows[name] = {
                "share": round(float(k.sum() / m.sum()), 4),
                "event_rate": round(float(y[k].mean()), 4) if k.any() else None,
                "n": int(k.sum()),
            }
        out[str(disease)] = rows
    return out


def personalization_comparison(cohort, X, y, d, meta, parts, p_cal, rates) -> dict:
    """PRD §16: generic AQ threshold → disease-specific → model → adaptive personal."""
    out = {}
    for part_name in ("test_seen", "test_unseen"):
        m = (parts == part_name).to_numpy()
        yy, dd = y[m], d[m]
        pm25, pm10 = X.loc[m, "exp_pm25_mean_24h"], X.loc[m, "exp_pm10_mean_24h"]
        # NAQI 24 h breakpoints; the guidance rules alert (LIMIT level) at Poor for the general
        # population and at Moderate for asthma/COPD (domain/guidance.py).
        generic = ((pm25 > 90) | (pm10 > 250)).to_numpy()
        disease_aq = ((pm25 > 60) | (pm10 > 100)).to_numpy()
        high = np.array([RISK_THRESHOLDS[x][1] for x in dd])
        model_alert = p_cal[part_name] >= high
        global_t = best_f1_threshold(p_cal["validation"], y[(parts == "validation").to_numpy()])
        model_global = p_cal[part_name] >= global_t

        # Personal offset from each patient's own diary before T (same code as serving).
        mm = meta[m]
        events = np.zeros(m.sum(), dtype=int)
        days = np.zeros(m.sum(), dtype=int)
        pos = np.arange(m.sum())
        for i, rows in mm.groupby("patient_idx").indices.items():
            e, n = personal_history_many(
                patient_health(cohort, i), pd.DatetimeIndex(mm["t"].iloc[rows])
            )
            events[pos[rows]], days[pos[rows]] = e, n
        strata = (mm["disease_type"] + "|" + mm["severity"]).to_numpy()
        p_personal = np.array(
            [
                apply_personal_offset(p, estimate_personal_baseline(e, n, rates.get(s, 0.0)))
                for p, e, n, s in zip(p_cal[part_name], events, days, strata, strict=True)
            ]
        )
        personal_alert = p_personal >= high

        def alert_stats(alert, yy=yy):
            s = threshold_metrics(alert.astype(float), yy, 0.5)
            return {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in s.items()
                if k != "threshold"
            }

        out[part_name] = {
            "generic_aq_threshold": alert_stats(generic),
            "disease_specific_aq_threshold": alert_stats(disease_aq),
            "model_global_threshold": alert_stats(model_global) | {"threshold": round(global_t, 4)},
            "model_disease_threshold": alert_stats(model_alert),
            "model_adaptive_personal": alert_stats(personal_alert),
            "probability_metrics": {
                "model": validate_risk_prediction(p_cal[part_name], yy),
                "model_adaptive_personal": validate_risk_prediction(p_personal, yy),
            },
        }
    return out


def exposure_error(cohort, meta, parts, factors=None, mode="estimated") -> dict:
    """How far the system's exposure (24 h mean PM2.5) is from the simulated truth."""
    from app.ml.risk_dataset import patient_exposure

    seen = sorted(set(meta.loc[parts == "test_seen", "patient_idx"]))
    lo = cohort.hours.searchsorted(ts(SPLITS["validation_end"]))
    est_all, true_all = [], []
    for i in seen:
        est = patient_exposure(cohort, i, mode, factors)["pm25"].iloc[lo:]
        true = pd.Series(cohort.true_exposure["pm25"][i][lo:], index=est.index)
        est_all.append(est.rolling(24, min_periods=12).mean().to_numpy())
        true_all.append(true.rolling(24, min_periods=12).mean().to_numpy())
    e, t = np.concatenate(est_all), np.concatenate(true_all)
    ok = ~np.isnan(e) & ~np.isnan(t)
    from scipy.stats import spearmanr

    return {
        "mae_24h_mean_pm25": round(float(np.mean(np.abs(e[ok] - t[ok]))), 3),
        "bias_24h_mean_pm25": round(float(np.mean(e[ok] - t[ok])), 3),
        "spearman": round(float(spearmanr(e[ok], t[ok]).statistic), 4),
        "coverage": round(float(ok.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--n-patients", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--aq-effect-scale", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    configure_logging("WARNING")
    version = f"risk_v{args.version}"
    t_start = time.time()

    with get_sessionmaker()() as db:
        obs = load_observations(db, start=pd.Timestamp(DATA_START, tz="UTC").to_pydatetime())
        weather = load_weather(db)
        coords = station_coordinates(db)
    if obs.empty or weather.empty:
        print("No observations/weather in the database. Run the backfills first.")
        return 1
    obs = obs[obs["pollutant"].isin(["pm25", "pm10", "o3"])]
    obs_id = dataset_fingerprint(obs)
    frames = build_station_frames(obs)
    start, end = obs["timestamp"].min(), obs["timestamp"].max()
    print(f"AQ dataset {obs_id}: {len(obs):,} PM2.5/PM10/O3 observations, {start} → {end}")

    cfg = CohortConfig(
        n_patients=args.n_patients, seed=args.seed, aq_effect_scale=args.aq_effect_scale
    )
    cohort = load_or_generate(cfg, frames, weather, coords, start, end, obs_id)
    summary = cohort.summary()
    print("cohort:", json.dumps(summary))

    origins = sample_origins(cohort)
    X, y_s, meta = build_cohort_matrix(cohort, origins)
    y = y_s.to_numpy()
    d = meta["disease_type"].to_numpy()
    parts = assign_parts(meta, len(cohort.patients), args.seed)
    groups = meta["patient_idx"].to_numpy()
    dataset_id = matrix_fingerprint(X, y_s)
    counts = {
        k: {"rows": int((parts == k).sum()), "events": int(y[parts == k].sum())}
        for k in ("train", "validation", "test_seen", "test_unseen")
    }
    print(f"matrix {dataset_id}: {X.shape[0]:,} rows × {X.shape[1]} features; {counts}")

    gbm = RiskGBMConfig()
    feats = list(RISK_FEATURE_NAMES)
    no_exposure = [f for f in feats if f not in EXPOSURE_FEATURES]
    no_health = [f for f in feats if f not in FEATURE_GROUPS["health"]]
    candidates = {
        "stratum_prior": lambda: StratumPrior(),
        "logistic": lambda: LogisticRisk(feats),
        "lightgbm_pooled": lambda: PooledGBM(feats, gbm),
        "lightgbm_separate": lambda: SeparateGBM(feats, gbm),
        "lightgbm_shared_heads": lambda: SharedHeadRiskModel(feats, gbm),
        "ablation_no_exposure": lambda: SharedHeadRiskModel(no_exposure, gbm),
        "ablation_no_health": lambda: SharedHeadRiskModel(no_health, gbm),
    }
    results, scores, proposed = {}, {}, None
    for name, factory in candidates.items():
        t0 = time.time()
        sc, model, cal = train_and_score(factory, X, y, d, parts)
        scores[name] = sc
        res = {}
        for part_name in ("validation", "test_seen", "test_unseen"):
            m = (parts == part_name).to_numpy()
            ci = args.bootstrap if part_name != "validation" else 0
            res[part_name] = metrics_with_ci(sc[part_name], y[m], groups[m], ci)
            res[part_name]["by_disease"] = {
                str(k): validate_risk_prediction(sc[part_name][d[m] == k], y[m][d[m] == k])
                for k in np.unique(d[m])
            }
        results[name] = res
        if name == "lightgbm_shared_heads":
            proposed = (model, cal)
        r = res["test_seen"]
        print(
            f"  {name:24} AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  "
            f"Brier {r['brier']:.5f}  ECE {r['ece']:.4f}  ({time.time() - t0:.0f}s)"
        )

    model, calibrators = proposed
    va = (parts == "validation").to_numpy()
    policy_derivation = derive_policy(scores["lightgbm_shared_heads"]["validation"], y[va], d[va])
    print("policy derivation (validation):", json.dumps(policy_derivation))
    te = (parts == "test_seen").to_numpy()
    p_te = scores["lightgbm_shared_heads"]["test_seen"]
    rates = stratum_rates(cohort, meta, parts)
    personalization = personalization_comparison(
        cohort, X, y, d, meta, parts, scores["lightgbm_shared_heads"], rates
    )
    for strat, s in personalization["test_seen"].items():
        if "f1" in s:
            print(
                f"  {strat:30} precision {s['precision']:.3f} recall {s['recall']:.3f} "
                f"F1 {s['f1']:.3f} alert-rate {s['alert_rate']:.3f}"
            )

    sensitivity = {}
    if not args.skip_sensitivity:
        scenarios = {
            "estimated": ("estimated", None),
            "factors_x0.8": ("estimated", sensitivity_factors(0.8)),
            "factors_x1.2": ("estimated", sensitivity_factors(1.2)),
            "home_ambient_only": ("home_ambient", None),
            "true_exposure_oracle": ("true", None),
        }
        for name, (mode, factors) in scenarios.items():
            t0 = time.time()
            Xs = (
                X if name == "estimated" else build_cohort_matrix(cohort, origins, mode, factors)[0]
            )
            sc, _, _ = train_and_score(lambda: SharedHeadRiskModel(feats, gbm), Xs, y, d, parts)
            sensitivity[name] = {
                "exposure_error": exposure_error(cohort, meta, parts, factors, mode),
                "test_seen": validate_risk_prediction(sc["test_seen"], y[te]),
            }
            r = sensitivity[name]
            print(
                f"  exposure {name:22} MAE {r['exposure_error']['mae_24h_mean_pm25']:6.2f}  "
                f"ρ {r['exposure_error']['spearman']:.3f}  AUROC {r['test_seen']['auroc']:.4f}"
                f"  Brier {r['test_seen']['brier']:.5f}  ({time.time() - t0:.0f}s)"
            )

    report = {
        "model_version": version,
        "synthetic_labels": True,
        "created_at": datetime.now(UTC).isoformat(),
        "aq_dataset": obs_id,
        "risk_dataset": dataset_id,
        "cohort": summary,
        "cohort_config": cfg.to_dict(),
        "feature_version": RISK_FEATURE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "policy_version": RISK_POLICY_VERSION,
        "splits": SPLITS | {"seen_fraction": SEEN_FRACTION, "counts": counts},
        "models": results,
        "calibration_methods": {k: c.method for k, c in calibrators.by_disease.items()},
        "reliability_test_seen": reliability_table(p_te, y[te]),
        "policy_derivation_validation": policy_derivation,
        "policy_levels_test_seen": level_table(p_te, y[te], d[te]),
        "policy_thresholds_in_use": {str(k): v for k, v in RISK_THRESHOLDS.items()},
        "stratum_daily_rates": rates,
        "personalization": personalization,
        "exposure_sensitivity": sensitivity,
        "head_iterations": model.head_iterations,
        "runtime_seconds": round(time.time() - t_start),
    }
    suffix = "" if args.aq_effect_scale == 1.0 else f"_aq{args.aq_effect_scale:g}"
    out = REPO / "docs" / "experiments" / f"risk_{args.version}{suffix}.json"
    out.write_text(json.dumps(report, indent=2, default=fmt))
    print(f"report → {out.relative_to(REPO)}")

    if not args.no_save:
        meta_obj = RiskArtifactMetadata(
            model_name=model.name,
            model_version=version,
            feature_version=RISK_FEATURE_VERSION,
            feature_names=feats,
            target="respiratory event onset within the next 24 h",
            cohort_version=COHORT_VERSION,
            cohort_config=cfg.to_dict(),
            dataset_id=dataset_id,
            splits=report["splits"],
            hyperparameters=vars(gbm),
            head_iterations=model.head_iterations,
            stratum_daily_rates=rates,
            calibration_version=calibrators.version,
            calibration_methods=report["calibration_methods"],
            validation_metrics=results["lightgbm_shared_heads"]["validation"],
            test_metrics=results["lightgbm_shared_heads"]["test_seen"],
        )
        path = save_risk_artifact(
            model, calibrators, meta_obj, REPO / "ml" / "artifacts" / "risk" / args.version
        )
        print(f"artifact → {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
