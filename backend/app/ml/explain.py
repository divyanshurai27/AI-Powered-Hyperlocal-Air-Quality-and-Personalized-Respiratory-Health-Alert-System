"""Explanations of risk predictions (PRD §17, BR-09), EXPLANATION_VERSION.

Method: exact TreeSHAP from LightGBM itself (`pred_contrib=True`), so the API needs no extra
dependency. SHAP values are additive, so for the shared-trunk model

    margin(x) = trunk(x) + head_d(x)  →  φ(x) = φ_trunk(x) + φ_head_d(x)

and Σφ + base = margin exactly. The reported probability is not the margin: Platt
calibration is linear in log-odds and the personal offset is added in log-odds, so

    logit(p_reported) = a·margin + b + offset
                      = (a·base + b)  +  Σ a·φ_i  +  offset

Each feature's contribution is therefore a·φ_i in the log-odds of the probability the patient
actually sees, and the parts add up to that probability (checked by tests). An isotonic
calibrator is not linear, so for it the explanation stays in the model's raw log-odds and
says so (`space`).

Wording rule (PRD §17): contributions say how the MODEL reached the estimate. They are not
causes ("PM2.5 caused your attack" is never said), and with synthetic labels they describe
what the synthetic cohort encoded, nothing clinical.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.calibration import Calibrator
from app.ml.risk import PooledGBM, SharedHeadRiskModel
from app.ml.risk_features import RISK_FEATURE_NAMES

EXPLANATION_VERSION = "treeshap_grouped_v1.0.0"
TOP_N = 5
# Factors whose |contribution| is below this (log-odds) are reported as "no clear effect".
NEGLIGIBLE_LOG_ODDS = 0.02

NOTE = (
    "These factors describe how the model arrived at this estimate. They are associations "
    "learned from a synthetic cohort, not causes of symptoms."
)

# Patient-facing factors: each feature belongs to exactly one (tests enforce this).
FACTORS: dict[str, tuple[str, list[str]]] = {
    "pm25_exposure": (
        "Estimated PM2.5 exposure",
        [
            "exp_pm25_now",
            "exp_pm25_mean_6h",
            "exp_pm25_mean_24h",
            "exp_pm25_mean_72h",
            "exp_pm25_max_24h",
        ],
    ),
    "pm25_change": ("Recent change in PM2.5 exposure", ["exp_pm25_change_24h"]),
    "exposure_completeness": (
        "How complete your recent location record is",
        ["exp_pm25_coverage_24h"],
    ),
    "pm10_exposure": (
        "Estimated PM10 exposure",
        ["exp_pm10_mean_24h", "exp_pm10_max_24h", "exp_pm10_change_24h"],
    ),
    "o3_exposure": ("Estimated ozone exposure", ["exp_o3_mean_8h", "exp_o3_max_24h"]),
    "weather": (
        "Weather (temperature, humidity, rain)",
        ["wx_temperature", "wx_relative_humidity", "wx_temp_change_24h", "wx_precip_24h"],
    ),
    "time": (
        "Time of day and season",
        ["cal_hour_sin", "cal_hour_cos", "cal_month_sin", "cal_month_cos", "cal_is_weekend"],
    ),
    "condition": ("Your condition and its severity", ["pt_is_copd", "pt_severity"]),
    "demographics": ("Age and sex", ["pt_age", "pt_is_male"]),
    "symptoms": (
        "Recent symptom scores",
        ["hx_symptom_last", "hx_symptom_mean_3d", "hx_symptom_mean_7d", "hx_symptom_trend"],
    ),
    "reliever_use": ("Recent reliever-inhaler use", ["hx_rescue_last", "hx_rescue_mean_3d"]),
    "flare_ups": ("Recent flare-ups", ["hx_events_90d", "hx_days_since_event"]),
    "diary_logging": ("How regularly you logged your diary", ["hx_diary_days_7d"]),
    "wearable": (
        "Wearable readings (SpO₂, heart rate, steps, sleep)",
        ["hx_spo2_last", "hx_heart_rate_last", "hx_steps_mean_3d", "hx_sleep_last"],
    ),
}
FEATURE_TO_FACTOR = {f: key for key, (_, feats) in FACTORS.items() for f in feats}


# --- SHAP ---------------------------------------------------------------------------------


def explain_margin(
    model: SharedHeadRiskModel | PooledGBM, X: pd.DataFrame, disease: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeSHAP of the model's raw log-odds: (contributions n×f, base value n).

    For every row, contributions.sum() + base == the margin behind predict_proba."""
    names = model.feature_names
    if isinstance(model, PooledGBM):
        out = model.model.predict(X[names], pred_contrib=True)
        return out[:, :-1], out[:, -1]
    out = model.trunk.model.predict(X[names], pred_contrib=True)
    disease = np.asarray(disease)
    unknown = ~np.isin(disease, list(model.heads))
    if unknown.any():
        raise ValueError(f"no head for disease values {set(disease[unknown])}")
    for d, head in model.heads.items():
        m = disease == d
        if m.any() and model.head_iterations[d] > 0:
            out[m] += head.predict(
                X.loc[m, names], pred_contrib=True, num_iteration=model.head_iterations[d]
            )
    return out[:, :-1], out[:, -1]


def calibration_scale(calibrator: Calibrator) -> tuple[float, float] | None:
    """(a, b) with logit(calibrated) = a·logit(raw) + b, or None if not linear in log-odds."""
    if calibrator.method == "platt":
        return float(calibrator.params["a"]), float(calibrator.params["b"])
    if calibrator.method == "identity":
        return 1.0, 0.0
    return None


def input_fingerprint(features: dict[str, float | None]) -> str:
    """Hash of the stored feature vector, so an explanation can be matched to its input."""
    # Normalised so a JSONB round trip does not change the fingerprint: float() for 365 ↔ 365.0,
    # and + 0.0 because JSONB stores -0.0 as 0 (a rounded tiny negative becomes -0.0).
    values = {
        k: None if features.get(k) is None else float(features[k]) + 0.0 for k in RISK_FEATURE_NAMES
    }
    body = json.dumps(values, sort_keys=True)
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class Contribution:
    key: str
    label: str
    contribution: float  # log-odds of the reported probability (or raw, see `space`)
    features: dict[str, float]  # per-feature contributions inside this factor
    all_missing: bool  # every feature of the factor was missing in the input


def explain_risk_prediction(
    contributions: np.ndarray,
    base: float,
    feature_names: list[str],
    feature_values: dict[str, float | None],
    calibrator: Calibrator,
    personal_offset: float | None,
) -> tuple[str, float, list[Contribution]]:
    """One row's SHAP values → (space, base value, factor contributions incl. personal).

    Space is "calibrated_log_odds" when calibration is linear in log-odds (Platt), so the
    parts add up to the reported probability; otherwise "raw_log_odds"."""
    scale = calibration_scale(calibrator)
    a, b = scale if scale else (1.0, 0.0)
    space = "calibrated_log_odds" if scale else "raw_log_odds"
    per_feature = {n: a * float(c) for n, c in zip(feature_names, contributions, strict=True)}
    factors = []
    for key, (label, feats) in FACTORS.items():
        present = [f for f in feats if f in per_feature]
        if not present:
            continue
        factors.append(
            Contribution(
                key=key,
                label=label,
                contribution=sum(per_feature[f] for f in present),
                features={f: round(per_feature[f], 6) for f in present},
                all_missing=all(_missing(feature_values.get(f)) for f in present),
            )
        )
    if personal_offset and scale:
        factors.append(
            Contribution(
                key="personal_history",
                label="Your own flare-up history (personalisation)",
                contribution=float(personal_offset),
                features={},
                all_missing=False,
            )
        )
    return space, a * float(base) + b, factors


def _missing(v: float | None) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def rank_top_contributors(factors: list[Contribution], n: int = TOP_N) -> list[Contribution]:
    """Largest |contribution| first; ties broken by key so the order is deterministic."""
    ranked = sorted(factors, key=lambda c: (-abs(c.contribution), c.key))
    return [c for c in ranked if abs(c.contribution) >= NEGLIGIBLE_LOG_ODDS][:n]


def format_explanation_for_user(contributors: list[Contribution]) -> list[str]:
    """Plain sentences with no causal claim (PRD §17)."""
    lines = []
    for c in contributors:
        direction = "raised" if c.contribution > 0 else "lowered"
        note = " (not recorded recently)" if c.all_missing else ""
        lines.append(f"{c.label}{note} {direction} this estimate.")
    return lines


def build_explanation(
    *,
    contributions: np.ndarray,
    base: float,
    feature_names: list[str],
    features: dict[str, float | None],
    calibrator: Calibrator,
    personal_offset: float | None,
    model_version: str,
    feature_version: str,
    calibration_version: str,
    risk_probability: float,
) -> dict:
    """The stored, versioned explanation of one prediction (BR-09 consistency fields)."""
    space, base_value, factors = explain_risk_prediction(
        contributions, base, feature_names, features, calibrator, personal_offset
    )
    top = rank_top_contributors(factors)
    total = base_value + sum(c.contribution for c in factors)
    return {
        "version": EXPLANATION_VERSION,
        "method": "TreeSHAP (LightGBM pred_contrib), trunk + disease head",
        "space": space,
        "model_version": model_version,
        "feature_version": feature_version,
        "calibration_version": calibration_version,
        "input_fingerprint": input_fingerprint(features),
        "base_value": round(base_value, 6),
        # Only meaningful in calibrated space: sigmoid(base + Σ) == risk_probability.
        "reconstructed_probability": round(1 / (1 + math.exp(-total)), 8)
        if space == "calibrated_log_odds"
        else None,
        "risk_probability": round(risk_probability, 8),
        "top_factors": [_factor_out(c) for c in top],
        "summary": format_explanation_for_user(top),
        "factors": [
            _factor_out(c) for c in sorted(factors, key=lambda c: (-abs(c.contribution), c.key))
        ],
        "note": NOTE,
    }


def _factor_out(c: Contribution) -> dict:
    return {
        "key": c.key,
        "label": c.label,
        "contribution": round(c.contribution, 6),
        "direction": "neutral"
        if abs(c.contribution) < NEGLIGIBLE_LOG_ODDS
        else ("raises" if c.contribution > 0 else "lowers"),
        "recently_missing": c.all_missing,
        "features": c.features,
    }
