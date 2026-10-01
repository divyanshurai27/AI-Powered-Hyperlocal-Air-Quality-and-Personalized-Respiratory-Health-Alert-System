"""Calibration and risk metrics (PRD §15, §18)."""

import numpy as np
import pytest

from app.ml.calibration import (
    ISOTONIC_MIN_POSITIVES,
    brier_score,
    calibration_error,
    cluster_bootstrap,
    fit_calibration_model,
    fit_disease_calibrators,
    threshold_metrics,
    validate_risk_prediction,
)


def calibrated_sample(n: int = 200_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 20, n)
    y = (rng.random(n) < p).astype(int)
    return p, y


def test_brier_known_values() -> None:
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0
    assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == 0.25


def test_well_calibrated_scores_have_small_ece() -> None:
    p, y = calibrated_sample()
    assert calibration_error(p, y)["ece"] < 0.003


def test_overconfident_scores_have_large_ece() -> None:
    p, y = calibrated_sample()
    assert calibration_error(np.clip(p * 3, 0, 1), y)["ece"] > 0.05


def test_threshold_metrics_counts() -> None:
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 0, 1, 0])
    m = threshold_metrics(p, y, 0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert m["alert_rate"] == 0.5


def test_validate_reports_prd_metrics() -> None:
    p, y = calibrated_sample(20_000)
    out = validate_risk_prediction(p, y)
    for key in ("auroc", "auprc", "brier", "ece", "mce", "brier_skill", "prevalence"):
        assert key in out
    assert 0.5 < out["auroc"] <= 1


def test_isotonic_fixes_a_miscalibrated_score() -> None:
    p, y = calibrated_sample()
    raw = p**0.5  # monotone but badly calibrated
    cal = fit_calibration_model(raw, y, method="isotonic")
    assert cal.method == "isotonic"
    fixed = cal.transform(raw)
    assert calibration_error(fixed, y)["ece"] < calibration_error(raw, y)["ece"] / 5
    assert ((fixed > 0) & (fixed < 1)).all()  # BR-01, and strictly inside for the logit


def test_platt_is_the_default_and_strictly_increasing() -> None:
    """Isotonic is a step function and can map a whole low block to ~0; Platt never flattens."""
    p, y = calibrated_sample()
    raw = p**0.5
    platt = fit_calibration_model(raw, y)
    iso = fit_calibration_model(raw, y, method="isotonic")
    assert platt.method == "platt"
    grid = np.linspace(0.01, 0.3, 60)
    assert (np.diff(platt.transform(grid)) > 0).all()
    assert (np.diff(iso.transform(grid)) == 0).any()
    assert calibration_error(platt.transform(raw), y)["ece"] < calibration_error(raw, y)["ece"]


def test_isotonic_falls_back_to_platt_with_few_events() -> None:
    rng = np.random.default_rng(1)
    raw = rng.random(2000) * 0.1
    y = (rng.random(2000) < raw).astype(int)
    assert y.sum() < ISOTONIC_MIN_POSITIVES
    cal = fit_calibration_model(raw, y, method="isotonic")
    assert cal.method == "platt"
    out = cal.transform(np.array([0.0, 0.5, 1.0]))
    assert ((out > 0) & (out < 1)).all()
    assert out[0] < out[1] < out[2]


def test_single_class_falls_back_to_identity() -> None:
    cal = fit_calibration_model(np.array([0.1, 0.2]), np.array([0, 0]))
    assert cal.method == "identity"


def test_disease_calibrators_route_by_disease() -> None:
    p, y = calibrated_sample(40_000)
    disease = np.where(np.arange(len(p)) % 2 == 0, "asthma", "copd")
    cals = fit_disease_calibrators(p, y, disease)
    assert set(cals.by_disease) == {"asthma", "copd"}
    assert cals.calibrate(p[:10], disease[:10]).shape == (10,)
    with pytest.raises(ValueError):
        cals.calibrate(p[:2], np.array(["asthma", "other"]))


def test_cluster_bootstrap_brackets_the_estimate() -> None:
    p, y = calibrated_sample(20_000)
    groups = np.arange(len(p)) // 100
    ci = cluster_bootstrap(groups, lambda idx: brier_score(p[idx], y[idx]), n_boot=50)
    assert ci["lo"] <= brier_score(p, y) <= ci["hi"]
