"""Explanations (PRD §17, BR-09): exact additivity, consistency with the reported probability,
determinism and non-causal wording."""

import math

import numpy as np
import pytest

from app.domain.risk_policy import PersonalBaseline, apply_personal_offset
from app.ml.calibration import Calibrator, fit_disease_calibrators
from app.ml.explain import (
    FACTORS,
    build_explanation,
    explain_margin,
    format_explanation_for_user,
    input_fingerprint,
    rank_top_contributors,
)
from app.ml.risk import PooledGBM, SharedHeadRiskModel
from app.ml.risk_features import RISK_FEATURE_NAMES
from tests.unit.test_risk_model import FAST, toy


@pytest.fixture(scope="module")
def fitted():
    X, y, d = toy(3000)
    Xv, yv, dv = toy(1500, seed=1)
    model = SharedHeadRiskModel(RISK_FEATURE_NAMES, FAST).fit(X, y, d, Xv, yv, dv)
    cals = fit_disease_calibrators(model.predict_proba(Xv, dv), yv, dv)
    return model, cals, Xv.iloc[:50].reset_index(drop=True), dv[:50]


def _logit(p):
    return np.log(p / (1 - p))


def test_every_feature_belongs_to_exactly_one_factor() -> None:
    listed = [f for _, feats in FACTORS.values() for f in feats]
    assert sorted(listed) == sorted(RISK_FEATURE_NAMES)
    assert len(listed) == len(set(listed))


@pytest.mark.parametrize("disease", ["asthma", "copd"])
def test_shap_adds_up_to_the_model_margin(fitted, disease) -> None:
    model, _, X, _ = fitted
    d = np.full(len(X), disease)
    contrib, base = explain_margin(model, X, d)
    assert contrib.shape == (len(X), len(RISK_FEATURE_NAMES))
    np.testing.assert_allclose(
        contrib.sum(1) + base, _logit(model.predict_proba(X, d)), rtol=0, atol=1e-9
    )


def test_pooled_model_is_explained_too(fitted) -> None:
    X, y, d = toy(2000, seed=3)
    pooled = PooledGBM(RISK_FEATURE_NAMES, FAST).fit(X, y, d, X, y, d)
    contrib, base = explain_margin(pooled, X)
    np.testing.assert_allclose(contrib.sum(1) + base, pooled.margin(X), atol=1e-9)


def test_unknown_disease_is_rejected(fitted) -> None:
    model, _, X, _ = fitted
    with pytest.raises(ValueError, match="no head"):
        explain_margin(model, X, np.full(len(X), "bronchitis"))


def _explain(fitted, i=0, offset=None, calibrator=None):
    model, cals, X, d = fitted
    contrib, base = explain_margin(model, X, d)
    disease = str(d[i])
    cal = calibrator or cals.by_disease[disease]
    p_cal = float(cal.transform(model.predict_proba(X.iloc[[i]], d[[i]]))[0])
    baseline = PersonalBaseline(0, 0, 0, offset) if offset else None
    p = apply_personal_offset(p_cal, baseline)
    features = {k: (None if math.isnan(v) else float(v)) for k, v in X.iloc[i].items()}
    exp = build_explanation(
        contributions=contrib[i],
        base=float(base[i]),
        feature_names=model.feature_names,
        features=features,
        calibrator=cal,
        personal_offset=offset,
        model_version="risk_vtest",
        feature_version="f",
        calibration_version="c",
        risk_probability=p,
    )
    return exp, p


@pytest.mark.parametrize("offset", [None, 0.4, -0.6])
def test_explanation_reconstructs_the_reported_probability(fitted, offset) -> None:
    for i in range(10):
        exp, p = _explain(fitted, i, offset)
        assert exp["space"] == "calibrated_log_odds"
        assert exp["reconstructed_probability"] == pytest.approx(p, abs=1e-6)
        total = exp["base_value"] + sum(f["contribution"] for f in exp["factors"])
        assert 1 / (1 + math.exp(-total)) == pytest.approx(p, abs=1e-5)
        keys = {f["key"] for f in exp["factors"]}
        assert ("personal_history" in keys) == bool(offset)


def test_same_prediction_gives_the_same_explanation(fitted) -> None:
    a, _ = _explain(fitted, 3, 0.2)
    b, _ = _explain(fitted, 3, 0.2)
    assert a == b


def test_non_linear_calibration_reports_raw_space(fitted) -> None:
    iso = Calibrator("isotonic", model=type("M", (), {"predict": lambda self, r: r})())
    exp, _ = _explain(fitted, 0, 0.3, calibrator=iso)
    assert exp["space"] == "raw_log_odds"
    assert exp["reconstructed_probability"] is None
    assert all(f["key"] != "personal_history" for f in exp["factors"])


def test_ranking_and_wording(fitted) -> None:
    exp, _ = _explain(fitted, 5)
    top = exp["top_factors"]
    assert 1 <= len(top) <= 5
    mags = [abs(f["contribution"]) for f in top]
    assert mags == sorted(mags, reverse=True)
    assert len(exp["summary"]) == len(top)
    text = " ".join(exp["summary"]).lower()
    for causal in ("caused", "because of", "due to", "inhaler dose", "medication"):
        assert causal not in text


def test_ties_and_negligible_factors() -> None:
    from app.ml.explain import Contribution

    cs = [
        Contribution("b", "B", 0.5, {}, False),
        Contribution("a", "A", -0.5, {}, False),
        Contribution("z", "Z", 0.001, {}, False),
    ]
    assert [c.key for c in rank_top_contributors(cs)] == ["a", "b"]
    lines = format_explanation_for_user(rank_top_contributors(cs))
    assert lines == ["A lowered this estimate.", "B raised this estimate."]


def test_fingerprint_ignores_json_int_float_round_trip() -> None:
    a = dict.fromkeys(RISK_FEATURE_NAMES, 365.0) | {"hx_symptom_last": None}
    b = dict.fromkeys(RISK_FEATURE_NAMES, 365) | {"hx_symptom_last": None}
    assert input_fingerprint(a) == input_fingerprint(b)
    # JSONB stores -0.0 as 0
    assert input_fingerprint(a | {"pt_age": -0.0}) == input_fingerprint(a | {"pt_age": 0.0})
    assert input_fingerprint(a) != input_fingerprint(a | {"pt_age": 1.0})
