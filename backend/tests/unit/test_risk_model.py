"""Risk models (BR-01 range, shared-head behaviour) and artifact contract (BR-08, PRD §39)."""

import json

import numpy as np
import pandas as pd
import pytest

from app.ml.calibration import fit_disease_calibrators
from app.ml.risk import (
    LogisticRisk,
    PooledGBM,
    RiskArtifactError,
    RiskArtifactMetadata,
    RiskGBMConfig,
    SeparateGBM,
    SharedHeadRiskModel,
    StratumPrior,
    load_risk_artifact,
    save_risk_artifact,
)
from app.ml.risk_features import RISK_FEATURE_NAMES, RISK_FEATURE_VERSION

FAST = RiskGBMConfig(
    n_estimators=150, min_child_samples=20, early_stopping_rounds=20, head_n_estimators=100
)


def toy(n: int = 6000, seed: int = 0):
    """Risk rises with symptoms for everyone and with PM2.5 only for COPD."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(RISK_FEATURE_NAMES))), columns=RISK_FEATURE_NAMES)
    copd = rng.random(n) < 0.4
    X["pt_is_copd"] = copd.astype(float)
    X["pt_severity"] = rng.integers(0, 4, n).astype(float)
    X.loc[rng.random(n) < 0.2, "hx_symptom_last"] = np.nan  # missing is allowed
    logit = (
        -3 + 1.2 * X["hx_symptom_last"].fillna(0) + np.where(copd, 1.5, 0) * X["exp_pm25_mean_24h"]
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    disease = np.where(copd, "copd", "asthma")
    return X, pd.Series(y), disease


@pytest.fixture(scope="module")
def data():
    X, y, d = toy()
    Xv, yv, dv = toy(3000, seed=1)
    return X, y, d, Xv, yv, dv


@pytest.fixture(scope="module")
def shared(data):
    return SharedHeadRiskModel(RISK_FEATURE_NAMES, FAST).fit(*data)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StratumPrior(),
        lambda: LogisticRisk(RISK_FEATURE_NAMES),
        lambda: PooledGBM(RISK_FEATURE_NAMES, FAST),
        lambda: SeparateGBM(RISK_FEATURE_NAMES, FAST),
    ],
)
def test_every_model_returns_probabilities(data, factory) -> None:
    X, y, d, Xv, yv, dv = data
    p = factory().fit(X, y, d, Xv, yv, dv).predict_proba(Xv, dv)
    assert p.shape == (len(Xv),)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_shared_heads_learn_the_disease_specific_effect(data, shared) -> None:
    _, _, _, Xv, _, _ = data
    lo, hi = Xv.iloc[:200].copy(), Xv.iloc[:200].copy()
    lo["exp_pm25_mean_24h"], hi["exp_pm25_mean_24h"] = -1.5, 1.5
    copd = np.full(200, "copd")
    asthma = np.full(200, "asthma")
    lo_c, hi_c = lo.assign(pt_is_copd=1.0), hi.assign(pt_is_copd=1.0)
    lo_a, hi_a = lo.assign(pt_is_copd=0.0), hi.assign(pt_is_copd=0.0)
    copd_gain = (shared.predict_proba(hi_c, copd) - shared.predict_proba(lo_c, copd)).mean()
    asthma_gain = (shared.predict_proba(hi_a, asthma) - shared.predict_proba(lo_a, asthma)).mean()
    assert copd_gain > 0.1
    assert copd_gain > 3 * abs(asthma_gain)


def test_unknown_disease_is_refused(data, shared) -> None:
    _, _, _, Xv, _, _ = data
    with pytest.raises(ValueError):
        shared.predict_proba(Xv.iloc[:3], np.array(["asthma", "copd", "flu"]))


def _meta(**over) -> RiskArtifactMetadata:
    base = dict(
        model_name="lightgbm_shared_heads",
        model_version="risk_v0.0.1",
        feature_version=RISK_FEATURE_VERSION,
        feature_names=list(RISK_FEATURE_NAMES),
        target="event_onset_next_24h",
        cohort_version="cohort_test",
        cohort_config={},
        dataset_id="risk_test",
        splits={},
        hyperparameters={},
        head_iterations={},
        stratum_daily_rates={"asthma|mild": 0.01},
        calibration_version="calibration_v1.0.0",
        calibration_methods={},
        validation_metrics={},
        test_metrics={},
    )
    return RiskArtifactMetadata(**(base | over))


def test_artifact_round_trip_and_tamper_detection(tmp_path, data, shared) -> None:
    X, y, d, Xv, yv, dv = data
    cals = fit_disease_calibrators(shared.predict_proba(Xv, dv), yv, dv)
    save_risk_artifact(shared, cals, _meta(), tmp_path)
    model, loaded_cals, meta = load_risk_artifact(tmp_path)
    np.testing.assert_allclose(model.predict_proba(Xv, dv), shared.predict_proba(Xv, dv))
    assert meta.synthetic_labels is True
    assert loaded_cals.version == meta.calibration_version

    (tmp_path / "calibration.joblib").write_bytes(b"tampered")
    with pytest.raises(RiskArtifactError, match="calibration checksum"):
        load_risk_artifact(tmp_path)


def test_artifact_with_another_feature_contract_is_refused(tmp_path, data, shared) -> None:
    X, y, d, Xv, yv, dv = data
    cals = fit_disease_calibrators(shared.predict_proba(Xv, dv), yv, dv)
    save_risk_artifact(shared, cals, _meta(), tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["feature_version"] = "risk_features_v0"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RiskArtifactError, match="trained on"):
        load_risk_artifact(tmp_path)
