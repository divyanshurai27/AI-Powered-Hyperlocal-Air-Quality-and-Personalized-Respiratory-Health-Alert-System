"""Respiratory-risk models and versioned artifacts (PRD §15.2, §39).

Proposed model: a shared trunk with one head per disease (PRD §15.2):

    shared trunk   LightGBM on ALL patients (disease is just a feature) → margin m(x)
    asthma head    LightGBM boosted on asthma rows, starting from m(x) (init_score)
    COPD head      same on COPD rows
    p(event in next 24 h) = sigmoid(m(x) + head_d(x))

The trunk learns what is common (pollution, weather, symptoms); each head only learns the
correction its disease needs, and early stopping on validation keeps a head near zero when
there is nothing disease-specific to add. Baselines use the same interface:

    stratum_prior     base rate of the patient's disease × severity stratum (no features)
    logistic          standardised logistic regression with missing-value indicators
    lightgbm_pooled   the trunk alone
    lightgbm_separate one independent LightGBM per disease

Outputs here are RAW probabilities; calibration (app/ml/calibration.py) is applied after.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.calibration import DiseaseCalibrators
from app.ml.risk_features import RISK_FEATURE_NAMES, RISK_FEATURE_VERSION


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@dataclass
class RiskGBMConfig:
    n_estimators: int = 3000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 300
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 100
    random_state: int = 42
    # heads: small and slow, so they can only add a modest correction
    head_n_estimators: int = 800
    head_learning_rate: float = 0.01
    head_num_leaves: int = 7


def _lgbm(cfg: RiskGBMConfig, head: bool = False):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg.head_n_estimators if head else cfg.n_estimators,
        learning_rate=cfg.head_learning_rate if head else cfg.learning_rate,
        num_leaves=cfg.head_num_leaves if head else cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=cfg.subsample_freq,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        verbose=-1,
    )


def _early_stop(cfg: RiskGBMConfig):
    import lightgbm as lgb

    return [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]


class StratumPrior:
    name = "stratum_prior"

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or ["pt_is_copd", "pt_severity"]
        self.rates: dict[tuple, float] = {}
        self.overall = 0.0

    def _keys(self, X: pd.DataFrame) -> list[tuple]:
        return list(zip(X["pt_is_copd"].to_numpy(), X["pt_severity"].to_numpy(), strict=True))

    def fit(self, X, y, disease, X_val=None, y_val=None, disease_val=None) -> "StratumPrior":
        df = pd.DataFrame({"k": self._keys(X), "y": np.asarray(y)})
        self.rates = df.groupby("k")["y"].mean().to_dict()
        self.overall = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame, disease) -> np.ndarray:
        return np.array([self.rates.get(k, self.overall) for k in self._keys(X)])


class LogisticRisk:
    name = "logistic"

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.pipeline = None

    def fit(self, X, y, disease, X_val=None, y_val=None, disease_val=None) -> "LogisticRisk":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000),
        )
        self.pipeline.fit(X[self.feature_names], np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame, disease) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.feature_names])[:, 1]


class PooledGBM:
    name = "lightgbm_pooled"

    def __init__(self, feature_names: list[str], config: RiskGBMConfig | None = None) -> None:
        self.feature_names = feature_names
        self.config = config or RiskGBMConfig()
        self.model = None

    def fit(self, X, y, disease, X_val, y_val, disease_val) -> "PooledGBM":
        self.model = _lgbm(self.config)
        self.model.fit(
            X[self.feature_names],
            np.asarray(y),
            eval_X=X_val[self.feature_names],
            eval_y=np.asarray(y_val),
            callbacks=_early_stop(self.config),
        )
        return self

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names], raw_score=True)

    def predict_proba(self, X: pd.DataFrame, disease=None) -> np.ndarray:
        return _sigmoid(self.margin(X))


class SeparateGBM:
    name = "lightgbm_separate"

    def __init__(self, feature_names: list[str], config: RiskGBMConfig | None = None) -> None:
        self.feature_names = feature_names
        self.config = config or RiskGBMConfig()
        self.models: dict[str, PooledGBM] = {}

    def fit(self, X, y, disease, X_val, y_val, disease_val) -> "SeparateGBM":
        y, y_val = np.asarray(y), np.asarray(y_val)
        disease, disease_val = np.asarray(disease), np.asarray(disease_val)
        for d in np.unique(disease):
            m, mv = disease == d, disease_val == d
            self.models[str(d)] = PooledGBM(self.feature_names, self.config).fit(
                X[m], y[m], None, X_val[mv], y_val[mv], None
            )
        return self

    def predict_proba(self, X: pd.DataFrame, disease) -> np.ndarray:
        disease = np.asarray(disease)
        out = np.full(len(X), np.nan)
        for d, model in self.models.items():
            m = disease == d
            if m.any():
                out[m] = model.predict_proba(X[m])
        return out


class SharedHeadRiskModel:
    """Shared trunk + per-disease heads (PRD §15.2). The proposed model."""

    name = "lightgbm_shared_heads"

    def __init__(self, feature_names: list[str], config: RiskGBMConfig | None = None) -> None:
        self.feature_names = feature_names
        self.config = config or RiskGBMConfig()
        self.trunk = PooledGBM(feature_names, self.config)
        self.heads: dict[str, Any] = {}
        self.head_iterations: dict[str, int] = {}

    def fit(self, X, y, disease, X_val, y_val, disease_val) -> "SharedHeadRiskModel":
        y, y_val = np.asarray(y), np.asarray(y_val)
        disease, disease_val = np.asarray(disease), np.asarray(disease_val)
        self.trunk.fit(X, y, disease, X_val, y_val, disease_val)
        base, base_val = self.trunk.margin(X), self.trunk.margin(X_val)
        for d in np.unique(disease):
            m, mv = disease == d, disease_val == d
            head = _lgbm(self.config, head=True)
            head.fit(
                X.loc[m, self.feature_names],
                y[m],
                init_score=base[m],
                eval_X=X_val.loc[mv, self.feature_names],
                eval_y=y_val[mv],
                eval_init_score=[base_val[mv]],
                callbacks=_early_stop(self.config),
            )
            self.heads[str(d)] = head
            self.head_iterations[str(d)] = int(head.best_iteration_ or 0)
        return self

    def predict_proba(self, X: pd.DataFrame, disease) -> np.ndarray:
        disease = np.asarray(disease)
        margin = self.trunk.margin(X)
        for d, head in self.heads.items():
            m = disease == d
            if m.any() and self.head_iterations[d] > 0:
                margin[m] += head.predict(
                    X.loc[m, self.feature_names],
                    raw_score=True,
                    num_iteration=self.head_iterations[d],
                )
        unknown = ~np.isin(disease, list(self.heads))
        if unknown.any():
            raise ValueError(f"no head for disease values {set(disease[unknown])}")
        return _sigmoid(margin)


# --- artifacts ---------------------------------------------------------------------------


class RiskArtifactError(RuntimeError):
    pass


@dataclass
class RiskArtifactMetadata:
    model_name: str
    model_version: str
    feature_version: str
    feature_names: list[str]
    target: str
    cohort_version: str
    cohort_config: dict[str, Any]
    dataset_id: str
    splits: dict[str, Any]
    hyperparameters: dict[str, Any]
    head_iterations: dict[str, int]
    # daily event rate per "disease|severity" stratum in the training period, used by the
    # personal baseline (domain/risk_policy.estimate_personal_baseline)
    stratum_daily_rates: dict[str, float]
    calibration_version: str
    calibration_methods: dict[str, str]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    synthetic_labels: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_sha256: str = ""
    calibration_sha256: str = ""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_risk_artifact(
    model: SharedHeadRiskModel,
    calibrators: DiseaseCalibrators,
    metadata: RiskArtifactMetadata,
    directory: Path,
) -> Path:
    """<dir>/model.joblib, calibration.joblib and metadata.json with both checksums."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    joblib.dump(calibrators, directory / "calibration.joblib")
    metadata.model_sha256 = _sha(directory / "model.joblib")
    metadata.calibration_sha256 = _sha(directory / "calibration.joblib")
    (directory / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, default=str))
    return directory


def load_risk_artifact(
    directory: Path,
) -> tuple[SharedHeadRiskModel, DiseaseCalibrators, RiskArtifactMetadata]:
    """Load and verify checksums, the feature contract and the calibration version."""
    meta = RiskArtifactMetadata(**json.loads((directory / "metadata.json").read_text()))
    if _sha(directory / "model.joblib") != meta.model_sha256:
        raise RiskArtifactError(f"{directory}: model checksum does not match metadata")
    if _sha(directory / "calibration.joblib") != meta.calibration_sha256:
        raise RiskArtifactError(f"{directory}: calibration checksum does not match metadata")
    if meta.feature_version != RISK_FEATURE_VERSION:
        raise RiskArtifactError(
            f"{directory}: trained on {meta.feature_version}, code builds {RISK_FEATURE_VERSION}"
        )
    if meta.feature_names != RISK_FEATURE_NAMES:
        raise RiskArtifactError(f"{directory}: feature list differs from the current contract")
    model = joblib.load(directory / "model.joblib")  # noqa: S301  (own artifact, verified)
    calibrators = joblib.load(directory / "calibration.joblib")  # noqa: S301
    if calibrators.version != meta.calibration_version:
        raise RiskArtifactError(f"{directory}: calibration version differs from metadata")
    return model, calibrators, meta


def matrix_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes())
    h.update(np.asarray(y).tobytes())
    return "risk_" + h.hexdigest()[:16]
