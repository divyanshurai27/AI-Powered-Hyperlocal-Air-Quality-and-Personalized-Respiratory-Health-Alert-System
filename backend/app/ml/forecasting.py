"""24-hour AQ forecasters, evaluation and versioned artifacts (PRD §12, §39).

Strategy: *direct* multi-horizon, one regressor per horizon h ∈ 1..24. Every horizon sees
only features at t, so there's no recursive error feedback and no chance of feeding a
forecast back in as if it were an observation (PRD §3.1).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_VERSION, HORIZONS

HORIZON_COLUMNS = [f"h{h}" for h in HORIZONS]


class Forecaster(Protocol):
    name: str

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...


class PersistenceForecaster:
    """ŷ(t+h) = y(t) for every h. The baseline every model must beat."""

    name = "persistence"

    def __init__(self, current_column: str) -> None:
        self.current_column = current_column

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        cur = X[self.current_column].to_numpy()
        return pd.DataFrame(
            np.repeat(cur[:, None], len(HORIZONS), axis=1), index=X.index, columns=HORIZON_COLUMNS
        )


class SeasonalNaiveForecaster:
    """ŷ(t+h) = y(t+h-24): same hour yesterday. Captures the daily cycle for free."""

    name = "seasonal_naive"

    def predict(self, seasonal: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        return seasonal[HORIZON_COLUMNS]


@dataclass
class GBMConfig:
    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42


class DirectGBMForecaster:
    """One LightGBM per horizon, trained on log1p(y) to tame the right-skew of pollution data."""

    name = "lightgbm_direct"

    def __init__(self, feature_names: list[str], config: GBMConfig | None = None) -> None:
        self.feature_names = feature_names
        self.config = config or GBMConfig()
        self.models: dict[int, Any] = {}
        self.best_iterations: dict[int, int] = {}

    def fit(
        self, X: pd.DataFrame, Y: pd.DataFrame, X_val: pd.DataFrame, Y_val: pd.DataFrame
    ) -> None:
        import lightgbm as lgb

        cfg = asdict(self.config)
        stopping = cfg.pop("early_stopping_rounds")
        for h in HORIZONS:
            col = f"h{h}"
            tr, va = Y[col].notna(), Y_val[col].notna()
            model = lgb.LGBMRegressor(objective="regression", verbose=-1, **cfg)
            model.fit(
                X.loc[tr, self.feature_names],
                np.log1p(Y.loc[tr, col]),
                eval_X=X_val.loc[va, self.feature_names],
                eval_y=np.log1p(Y_val.loc[va, col]),
                callbacks=[lgb.early_stopping(stopping, verbose=False)],
            )
            self.models[h] = model
            self.best_iterations[h] = int(model.best_iteration_ or cfg["n_estimators"])

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if set(self.models) != set(HORIZONS):
            raise RuntimeError("model is not trained for every horizon")
        feats = X[self.feature_names]
        preds = {f"h{h}": np.expm1(self.models[h].predict(feats)).clip(min=0) for h in HORIZONS}
        return pd.DataFrame(preds, index=X.index)[HORIZON_COLUMNS]


# --- evaluation --------------------------------------------------------------------------


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    err = pred - obs
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
        "n": int(obs.size),
    }


def validate_forecast(predictions: pd.DataFrame, observations: pd.DataFrame) -> dict[str, Any]:
    """MAE/RMSE/R² per horizon and pooled, on pairs where BOTH forecast and observation exist.

    Scored only against observed values (never imputed ones, PRD §3.1). Pairs where a
    baseline can't forecast (e.g. no reading 24 h earlier) are excluded for that model, so
    the report also states `n` per horizon.
    """
    per_h = {}
    all_p, all_o = [], []
    for col in HORIZON_COLUMNS:
        mask = predictions[col].notna() & observations[col].notna()
        p, o = predictions.loc[mask, col].to_numpy(), observations.loc[mask, col].to_numpy()
        if o.size:
            per_h[col] = _metrics(p, o)
            all_p.append(p)
            all_o.append(o)
    pooled = _metrics(np.concatenate(all_p), np.concatenate(all_o)) if all_o else {}
    return {"overall": pooled, "per_horizon": per_h}


def compare_forecast_models(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Structured comparison table, sorted by pooled MAE, with skill vs persistence.

    Skill = 1 - MAE_model / MAE_persistence (>0 means better than persistence).
    """
    base = results.get("persistence", {}).get("overall", {}).get("mae")
    rows = []
    for name, res in results.items():
        o = res["overall"]
        rows.append(
            {
                "model": name,
                "mae": round(o["mae"], 3),
                "rmse": round(o["rmse"], 3),
                "r2": round(o["r2"], 4),
                "bias": round(o["bias"], 3),
                "n": o["n"],
                "mae_h1": round(res["per_horizon"]["h1"]["mae"], 3),
                "mae_h6": round(res["per_horizon"]["h6"]["mae"], 3),
                "mae_h24": round(res["per_horizon"]["h24"]["mae"], 3),
                "skill_vs_persistence": round(1 - o["mae"] / base, 4) if base else None,
            }
        )
    return sorted(rows, key=lambda r: r["mae"])


# --- artifacts ---------------------------------------------------------------------------


class ArtifactContractError(RuntimeError):
    pass


@dataclass
class ForecastArtifactMetadata:
    model_name: str
    model_version: str
    pollutant: str
    feature_version: str
    feature_names: list[str]
    horizons: list[int]
    training_dataset: str
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    hyperparameters: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    target_transform: str = "log1p"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_sha256: str = ""


def save_model_artifact(
    model: DirectGBMForecaster, metadata: ForecastArtifactMetadata, directory: Path
) -> Path:
    """Write <dir>/model.joblib + <dir>/metadata.json. The sidecar carries the model's SHA-256
    so the API can prove it serves exactly the artifact that was evaluated (PRD §38)."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)
    metadata.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (directory / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, default=str))
    return directory


def load_model_artifact(
    directory: Path, expected_feature_names: list[str]
) -> tuple[DirectGBMForecaster, ForecastArtifactMetadata]:
    """Load and verify: checksum, feature version, feature list and horizons must all match."""
    meta = ForecastArtifactMetadata(**json.loads((directory / "metadata.json").read_text()))
    model_path = directory / "model.joblib"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != meta.model_sha256:
        raise ArtifactContractError(f"{directory}: model file checksum does not match metadata")
    if meta.feature_version != FEATURE_VERSION:
        raise ArtifactContractError(
            f"{directory}: trained on {meta.feature_version}, code builds {FEATURE_VERSION}"
        )
    if meta.feature_names != expected_feature_names:
        raise ArtifactContractError(f"{directory}: feature list differs from the current contract")
    if meta.horizons != list(HORIZONS):
        raise ArtifactContractError(f"{directory}: horizons {meta.horizons} != {list(HORIZONS)}")
    model = joblib.load(model_path)  # noqa: S301  (our own artifact, checksum-verified above)
    return model, meta


def dataset_fingerprint(observations: pd.DataFrame) -> str:
    """Deterministic ID of a dataset snapshot: same rows → same ID, any change → new ID."""
    cols = ["station_id", "pollutant", "timestamp", "concentration", "quality_flag"]
    canon = observations[cols].sort_values(cols[:3]).to_csv(index=False).encode()
    return "aq_" + hashlib.sha256(canon).hexdigest()[:16]
