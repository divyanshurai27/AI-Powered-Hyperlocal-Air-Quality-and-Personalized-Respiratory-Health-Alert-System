"""Probability calibration and classification metrics for the risk model (PRD §15, §18).

Calibration is fitted on the validation period only and is versioned separately from the
model (CALIBRATION_VERSION), so the model and the mapping to reported probabilities can be
audited independently. One calibrator per disease head, because the heads' raw scores are
on different scales.

Method: Platt (logistic) scaling for every disease (v1.1.0). v1.0.0 used isotonic regression
whenever validation had ≥ ISOTONIC_MIN_POSITIVES events. Isotonic's lowest block can hold zero
events, so it reported a risk of ~0 for 6% of asthma and 4% of COPD test predictions, and 25
events then happened among those "zero-risk" moments, with no Brier gain over Platt. An
exact-zero probability is not credible for a person with a ~1.5% daily base rate, so Platt
(monotone, never 0 or 1) became the rule. Isotonic remains available for comparison.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

CALIBRATION_VERSION = "calibration_v1.1.0"
ISOTONIC_MIN_POSITIVES = 200
ECE_BINS = 10
EPS = 1e-6


# --- metrics ------------------------------------------------------------------------------


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> list[dict]:
    """Equal-frequency bins (quantiles of p): with rare events, equal-width bins above ~0.2
    would be nearly empty. Each row: mean predicted, observed rate, count."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.size == 0:
        return []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = []
    for b in range(max(len(edges) - 1, 1)):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "mean_predicted": float(p[m].mean()),
                    "observed_rate": float(y[m].mean()),
                    "n": int(m.sum()),
                }
            )
    return rows


def calibration_error(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> dict[str, float]:
    """ECE = count-weighted mean |observed − predicted| over bins; MCE = the worst bin."""
    rows = reliability_table(p, y, bins)
    n = sum(r["n"] for r in rows)
    if not n:
        return {"ece": float("nan"), "mce": float("nan")}
    gaps = [abs(r["observed_rate"] - r["mean_predicted"]) for r in rows]
    return {
        "ece": float(sum(g * r["n"] for g, r in zip(gaps, rows, strict=True)) / n),
        "mce": float(max(gaps)),
    }


def threshold_metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    alert = p >= threshold
    tp = int((alert & y).sum())
    fp = int((alert & ~y).sum())
    fn = int((~alert & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": float(alert.mean()) if alert.size else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def best_f1_threshold(p: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    grid = grid if grid is not None else np.unique(np.quantile(p, np.linspace(0.5, 0.999, 300)))
    scores = [threshold_metrics(p, y, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])


def precision_threshold(p: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """Lowest threshold whose precision reaches `target` (None if never reached)."""
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 400))):
        m = threshold_metrics(p, y, t)
        if m["tp"] >= 5 and m["precision"] >= target:
            return float(t)
    return None


def validate_risk_prediction(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """PRD §15 metric set for a probability vector (threshold metrics are reported by the
    caller at the policy thresholds)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    if p.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(p.size),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "brier": brier_score(p, y),
    }
    both = 0 < y.sum() < y.size
    out["auroc"] = float(roc_auc_score(y, p)) if both else float("nan")
    out["auprc"] = float(average_precision_score(y, p)) if both else float("nan")
    # Brier skill vs always predicting the observed prevalence (>0 = better than the base rate).
    ref = brier_score(np.full_like(p, y.mean()), y)
    out["brier_skill"] = 1 - out["brier"] / ref if ref > 0 else float("nan")
    out |= calibration_error(p, y)
    return out


def cluster_bootstrap(
    groups: np.ndarray,
    stat,
    n_boot: int = 200,
    seed: int = 7,
) -> dict[str, float]:
    """Percentile 95% CI of `stat(index_array)` resampling whole patients, since rows from
    the same patient are correlated. `stat` receives row indices into the original arrays."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[g] for g in pick])
        v = stat(idx)
        if np.isfinite(v):
            values.append(v)
    if not values:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.percentile(values, 2.5)), "hi": float(np.percentile(values, 97.5))}


# --- calibrators ---------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    method: str  # "isotonic" | "platt" | "identity"
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, float)
        if self.method == "identity":
            out = raw
        elif self.method == "platt":
            out = 1 / (1 + np.exp(-(self.params["a"] * _logit(raw) + self.params["b"])))
        else:
            out = self.model.predict(raw)
        # Isotonic can return exact 0/1 at the extremes; keep strictly inside (0, 1) so the
        # personal logit offset stays defined. Still satisfies BR-01.
        return np.clip(out, EPS, 1 - EPS)


def fit_calibration_model(raw: np.ndarray, y: np.ndarray, method: str = "platt") -> Calibrator:
    """method="platt" (the rule) or "isotonic" (comparison only; needs enough events)."""
    raw, y = np.asarray(raw, float), np.asarray(y, int)
    if y.sum() == 0 or y.sum() == y.size:
        return Calibrator("identity")
    if method == "isotonic" and y.sum() >= ISOTONIC_MIN_POSITIVES:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw, y)
        return Calibrator("isotonic", {"n": int(y.size), "positives": int(y.sum())}, iso)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(raw).reshape(-1, 1), y)
    return Calibrator(
        "platt",
        {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "n": int(y.size)},
    )


@dataclass
class DiseaseCalibrators:
    """One calibrator per disease head (keys are DiseaseType values)."""

    by_disease: dict[str, Calibrator]
    version: str = CALIBRATION_VERSION

    def calibrate(self, raw: np.ndarray, disease: np.ndarray) -> np.ndarray:
        raw, disease = np.asarray(raw, float), np.asarray(disease)
        out = np.full(raw.shape, np.nan)
        for d, cal in self.by_disease.items():
            m = disease == d
            if m.any():
                out[m] = cal.transform(raw[m])
        if np.isnan(out).any():
            raise ValueError("no calibrator for some disease values")
        return out


def fit_disease_calibrators(
    raw: np.ndarray, y: np.ndarray, disease: np.ndarray
) -> DiseaseCalibrators:
    raw, y, disease = np.asarray(raw, float), np.asarray(y, int), np.asarray(disease)
    return DiseaseCalibrators(
        {
            str(d): fit_calibration_model(raw[disease == d], y[disease == d])
            for d in np.unique(disease)
        }
    )
