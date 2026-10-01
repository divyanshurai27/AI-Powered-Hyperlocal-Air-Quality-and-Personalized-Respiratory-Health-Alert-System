"""Supervised risk dataset from the synthetic cohort (labels, origins, exposure scenarios).

Label (the risk target): y = 1 if a respiratory event STARTS in (T, T+24 h]. It is taken
from the cohort's true onsets, never from anything the features can see.

Origins: two random whole hours per patient-day (seeded), so every hour of the day is
represented and the model can be applied at any hour, as the rolling 24 h forecast does.
Excluded origins:
  - the first 10 days (the longest health lookback, symptom trend, needs 240 h)
  - the last 24 h (the label window would run past the data)
  - origins inside an ongoing episode (an onset in the previous `episode_hours`): the
    person is already having an event, so "risk of a new one" is not the question.

Exposure scenarios (the exposure sensitivity analysis deferred from Phase 3):
  estimated     what production computes: live-rule ambient at the logged place × the
                microenvironment factor table (unknown context → 1.0)
  factors×k     the same with every non-outdoor factor multiplied by k
  home_ambient  ambient at the home address only: no mobility, no microenvironment
  true          the simulated TRUE personal exposure (an oracle upper bound; unavailable in
                reality)
"""

import numpy as np
import pandas as pd

from app.domain.exposure import MICROENVIRONMENT_FACTORS
from app.ml.cohort import COHORT_POLLUTANTS, ENV_CODES, Cohort
from app.ml.risk_features import build_risk_features

WARMUP_HOURS = 240
LABEL_HOURS = 24


def patient_profile(cohort: Cohort, i: int) -> dict:
    p = cohort.patients.iloc[i]
    return {
        "age": int(p.age),
        "sex": p.sex,
        "disease_type": p.disease_type,
        "severity": p.severity,
    }


def patient_exposure(
    cohort: Cohort, i: int, mode: str = "estimated", factors: dict | None = None
) -> pd.DataFrame:
    table = factors or MICROENVIRONMENT_FACTORS
    out = {}
    for pol in COHORT_POLLUTANTS:
        if mode == "true":
            out[pol.value] = cohort.true_exposure[pol.value][i]
        elif mode == "home_ambient":
            out[pol.value] = cohort.amb_est_home[pol.value][i]
        else:
            f = np.array([table[e][pol] for e in ENV_CODES])
            out[pol.value] = cohort.amb_est[pol.value][i] * f[cohort.env_reported[i]]
    return pd.DataFrame(out, index=cohort.hours).astype(float)


def patient_health(cohort: Cohort, i: int) -> pd.DataFrame:
    d = cohort.diary
    return d[d["patient_idx"] == i]


def sample_origins(cohort: Cohort, origins_per_day: int = 2, seed: int = 11) -> list[np.ndarray]:
    """Hour indices of the origins per patient (episode exclusion applied later)."""
    rng = np.random.default_rng(seed)
    n_hours = len(cohort.hours)
    first, last = WARMUP_HOURS, n_hours - LABEL_HOURS - 1
    days = np.arange(first // 24, last // 24)
    out = []
    for _ in range(len(cohort.patients)):
        picks = days[:, None] * 24 + rng.integers(0, 24, (len(days), origins_per_day))
        picks = np.unique(picks.ravel())
        out.append(picks[(picks >= first) & (picks <= last)])
    return out


def labels_and_mask(onset: np.ndarray, idx: np.ndarray, episode_hours: int):
    """y = onset in (t, t+24]; keep = no onset in (t − episode_hours, t]."""
    c = np.concatenate([[0], np.cumsum(onset)])  # c[k] = onsets in hours < k
    y = (c[idx + LABEL_HOURS + 1] - c[idx + 1]) > 0
    lo = np.clip(idx - episode_hours + 1, 0, None)
    in_episode = (c[idx + 1] - c[lo]) > 0
    return y, ~in_episode


def build_cohort_matrix(
    cohort: Cohort,
    origins: list[np.ndarray] | None = None,
    mode: str = "estimated",
    factors: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(X, y, meta) stacked over patients; meta has patient_idx, t, disease_type, severity."""
    origins = origins if origins is not None else sample_origins(cohort)
    xs, ys, metas = [], [], []
    for i in range(len(cohort.patients)):
        idx = origins[i]
        y, keep = labels_and_mask(cohort.onset[i], idx, cohort.config.episode_hours)
        idx, y = idx[keep], y[keep]
        if idx.size == 0:
            continue
        t = cohort.hours[idx]
        x = build_risk_features(
            patient_profile(cohort, i),
            patient_exposure(cohort, i, mode, factors),
            cohort.weather,
            patient_health(cohort, i),
            t,
        )
        xs.append(x.reset_index(drop=True))
        ys.append(pd.Series(y.astype(int)))
        metas.append(
            pd.DataFrame(
                {
                    "patient_idx": i,
                    "t": t,
                    "disease_type": cohort.patients.at[i, "disease_type"],
                    "severity": cohort.patients.at[i, "severity"],
                }
            )
        )
    X = pd.concat(xs, ignore_index=True).astype("float32")
    return X, pd.concat(ys, ignore_index=True), pd.concat(metas, ignore_index=True)
