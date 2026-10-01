"""Small, fast synthetic pollution field + cohort for risk-layer tests (no database)."""

import numpy as np
import pandas as pd

from app.ml.cohort import Cohort, CohortConfig, generate_cohort
from app.ml.features import build_station_frames

START = pd.Timestamp("2025-03-01 00:30", tz="UTC")
STATIONS = {
    "openaq:1": (12.9135, 77.5951),
    "openaq:2": (12.9716, 77.5946),
    "openaq:3": (13.0350, 77.5970),
}


def station_observations(days: int = 90, seed: int = 0, gap_fraction: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=days * 24, freq="h")
    # A slow "episode" wave on top of a daily cycle so exposure actually varies.
    wave = 1 + 0.6 * np.sin(np.arange(len(idx)) / (24 * 9) * 2 * np.pi)
    daily = 1 + 0.35 * np.sin(2 * np.pi * (idx.hour + 5.5) / 24)
    rows = []
    for k, sid in enumerate(STATIONS):
        for pol, base in (("pm25", 30.0), ("pm10", 60.0), ("o3", 40.0)):
            vals = base * wave * daily * (1 + 0.1 * k) * np.exp(rng.normal(0, 0.15, len(idx)))
            keep = rng.random(len(idx)) > gap_fraction
            rows.append(
                pd.DataFrame(
                    {
                        "station_id": sid,
                        "pollutant": pol,
                        "timestamp": idx[keep],
                        "concentration": vals[keep],
                        "quality_flag": "valid",
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def weather(days: int = 91) -> pd.DataFrame:
    idx = pd.date_range(START.floor("h") - pd.Timedelta(hours=24), periods=days * 24, freq="h")
    n = len(idx)
    return pd.DataFrame(
        {
            "temperature": 22 + 5 * np.sin(np.arange(n) / 24 * 2 * np.pi),
            "relative_humidity": 70 + 15 * np.sin(np.arange(n) / 200),
            "precipitation": 0.0,
        },
        index=idx,
    )


def small_cohort(n_patients: int = 40, days: int = 90, **overrides) -> Cohort:
    frames = build_station_frames(station_observations(days))
    cfg = CohortConfig(n_patients=n_patients, seed=5, **overrides)
    end = START + pd.Timedelta(days=days) - pd.Timedelta(hours=1)
    return generate_cohort(frames, weather(days + 1), STATIONS, START, end, cfg)
