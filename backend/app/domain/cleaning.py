"""Time-series cleaning for one station × pollutant series (PRD §9).

Every function preserves missingness: gaps that are filled get `was_imputed=True`, gaps that
are too long stay NaN, and outliers are flagged, never removed.

Leakage rule (BR-04): `clean_series` and `fill_forward_limited` only use the past, so their
output is safe as model input. `handle_missing_values` interpolates between both ends of a
gap, which uses the value *after* the gap. It is for descriptive statistics and charts only
and must never feed a feature.
"""

import numpy as np
import pandas as pd

CLEANING_VERSION = "cleaning_v1.1.0"

# Filling is only defensible over short gaps; longer gaps stay missing.
MAX_INTERPOLATION_GAP_HOURS = 3
MAX_FORWARD_FILL_HOURS = 3

# Trailing-window robust z-score. 1.4826 × MAD estimates σ for normally distributed data.
OUTLIER_WINDOW_HOURS = 24
OUTLIER_MIN_PERIODS = 6
OUTLIER_Z_THRESHOLD = 6.0
_MAD_TO_SIGMA = 1.4826


def resample_hourly(series: pd.Series, freq_offset: str = "30min") -> pd.DataFrame:
    """Place observations on a complete hourly UTC grid without inventing values.

    `series` is indexed by tz-aware UTC period-end timestamps. Indian hourly means end at :30
    UTC (IST = UTC+5:30), hence the default 30-minute grid offset. Several readings that land
    in the same grid hour are averaged (only happens if a source reports sub-hourly).
    """
    if series.index.tz is None:
        raise ValueError("series index must be timezone-aware")
    s = series.sort_index()
    s.index = s.index.tz_convert("UTC")
    if s.empty:
        return pd.DataFrame({"value": pd.Series(dtype=float), "is_missing": pd.Series(dtype=bool)})

    offset = pd.Timedelta(freq_offset)
    # Label each reading with the grid hour it belongs to (ceil: period-end convention).
    labels = (s.index - offset).ceil("h") + offset
    hourly = s.groupby(labels).mean()
    grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz="UTC")
    out = pd.DataFrame({"value": hourly.reindex(grid)})
    out["is_missing"] = out["value"].isna()
    return out


def handle_missing_values(
    frame: pd.DataFrame, max_gap_hours: int = MAX_INTERPOLATION_GAP_HOURS
) -> pd.DataFrame:
    """Linearly interpolate interior gaps of ≤ max_gap_hours; leave longer gaps as NaN.

    DESCRIPTIVE USE ONLY. Interpolation needs the observation after the gap, so a filled
    value contains future information. Use `fill_forward_limited` for model inputs.
    """
    out = frame.copy()
    missing = out["value"].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    short_gap = missing & (run_len <= max_gap_hours)

    interpolated = out["value"].interpolate(method="linear", limit_area="inside")
    fill = short_gap & interpolated.notna()
    out.loc[fill, "value"] = interpolated[fill]
    out["was_imputed"] = fill
    return out


def _mad(window: np.ndarray) -> float:
    w = window[~np.isnan(window)]
    return float(np.median(np.abs(w - np.median(w)))) if w.size else np.nan


def detect_outliers(
    values: pd.Series,
    window: int = OUTLIER_WINDOW_HOURS,
    z_threshold: float = OUTLIER_Z_THRESHOLD,
) -> pd.Series:
    """Boolean flag per point: robust z-score against the *preceding* `window` hours.

    Trailing (not centred) so a flag at time T depends only on data up to T. Extreme but
    real events (festival fireworks, crop burning) will be flagged, which is the point:
    they are reviewed, not deleted.
    """
    prior = values.shift(1).rolling(window, min_periods=OUTLIER_MIN_PERIODS)
    median = prior.median()
    mad = prior.apply(_mad, raw=True)
    scale = (mad * _MAD_TO_SIGMA).replace(0, np.nan)
    z = (values - median).abs() / scale
    return (z > z_threshold).fillna(False).astype(bool)


def fill_forward_limited(
    frame: pd.DataFrame, max_gap_hours: int = MAX_FORWARD_FILL_HOURS
) -> pd.DataFrame:
    """Carry the last observed value forward for at most `max_gap_hours` hours.

    Past-only: the value at T depends on nothing after T, so it is safe as a model input.
    Only gaps no longer than the limit are filled; a longer gap stays entirely missing, so
    we never pretend a sensor outage was a flat line.
    """
    out = frame.copy()
    missing = out["value"].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    filled = out["value"].ffill(limit=max_gap_hours)
    fill = missing & (run_len <= max_gap_hours) & filled.notna()
    out.loc[fill, "value"] = filled[fill]
    out["was_imputed"] = fill
    return out


def clean_series(series: pd.Series) -> pd.DataFrame:
    """Model-safe cleaning: resample → flag outliers → forward-fill short gaps.

    Columns: value, is_missing (before filling), was_imputed, is_outlier.
    """
    frame = resample_hourly(series)
    frame["is_outlier"] = detect_outliers(frame["value"])
    return fill_forward_limited(frame)
