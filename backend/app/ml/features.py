"""Feature engineering for 24-hour AQ forecasting (PRD §11).

Row = (station, origin time t). Every feature uses only data at or before t (BR-04). The
contract is FEATURE_VERSION + the ordered feature list; a model artifact trained on one
contract is refused at load time if the code builds a different one (PRD §39).

Time convention: AQ index = UTC end of the averaging hour (:30 in UTC for IST hours).
Weather (whole UTC hours) is joined as the latest row at or before t.
"""

import numpy as np
import pandas as pd

from app.domain.cleaning import clean_series
from app.domain.observations import QualityFlag
from app.domain.pollutants import Pollutant

FEATURE_VERSION = "aq_features_v1.0.0"

HORIZONS = tuple(range(1, 25))
LAGS_HOURS = (1, 6, 12, 24, 48)
ROLLING_WINDOWS_HOURS = (6, 24)
WEATHER_COLUMNS = (
    "temperature",
    "relative_humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
)
WEATHER_MAX_AGE = pd.Timedelta(hours=3)
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
MODELLED_POLLUTANTS = (Pollutant.PM25, Pollutant.PM10, Pollutant.NO2, Pollutant.O3, Pollutant.CO)


# --- station frames ----------------------------------------------------------------------


def build_station_frames(observations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long observations → one wide hourly frame per station.

    Input columns: station_id, pollutant, timestamp (tz-aware UTC), concentration, quality_flag.
    Output columns per pollutant p present at the station:
        p            cleaned value (short gaps forward-filled; model-safe)
        p__observed  True where a real, non-implausible reading exists
        p__valid     True where that reading is also flagged VALID (≥75% coverage, not
                     provider-flagged). Ground truth for training and primary scoring.
    Implausible readings are removed here; they are neither inputs nor ground truth.
    Low-coverage/provider-flagged readings remain usable as inputs but are not targets.
    """
    obs = observations[observations["quality_flag"] != QualityFlag.IMPLAUSIBLE.value]
    frames: dict[str, pd.DataFrame] = {}
    for station_id, st in obs.groupby("station_id", sort=True):
        cols = {}
        for pollutant, series in st.groupby("pollutant"):
            s = pd.Series(
                series["concentration"].to_numpy(), index=pd.DatetimeIndex(series["timestamp"])
            )
            s = s[~s.index.duplicated(keep="last")]
            cleaned = clean_series(s)
            valid_times = pd.DatetimeIndex(
                series.loc[series["quality_flag"] == QualityFlag.VALID.value, "timestamp"]
            )
            cols[str(pollutant)] = cleaned["value"]
            cols[f"{pollutant}__observed"] = ~cleaned["is_missing"]
            cols[f"{pollutant}__valid"] = cleaned.index.isin(valid_times) & ~cleaned["is_missing"]
        frame = pd.DataFrame(cols).sort_index()
        grid = pd.date_range(frame.index.min(), frame.index.max(), freq="h", tz="UTC")
        frame = frame.reindex(grid)
        for c in frame.columns:
            if c.endswith(("__observed", "__valid")):
                frame[c] = frame[c].fillna(False).astype(bool)
        frames[str(station_id)] = frame
    return frames


# --- feature groups ----------------------------------------------------------------------


def build_temporal_features(frame: pd.DataFrame, target: Pollutant) -> pd.DataFrame:
    p = target.value
    v = frame[p] if p in frame else pd.Series(np.nan, index=frame.index)
    observed = frame.get(f"{p}__observed", pd.Series(False, index=frame.index))
    out = pd.DataFrame(index=frame.index)
    out[f"{p}_current"] = v
    out[f"{p}_current_imputed"] = (v.notna() & ~observed).astype(float)
    for k in LAGS_HOURS:
        out[f"{p}_lag_{k}h"] = v.shift(k)
    out[f"{p}_diff_1h"] = v - v.shift(1)
    for w in ROLLING_WINDOWS_HOURS:
        roll = v.rolling(w, min_periods=max(2, w // 2))
        out[f"{p}_roll{w}_mean"] = roll.mean()
        out[f"{p}_roll{w}_max"] = roll.max()
        out[f"{p}_roll{w}_std"] = roll.std()
    out[f"{p}_observed_24h"] = observed.astype(float).rolling(24, min_periods=1).sum()
    for q in MODELLED_POLLUTANTS:
        if q != target:
            out[f"{q.value}_current"] = frame[q.value] if q.value in frame else np.nan
    return out


def build_weather_features(index: pd.DatetimeIndex, weather: pd.DataFrame) -> pd.DataFrame:
    """Latest weather at or before each t, within WEATHER_MAX_AGE (older → missing)."""
    w = weather.sort_index()
    w = w[~w.index.duplicated(keep="last")]
    aligned = w.reindex(index, method="ffill", tolerance=WEATHER_MAX_AGE)
    out = pd.DataFrame(index=index)
    for col in ("temperature", "relative_humidity", "wind_speed", "pressure", "precipitation"):
        out[f"wx_{col}"] = aligned[col] if col in aligned else np.nan
    direction = np.deg2rad(aligned["wind_direction"]) if "wind_direction" in aligned else np.nan
    out["wx_wind_dir_sin"] = np.sin(direction)
    out["wx_wind_dir_cos"] = np.cos(direction)
    # 24 h precipitation total ending at t, computed on the weather grid, then aligned.
    precip_24h = w["precipitation"].rolling("24h").sum() if "precipitation" in w else None
    out["wx_precip_24h"] = (
        precip_24h.reindex(index, method="ffill", tolerance=WEATHER_MAX_AGE)
        if precip_24h is not None
        else np.nan
    )
    return out


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index + IST_OFFSET  # IST has no DST, so a fixed offset is exact
    hour = local.hour + local.minute / 60
    out = pd.DataFrame(index=index)
    out["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["cal_dow"] = local.dayofweek.astype(float)
    out["cal_is_weekend"] = (local.dayofweek >= 5).astype(float)
    out["cal_month_sin"] = np.sin(2 * np.pi * (local.month - 1) / 12)
    out["cal_month_cos"] = np.cos(2 * np.pi * (local.month - 1) / 12)
    return out


def generate_features(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
    target: Pollutant,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """All features for one station, one row per origin time t, in contract order."""
    parts = [
        build_temporal_features(frame, target),
        build_weather_features(frame.index, weather),
        build_calendar_features(frame.index),
    ]
    out = pd.concat(parts, axis=1)
    out["st_latitude"] = latitude
    out["st_longitude"] = longitude
    return out[feature_names(target)]


def feature_names(target: Pollutant) -> list[str]:
    p = target.value
    names = [f"{p}_current", f"{p}_current_imputed"]
    names += [f"{p}_lag_{k}h" for k in LAGS_HOURS]
    names += [f"{p}_diff_1h"]
    for w in ROLLING_WINDOWS_HOURS:
        names += [f"{p}_roll{w}_mean", f"{p}_roll{w}_max", f"{p}_roll{w}_std"]
    names += [f"{p}_observed_24h"]
    names += [f"{q.value}_current" for q in MODELLED_POLLUTANTS if q != target]
    names += [
        "wx_temperature",
        "wx_relative_humidity",
        "wx_wind_speed",
        "wx_pressure",
        "wx_precipitation",
        "wx_wind_dir_sin",
        "wx_wind_dir_cos",
        "wx_precip_24h",
        "cal_hour_sin",
        "cal_hour_cos",
        "cal_dow",
        "cal_is_weekend",
        "cal_month_sin",
        "cal_month_cos",
        "st_latitude",
        "st_longitude",
    ]
    return names


def feature_lookback_hours(target: Pollutant) -> dict[str, float]:
    """Earliest source offset each feature reads, in hours relative to t (all ≤ 0).

    Used by the leakage audit: a positive offset would mean the feature reads the future.
    """
    p = target.value
    spec: dict[str, float] = {f"{p}_current": 0, f"{p}_current_imputed": 0, f"{p}_diff_1h": -1}
    spec |= {f"{p}_lag_{k}h": -k for k in LAGS_HOURS}
    for w in ROLLING_WINDOWS_HOURS:
        spec |= {f"{p}_roll{w}_{s}": -(w - 1) for s in ("mean", "max", "std")}
    spec[f"{p}_observed_24h"] = -23
    spec |= {f"{q.value}_current": 0 for q in MODELLED_POLLUTANTS if q != target}
    spec |= {
        n: -WEATHER_MAX_AGE.total_seconds() / 3600
        for n in feature_names(target)
        if n.startswith("wx_")
    }
    spec["wx_precip_24h"] = -24 - WEATHER_MAX_AGE.total_seconds() / 3600
    spec |= {n: 0 for n in feature_names(target) if n.startswith(("cal_", "st_"))}
    return spec


# --- supervised matrix ------------------------------------------------------------------


def build_targets(frame: pd.DataFrame, target: Pollutant, valid_only: bool = True) -> pd.DataFrame:
    """Y[h] = the observed value at t+h. Imputed or missing hours are NaN: we never score a
    forecast against a value we filled in ourselves (PRD §3.1).

    valid_only=True (default, used for training and primary scoring) also drops
    low-coverage and provider-flagged hours, which include isolated sensor glitches.
    valid_only=False gives the secondary "all observed hours" evaluation.
    """
    p = target.value
    if p not in frame:
        return pd.DataFrame(np.nan, index=frame.index, columns=[f"h{h}" for h in HORIZONS])
    mask = frame[f"{p}__valid"] if valid_only else frame[f"{p}__observed"]
    observed_value = frame[p].where(mask)
    return pd.DataFrame({f"h{h}": observed_value.shift(-h) for h in HORIZONS}, index=frame.index)


def build_seasonal_naive(frame: pd.DataFrame, target: Pollutant) -> pd.DataFrame:
    """Baseline: forecast for t+h is the observed value 24 h before it, i.e. at t+h-24 ≤ t."""
    p = target.value
    v = (
        frame[p].where(frame[f"{p}__observed"])
        if p in frame
        else pd.Series(np.nan, index=frame.index)
    )
    return pd.DataFrame({f"h{h}": v.shift(24 - h) for h in HORIZONS}, index=frame.index)


def build_forecasting_matrix(
    frames: dict[str, pd.DataFrame],
    weather: pd.DataFrame,
    target: Pollutant,
    station_coords: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stack all stations → (X, Y, seasonal_naive, Y_all), MultiIndex (station_id, t),
    time-ordered. Y = valid-only targets (training + primary score); Y_all = every observed
    hour including flagged ones (secondary score).

    Rows are kept only where the target pollutant was actually observed at t, because a
    forecast issued without a current reading is not something we'd serve (PRD §12 gate).
    """
    xs, ys, sn, ya = [], [], [], []
    for station_id, frame in frames.items():
        if target.value not in frame:
            continue
        lat, lon = station_coords[station_id]
        x = generate_features(frame, weather, target, lat, lon)
        y = build_targets(frame, target)
        y_all = build_targets(frame, target, valid_only=False)
        s = build_seasonal_naive(frame, target)
        keep = frame[f"{target.value}__observed"]
        idx = pd.MultiIndex.from_arrays(
            [[station_id] * keep.sum(), frame.index[keep]], names=["station_id", "t"]
        )
        xs.append(x[keep].set_axis(idx))
        ys.append(y[keep].set_axis(idx))
        ya.append(y_all[keep].set_axis(idx))
        sn.append(s[keep].set_axis(idx))
    if not xs:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    X, Y, S, YA = (pd.concat(p).sort_index(level=["t", "station_id"]) for p in (xs, ys, sn, ya))
    return X, Y, S, YA
