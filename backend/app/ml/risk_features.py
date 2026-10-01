"""Respiratory-risk features (PRD §15.1, `build_risk_features`), RISK_FEATURE_VERSION.

One row per (patient, origin time T). Every feature reads only data dated ≤ T (BR-04); the
lookback of each is declared in `risk_feature_lookback_hours` and audited by tests.

The SAME functions build training rows from the synthetic cohort and the live row in the
API, so there is one definition of every feature (no train/serve drift):

    exposure  hourly estimated personal exposure (ambient at the logged place × the
              microenvironment factor), µg/m³, NaN where there was no estimate
    weather   hourly city weather: temperature, relative_humidity, precipitation
    health    diary / wearable records with a `timestamp` column (PRD §6.4)
    profile   age, sex, disease_type, severity

Missing inputs stay NaN (BR-06). LightGBM routes missing values down a learned branch, and
the training data contains the same kinds of gaps (unlogged hours, skipped diaries, no
wearable), so "missing" is a supported input rather than a silent zero.

Pollutants: PM2.5, PM10 and O3 only. NO2/SO2 are excluded until their source units are
verified (docs/data_and_methods.md §1).
"""

import numpy as np
import pandas as pd

from app.domain.enums import DiseaseType, Severity, Sex
from app.domain.risk_policy import PERSONAL_HISTORY_DAYS

RISK_FEATURE_VERSION = "risk_features_v1.0.0"
IST = pd.Timedelta(hours=5, minutes=30)
H = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)

EXPOSURE_FEATURES = [
    "exp_pm25_now",
    "exp_pm25_mean_6h",
    "exp_pm25_mean_24h",
    "exp_pm25_mean_72h",
    "exp_pm25_max_24h",
    "exp_pm25_change_24h",
    "exp_pm25_coverage_24h",
    "exp_pm10_mean_24h",
    "exp_pm10_max_24h",
    "exp_pm10_change_24h",
    "exp_o3_mean_8h",
    "exp_o3_max_24h",
]
WEATHER_FEATURES = ["wx_temperature", "wx_relative_humidity", "wx_temp_change_24h", "wx_precip_24h"]
CALENDAR_FEATURES = [
    "cal_hour_sin",
    "cal_hour_cos",
    "cal_month_sin",
    "cal_month_cos",
    "cal_is_weekend",
]
PROFILE_FEATURES = ["pt_age", "pt_is_male", "pt_is_copd", "pt_severity"]
HEALTH_FEATURES = [
    "hx_symptom_last",
    "hx_symptom_mean_3d",
    "hx_symptom_mean_7d",
    "hx_symptom_trend",
    "hx_rescue_last",
    "hx_rescue_mean_3d",
    "hx_diary_days_7d",
    "hx_events_90d",
    "hx_days_since_event",
    "hx_spo2_last",
    "hx_heart_rate_last",
    "hx_steps_mean_3d",
    "hx_sleep_last",
]
RISK_FEATURE_NAMES = (
    EXPOSURE_FEATURES + WEATHER_FEATURES + CALENDAR_FEATURES + PROFILE_FEATURES + HEALTH_FEATURES
)
FEATURE_GROUPS = {
    "exposure": EXPOSURE_FEATURES,
    "weather": WEATHER_FEATURES,
    "calendar": CALENDAR_FEATURES,
    "profile": PROFILE_FEATURES,
    "health": HEALTH_FEATURES,
}

HEALTH_COLUMNS = (
    "symptom_score",
    "rescue_inhaler_puffs",
    "exacerbation",
    "spo2",
    "heart_rate",
    "steps",
    "sleep_hours",
)
LAST_VALUE_MAX_AGE = {
    "symptom_score": 48,
    "rescue_inhaler_puffs": 48,
    "spo2": 72,
    "heart_rate": 72,
    "sleep_hours": 48,
}  # hours
DAYS_SINCE_EVENT_CAP = 365.0
SEVERITY_ORDER = [s.value for s in Severity]


def risk_feature_lookback_hours() -> dict[str, float]:
    """Earliest offset (hours, ≤ 0) of the data each feature reads, relative to T."""
    spec = {
        "exp_pm25_now": 0,
        "exp_pm25_mean_6h": -5,
        "exp_pm25_mean_24h": -23,
        "exp_pm25_mean_72h": -71,
        "exp_pm25_max_24h": -23,
        "exp_pm25_change_24h": -29,
        "exp_pm25_coverage_24h": -23,
        "exp_pm10_mean_24h": -23,
        "exp_pm10_max_24h": -23,
        "exp_pm10_change_24h": -29,
        "exp_o3_mean_8h": -7,
        "exp_o3_max_24h": -23,
        "wx_temperature": -3,
        "wx_relative_humidity": -3,
        "wx_temp_change_24h": -27,
        "wx_precip_24h": -26,
        "hx_symptom_last": -48,
        "hx_symptom_mean_3d": -72,
        "hx_symptom_mean_7d": -168,
        "hx_symptom_trend": -240,
        "hx_rescue_last": -48,
        "hx_rescue_mean_3d": -72,
        "hx_diary_days_7d": -168,
        "hx_events_90d": -90 * 24,
        "hx_days_since_event": -DAYS_SINCE_EVENT_CAP * 24,
        "hx_spo2_last": -72,
        "hx_heart_rate_last": -72,
        "hx_steps_mean_3d": -72,
        "hx_sleep_last": -48,
    }
    spec |= {n: 0 for n in CALENDAR_FEATURES + PROFILE_FEATURES}
    return spec


# --- feature groups -------------------------------------------------------------------------


def _hourly(frame: pd.DataFrame, col: str, index: pd.DatetimeIndex) -> pd.Series:
    """Column on a complete hourly grid (missing hours → NaN), so rolling windows count hours."""
    s = frame[col] if col in frame else pd.Series(np.nan, index=frame.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    grid = (
        pd.date_range(min(s.index.min(), index.min()), max(s.index.max(), index.max()), freq="h")
        if len(s)
        else index
    )
    return s.reindex(grid).astype(float)


def exposure_features(exposure: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    pm25 = _hourly(exposure, "pm25", origins)
    pm10 = _hourly(exposure, "pm10", origins)
    o3 = _hourly(exposure, "o3", origins)

    def mean(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(w, min_periods=max(1, w // 2)).mean()

    def mx(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(w, min_periods=max(1, w // 2)).max()

    pm25_6 = mean(pm25, 6)
    pm10_6 = mean(pm10, 6)
    cols = {
        "exp_pm25_now": pm25,
        "exp_pm25_mean_6h": pm25_6,
        "exp_pm25_mean_24h": mean(pm25, 24),
        "exp_pm25_mean_72h": mean(pm25, 72),
        "exp_pm25_max_24h": mx(pm25, 24),
        "exp_pm25_change_24h": pm25_6 - pm25_6.shift(24),
        "exp_pm25_coverage_24h": pm25.notna().astype(float).rolling(24, min_periods=1).sum() / 24,
        "exp_pm10_mean_24h": mean(pm10, 24),
        "exp_pm10_max_24h": mx(pm10, 24),
        "exp_pm10_change_24h": pm10_6 - pm10_6.shift(24),
        "exp_o3_mean_8h": mean(o3, 8),
        "exp_o3_max_24h": mx(o3, 24),
    }
    return pd.DataFrame({k: v.reindex(origins) for k, v in cols.items()}, index=origins)


def weather_features(weather: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Latest weather at or before T within 3 h (older → missing)."""
    w = weather.sort_index()
    w = w[~w.index.duplicated(keep="last")]
    tol = pd.Timedelta(hours=3)

    def at(col: str, when: pd.DatetimeIndex) -> np.ndarray:
        if col not in w or w.empty:
            return np.full(len(when), np.nan)
        return w[col].reindex(when, method="ffill", tolerance=tol).to_numpy(float)

    temp = at("temperature", origins)
    precip_24h = (
        w["precipitation"].rolling("24h").sum().reindex(origins, method="ffill", tolerance=tol)
        if "precipitation" in w and not w.empty
        else pd.Series(np.nan, index=origins)
    )
    return pd.DataFrame(
        {
            "wx_temperature": temp,
            "wx_relative_humidity": at("relative_humidity", origins),
            "wx_temp_change_24h": temp - at("temperature", origins - 24 * H),
            "wx_precip_24h": precip_24h.to_numpy(float),
        },
        index=origins,
    )


def calendar_features(origins: pd.DatetimeIndex) -> pd.DataFrame:
    local = origins + IST
    hour = local.hour + local.minute / 60
    return pd.DataFrame(
        {
            "cal_hour_sin": np.sin(2 * np.pi * hour / 24),
            "cal_hour_cos": np.cos(2 * np.pi * hour / 24),
            "cal_month_sin": np.sin(2 * np.pi * (local.month - 1) / 12),
            "cal_month_cos": np.cos(2 * np.pi * (local.month - 1) / 12),
            "cal_is_weekend": (local.dayofweek >= 5).astype(float),
        },
        index=origins,
    )


def profile_features(profile: dict, origins: pd.DatetimeIndex) -> pd.DataFrame:
    sex = profile.get("sex")
    severity = profile.get("severity")
    disease = profile.get("disease_type")
    row = {
        "pt_age": float(profile["age"]) if profile.get("age") is not None else np.nan,
        "pt_is_male": {Sex.MALE.value: 1.0, Sex.FEMALE.value: 0.0}.get(str(sex), np.nan),
        "pt_is_copd": {DiseaseType.COPD.value: 1.0, DiseaseType.ASTHMA.value: 0.0}.get(
            str(disease), np.nan
        ),
        "pt_severity": float(SEVERITY_ORDER.index(str(severity)))
        if str(severity) in SEVERITY_ORDER
        else np.nan,
    }
    return pd.DataFrame([row] * len(origins), index=origins)


class _Timeline:
    """Non-null values of one health column, sorted by time, with prefix sums."""

    def __init__(self, t: np.ndarray, v: np.ndarray) -> None:
        keep = ~np.isnan(v)
        self.t, self.v = t[keep], v[keep]
        self.csum = np.concatenate([[0.0], np.cumsum(self.v)])

    def _span(self, origins: np.ndarray, hours: float) -> tuple[np.ndarray, np.ndarray]:
        hi = np.searchsorted(self.t, origins, side="right")
        lo = np.searchsorted(self.t, origins - np.int64(hours * 3600e9), side="right")
        return lo, hi

    def window(self, origins: np.ndarray, hours: float, end_offset_hours: float = 0):
        """(sum, count) of values in (T − end_offset − hours, T − end_offset]."""
        shifted = origins - np.int64(end_offset_hours * 3600e9)
        lo, hi = self._span(shifted, hours)
        return self.csum[hi] - self.csum[lo], (hi - lo).astype(float)

    def mean(self, origins: np.ndarray, hours: float, end_offset_hours: float = 0) -> np.ndarray:
        s, n = self.window(origins, hours, end_offset_hours)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / n, np.nan)

    def last(self, origins: np.ndarray, max_age_hours: float) -> np.ndarray:
        i = np.searchsorted(self.t, origins, side="right") - 1
        ok = i >= 0
        out = np.full(len(origins), np.nan)
        age_ok = ok.copy()
        age_ok[ok] = origins[ok] - self.t[i[ok]] <= np.int64(max_age_hours * 3600e9)
        out[age_ok] = self.v[i[age_ok]]
        return out


def health_features(health: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    o = origins.as_unit("ns").asi8  # pandas 3 may carry µs resolution; compare in ns
    if health is None or health.empty:
        t = np.array([], dtype=np.int64)
        cols = {c: np.array([], dtype=float) for c in HEALTH_COLUMNS}
    else:
        h = health.sort_values("timestamp")
        t = pd.DatetimeIndex(pd.to_datetime(h["timestamp"], utc=True)).as_unit("ns").asi8
        cols = {
            c: pd.to_numeric(h[c], errors="coerce").to_numpy(float)
            if c in h
            else np.full(len(h), np.nan)
            for c in HEALTH_COLUMNS
        }
    tl = {c: _Timeline(t, v) for c, v in cols.items()}
    sym, rescue = tl["symptom_score"], tl["rescue_inhaler_puffs"]
    diary_days = sym.window(o, 168)[1]

    ev = cols["exacerbation"]
    events = _Timeline(t, np.where(ev > 0, 1.0, np.where(np.isnan(ev), np.nan, 0.0)))
    event_times = events.t[events.v > 0]
    j = np.searchsorted(event_times, o, side="right") - 1
    days_since = np.full(len(o), DAYS_SINCE_EVENT_CAP)
    has = j >= 0
    days_since[has] = np.minimum(DAYS_SINCE_EVENT_CAP, (o[has] - event_times[j[has]]) / (86400e9))
    return pd.DataFrame(
        {
            "hx_symptom_last": sym.last(o, LAST_VALUE_MAX_AGE["symptom_score"]),
            "hx_symptom_mean_3d": sym.mean(o, 72),
            "hx_symptom_mean_7d": sym.mean(o, 168),
            "hx_symptom_trend": sym.mean(o, 72) - sym.mean(o, 168, end_offset_hours=72),
            "hx_rescue_last": rescue.last(o, LAST_VALUE_MAX_AGE["rescue_inhaler_puffs"]),
            "hx_rescue_mean_3d": rescue.mean(o, 72),
            "hx_diary_days_7d": diary_days,
            "hx_events_90d": events.window(o, 90 * 24)[0],
            "hx_days_since_event": days_since,
            "hx_spo2_last": tl["spo2"].last(o, LAST_VALUE_MAX_AGE["spo2"]),
            "hx_heart_rate_last": tl["heart_rate"].last(o, LAST_VALUE_MAX_AGE["heart_rate"]),
            "hx_steps_mean_3d": tl["steps"].mean(o, 72),
            "hx_sleep_last": tl["sleep_hours"].last(o, LAST_VALUE_MAX_AGE["sleep_hours"]),
        },
        index=origins,
    )


def build_risk_features(
    profile: dict,
    exposure: pd.DataFrame,
    weather: pd.DataFrame,
    health: pd.DataFrame | None,
    origins: pd.DatetimeIndex,
) -> pd.DataFrame:
    """All risk features for one patient at each origin, in contract order."""
    origins = pd.DatetimeIndex(origins)
    parts = [
        exposure_features(exposure, origins),
        weather_features(weather, origins),
        calendar_features(origins),
        profile_features(profile, origins),
        health_features(health, origins),
    ]
    return pd.concat(parts, axis=1)[RISK_FEATURE_NAMES]


def personal_history_many(
    health: pd.DataFrame | None, origins: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """(reported exacerbations, diary days) in (T − PERSONAL_HISTORY_DAYS, T] for each origin.
    A diary day is a record that answered the exacerbation question. Input to
    domain.risk_policy.estimate_personal_baseline; shared by training and serving."""
    o = pd.DatetimeIndex(origins).as_unit("ns").asi8
    if health is None or health.empty or "exacerbation" not in health:
        return np.zeros(len(o), dtype=int), np.zeros(len(o), dtype=int)
    h = health.sort_values("timestamp")
    t = pd.DatetimeIndex(pd.to_datetime(h["timestamp"], utc=True)).as_unit("ns").asi8
    answered = _Timeline(t, pd.to_numeric(h["exacerbation"], errors="coerce").to_numpy(float))
    flagged, days = answered.window(o, PERSONAL_HISTORY_DAYS * 24)
    return flagged.round().astype(int), days.astype(int)


def personal_history(health: pd.DataFrame | None, origin: pd.Timestamp) -> tuple[int, int]:
    events, days = personal_history_many(health, pd.DatetimeIndex([origin]))
    return int(events[0]), int(days[0])
