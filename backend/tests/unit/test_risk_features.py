"""Risk feature contract: past-only windows (BR-04), explicit missingness (BR-06)."""

import numpy as np
import pandas as pd
import pytest

from app.ml.risk_features import (
    DAYS_SINCE_EVENT_CAP,
    RISK_FEATURE_NAMES,
    build_risk_features,
    health_features,
    risk_feature_lookback_hours,
)

T0 = pd.Timestamp("2025-06-01 00:30", tz="UTC")
HOURS = pd.date_range(T0, periods=24 * 20, freq="h")
PROFILE = {"age": 40, "sex": "female", "disease_type": "asthma", "severity": "moderate"}


def exposure(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {p: rng.uniform(5, 80, len(HOURS)) for p in ("pm25", "pm10", "o3")}, index=HOURS
    )


def weather() -> pd.DataFrame:
    n = len(HOURS)
    return pd.DataFrame(
        {
            "temperature": 20 + np.arange(n) % 10,
            "relative_humidity": 60.0,
            "precipitation": np.where(np.arange(n) % 5 == 0, 1.0, 0.0),
        },
        index=HOURS - pd.Timedelta(minutes=30),
    )


def diary() -> pd.DataFrame:
    stamps = pd.date_range(T0 + pd.Timedelta(hours=15), periods=19, freq="D")
    n = len(stamps)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "symptom_score": np.arange(n) % 5,
            "rescue_inhaler_puffs": np.arange(n) % 3,
            "exacerbation": np.where(np.arange(n) == 10, 1.0, 0.0),
            "spo2": 96.0,
            "heart_rate": 70.0,
            "steps": 5000.0,
            "sleep_hours": 7.0,
        }
    )


def features(exp=None, wx=None, hx=None, origins=None) -> pd.DataFrame:
    origins = origins if origins is not None else HOURS[240:260]
    return build_risk_features(
        PROFILE,
        exposure() if exp is None else exp,
        weather() if wx is None else wx,
        diary() if hx is None else hx,
        origins,
    )


def test_columns_follow_the_contract() -> None:
    assert list(features().columns) == RISK_FEATURE_NAMES
    assert len(set(RISK_FEATURE_NAMES)) == len(RISK_FEATURE_NAMES)


def test_every_feature_declares_a_past_only_lookback() -> None:
    spec = risk_feature_lookback_hours()
    assert set(spec) == set(RISK_FEATURE_NAMES)
    assert all(v <= 0 for v in spec.values())


def test_changing_the_future_does_not_change_features() -> None:
    """Scramble every input after T: features at T must be identical (BR-04)."""
    T = HOURS[300]
    base = features(origins=pd.DatetimeIndex([T]))
    exp = exposure()
    exp.loc[exp.index > T] = 999.0
    wx = weather()
    wx.loc[wx.index > T] = -50.0
    hx = diary()
    later = hx["timestamp"] > T
    hx.loc[later, ["symptom_score", "rescue_inhaler_puffs", "exacerbation"]] = [10, 30, 1.0]
    changed = features(exp, wx, hx, pd.DatetimeIndex([T]))
    pd.testing.assert_frame_equal(base, changed)


def test_changing_the_past_does_change_features() -> None:
    T = HOURS[300]
    exp = exposure()
    exp.loc[T - pd.Timedelta(hours=2), "pm25"] = 500.0
    a = features(origins=pd.DatetimeIndex([T]))
    b = features(exp=exp, origins=pd.DatetimeIndex([T]))
    assert b["exp_pm25_max_24h"].iat[0] == 500.0 != a["exp_pm25_max_24h"].iat[0]


def test_exposure_windows_match_a_direct_computation() -> None:
    exp = exposure()
    T = HOURS[300]
    f = features(exp=exp, origins=pd.DatetimeIndex([T])).iloc[0]
    win24 = exp.loc[(exp.index > T - pd.Timedelta(hours=24)) & (exp.index <= T), "pm25"]
    assert f["exp_pm25_mean_24h"] == pytest.approx(win24.mean())
    assert f["exp_pm25_max_24h"] == pytest.approx(win24.max())
    assert f["exp_pm25_now"] == pytest.approx(exp.at[T, "pm25"])
    assert f["exp_pm25_coverage_24h"] == 1.0


def test_missing_exposure_stays_missing_and_lowers_coverage() -> None:
    exp = exposure()
    T = HOURS[300]
    exp.loc[exp.index > T - pd.Timedelta(hours=18), "pm25"] = np.nan
    f = features(exp=exp, origins=pd.DatetimeIndex([T])).iloc[0]
    assert np.isnan(f["exp_pm25_now"])
    assert f["exp_pm25_coverage_24h"] == pytest.approx(6 / 24)
    assert np.isnan(f["exp_pm25_mean_24h"])  # fewer than half the hours → no mean
    assert not np.isnan(f["exp_pm25_mean_72h"])


def test_health_windows_are_exact() -> None:
    hx = diary()
    T = hx["timestamp"].iloc[12] + pd.Timedelta(hours=1)
    f = health_features(hx, pd.DatetimeIndex([T])).iloc[0]
    last3 = hx["symptom_score"].iloc[10:13]
    assert f["hx_symptom_last"] == hx["symptom_score"].iloc[12]
    assert f["hx_symptom_mean_3d"] == pytest.approx(last3.mean())
    assert f["hx_symptom_trend"] == pytest.approx(
        last3.mean() - hx["symptom_score"].iloc[3:10].mean()
    )
    assert f["hx_diary_days_7d"] == 7
    assert f["hx_events_90d"] == 1
    assert f["hx_days_since_event"] == pytest.approx(2 + 1 / 24)


def test_no_health_data_is_missing_not_zero() -> None:
    f = health_features(pd.DataFrame(), pd.DatetimeIndex([HOURS[100]])).iloc[0]
    for col in ("hx_symptom_last", "hx_symptom_mean_3d", "hx_spo2_last", "hx_steps_mean_3d"):
        assert np.isnan(f[col]), col
    # Counts over an empty history are genuinely zero; "no event on record" is the cap.
    assert f["hx_diary_days_7d"] == 0
    assert f["hx_events_90d"] == 0
    assert f["hx_days_since_event"] == DAYS_SINCE_EVENT_CAP


def test_stale_last_values_are_dropped() -> None:
    hx = diary().iloc[:5]
    T = hx["timestamp"].iloc[-1] + pd.Timedelta(hours=49)
    f = health_features(hx, pd.DatetimeIndex([T])).iloc[0]
    assert np.isnan(f["hx_symptom_last"])
    assert not np.isnan(f["hx_spo2_last"])  # 72 h allowance for wearables


def test_profile_encoding_and_unknowns() -> None:
    origins = pd.DatetimeIndex([HOURS[250]])
    f = features(origins=origins).iloc[0]
    assert (f["pt_age"], f["pt_is_male"], f["pt_is_copd"], f["pt_severity"]) == (40, 0, 0, 1)
    other = build_risk_features(
        {"age": None, "sex": "prefer_not_to_say", "disease_type": "copd", "severity": None},
        exposure(),
        weather(),
        diary(),
        origins,
    ).iloc[0]
    assert np.isnan(other["pt_age"]) and np.isnan(other["pt_is_male"])
    assert other["pt_is_copd"] == 1 and np.isnan(other["pt_severity"])


def test_personal_history_counts_answered_diary_days_before_t() -> None:
    from app.ml.risk_features import personal_history, personal_history_many

    hx = diary()
    hx.loc[3, "exacerbation"] = np.nan  # diary skipped the question that day
    T = hx["timestamp"].iloc[12]
    events, days = personal_history(hx, T)
    assert (events, days) == (1, 12)  # rows 0..12 minus the unanswered one; event at row 10
    before = personal_history(hx, hx["timestamp"].iloc[10] - pd.Timedelta(seconds=1))
    assert before == (0, 9)
    e, n = personal_history_many(pd.DataFrame(), pd.DatetimeIndex([T]))
    assert (e[0], n[0]) == (0, 0)
