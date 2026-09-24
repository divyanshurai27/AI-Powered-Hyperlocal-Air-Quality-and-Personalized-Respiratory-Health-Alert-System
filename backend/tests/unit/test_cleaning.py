import numpy as np
import pandas as pd
import pytest

from app.domain.cleaning import (
    clean_series,
    detect_outliers,
    fill_forward_limited,
    handle_missing_values,
    resample_hourly,
)


def hourly(values, start="2026-09-20T00:30Z") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_resample_places_values_on_ist_hour_grid() -> None:
    s = hourly([10, 11, 12])
    out = resample_hourly(s)
    assert list(out.index.strftime("%H:%M")) == ["00:30", "01:30", "02:30"]
    assert out["value"].tolist() == [10, 11, 12]
    assert not out["is_missing"].any()


def test_resample_does_not_invent_values_for_gaps() -> None:
    s = hourly([10, 11, 12, 13, 14]).drop(pd.Timestamp("2026-09-20T02:30Z"))
    out = resample_hourly(s)
    assert len(out) == 5
    assert np.isnan(out.loc["2026-09-20T02:30Z", "value"])
    assert out["is_missing"].sum() == 1


def test_resample_assigns_sub_hourly_readings_to_period_end() -> None:
    idx = pd.to_datetime(["2026-09-20T00:15Z", "2026-09-20T00:30Z", "2026-09-20T00:45Z"])
    out = resample_hourly(pd.Series([1.0, 3.0, 8.0], index=idx))
    # 00:15 and 00:30 belong to the hour ending 00:30; 00:45 to the hour ending 01:30.
    assert out["value"].tolist() == [2.0, 8.0]


def test_resample_rejects_naive_index() -> None:
    with pytest.raises(ValueError):
        resample_hourly(pd.Series([1.0], index=pd.to_datetime(["2026-09-20T00:30"])))


def test_resample_empty_series() -> None:
    out = resample_hourly(pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float))
    assert out.empty


def test_short_gap_interpolated_and_flagged() -> None:
    frame = resample_hourly(hourly([10, np.nan, np.nan, 40]))
    out = handle_missing_values(frame, max_gap_hours=3)
    assert out["value"].tolist() == [10, 20, 30, 40]
    assert out["was_imputed"].tolist() == [False, True, True, False]
    assert out["is_missing"].tolist() == [False, True, True, False]  # original missingness kept


def test_long_gap_stays_missing() -> None:
    frame = resample_hourly(hourly([10, np.nan, np.nan, np.nan, np.nan, 60]))
    out = handle_missing_values(frame, max_gap_hours=3)
    assert out["value"].isna().sum() == 4
    assert not out["was_imputed"].any()


def test_gap_at_end_is_never_filled() -> None:
    # A trailing gap has no future anchor; filling it would need data we don't have yet.
    frame = pd.DataFrame({"value": [10.0, 11.0, np.nan]}, index=hourly([0, 0, 0]).index)
    frame["is_missing"] = frame["value"].isna()
    out = handle_missing_values(frame)
    assert np.isnan(out["value"].iloc[-1])
    assert not out["was_imputed"].iloc[-1]


def test_spike_flagged_as_outlier() -> None:
    rng = np.random.default_rng(0)
    values = list(30 + rng.normal(0, 2, 30)) + [400.0] + list(30 + rng.normal(0, 2, 5))
    flags = detect_outliers(hourly(values))
    assert flags.iloc[30]
    assert flags.sum() <= 2


def test_outlier_detection_is_trailing_only() -> None:
    base = [30.0 + (i % 3) for i in range(30)]
    with_future_spike = hourly(base + [500.0])
    without = hourly(base + [31.0])
    # Flags for the first 30 hours must not change when a future value changes.
    pd.testing.assert_series_equal(
        detect_outliers(with_future_spike).iloc[:30], detect_outliers(without).iloc[:30]
    )


def test_outlier_needs_minimum_history() -> None:
    assert not detect_outliers(hourly([10, 10, 500])).any()


def test_outliers_are_flagged_not_removed() -> None:
    values = [30.0 + (i % 3) for i in range(30)] + [500.0]
    out = clean_series(hourly(values))
    assert out["value"].iloc[-1] == 500.0
    assert out["is_outlier"].iloc[-1]


# --- model-safe filling ------------------------------------------------------------------


def test_forward_fill_short_gap_and_flag() -> None:
    out = fill_forward_limited(resample_hourly(hourly([10, np.nan, np.nan, 40])), max_gap_hours=3)
    assert out["value"].tolist() == [10, 10, 10, 40]
    assert out["was_imputed"].tolist() == [False, True, True, False]


def test_forward_fill_long_gap_stays_entirely_missing() -> None:
    out = fill_forward_limited(resample_hourly(hourly([10] + [np.nan] * 4 + [60])), max_gap_hours=3)
    assert out["value"].isna().sum() == 4  # not partially filled either
    assert not out["was_imputed"].any()


def test_forward_fill_is_past_only() -> None:
    """Changing a future value must not change any earlier filled value (BR-04)."""
    a = fill_forward_limited(resample_hourly(hourly([10, np.nan, np.nan, 40, 50])))
    b = fill_forward_limited(resample_hourly(hourly([10, np.nan, np.nan, 999, 50])))
    pd.testing.assert_series_equal(a["value"].iloc[:3], b["value"].iloc[:3])


def test_interpolation_uses_future_which_is_why_features_must_not_use_it() -> None:
    a = handle_missing_values(resample_hourly(hourly([10, np.nan, 40])))
    b = handle_missing_values(resample_hourly(hourly([10, np.nan, 90])))
    assert a["value"].iloc[1] != b["value"].iloc[1]


def test_clean_series_is_model_safe() -> None:
    a = clean_series(hourly([10, np.nan, 30, 31, 32]))
    b = clean_series(hourly([10, np.nan, 999, 31, 32]))
    assert a["value"].iloc[1] == b["value"].iloc[1] == 10
