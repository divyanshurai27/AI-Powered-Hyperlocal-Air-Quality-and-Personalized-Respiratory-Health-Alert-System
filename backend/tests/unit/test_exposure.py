from datetime import UTC, datetime, timedelta

import pytest

from app.domain.exposure import (
    MICROENVIRONMENT_FACTORS,
    MicroEnvironment,
    build_exposure_profile,
    calculate_cumulative_exposure,
    calculate_lagged_exposure,
    estimate_hourly_exposure,
    sensitivity_factors,
)
from app.domain.pollutants import Pollutant

END = datetime(2026, 9, 20, 12, 30, tzinfo=UTC)
P = Pollutant.PM25


def series(values, env=MicroEnvironment.OUTDOOR):
    n = len(values)
    return [
        estimate_hourly_exposure(END - timedelta(hours=n - 1 - i), P, v, env)
        if v is not None
        else estimate_hourly_exposure(
            END - timedelta(hours=n - 1 - i), P, None, None, location_available=False
        )
        for i, v in enumerate(values)
    ]


def test_outdoor_exposure_equals_ambient() -> None:
    h = estimate_hourly_exposure(END, P, 40.0, MicroEnvironment.OUTDOOR)
    assert (h.exposure, h.factor, h.assumed_outdoor) == (40.0, 1.0, False)


def test_indoor_reduces_and_transit_increases_pm25() -> None:
    home = estimate_hourly_exposure(END, P, 40.0, MicroEnvironment.INDOOR_HOME).exposure
    transit = estimate_hourly_exposure(END, P, 40.0, MicroEnvironment.IN_TRANSIT).exposure
    assert home < 40.0 < transit


def test_unknown_microenvironment_assumes_ambient_and_says_so() -> None:
    h = estimate_hourly_exposure(END, P, 40.0, None)
    assert h.exposure == 40.0
    assert h.assumed_outdoor
    assert h.microenvironment == MicroEnvironment.UNKNOWN


def test_missing_location_gives_missing_exposure_not_zero() -> None:
    h = estimate_hourly_exposure(END, P, 40.0, MicroEnvironment.OUTDOOR, location_available=False)
    assert h.exposure is None
    assert h.missing_reason == "location_unavailable"


@pytest.mark.parametrize("ambient", [None, float("nan")])
def test_missing_ambient_gives_missing_exposure(ambient) -> None:
    h = estimate_hourly_exposure(END, P, ambient, MicroEnvironment.OUTDOOR)
    assert h.exposure is None and h.missing_reason == "ambient_unavailable"


def test_every_microenvironment_has_a_factor_for_every_pollutant() -> None:
    for env in MicroEnvironment:
        assert set(MICROENVIRONMENT_FACTORS[env]) == set(Pollutant)
        assert all(f > 0 for f in MICROENVIRONMENT_FACTORS[env].values())


def test_cumulative_window_sum_mean_max() -> None:
    w = calculate_cumulative_exposure(series([10, 20, 30, 40]), 4, END)
    assert (w.cumulative, w.mean, w.maximum, w.coverage, w.complete) == (100, 25, 40, 1.0, True)


def test_window_with_gaps_reports_coverage_and_incomplete() -> None:
    w = calculate_cumulative_exposure(series([10, None, None, 40]), 4, END)
    assert w.coverage == 0.5 and not w.complete
    assert w.mean == 25 and w.cumulative == 50  # over hours that have data only


def test_window_excludes_hours_outside_it() -> None:
    w = calculate_cumulative_exposure(series([999, 10, 20]), 2, END)
    assert w.cumulative == 30


def test_empty_window() -> None:
    w = calculate_cumulative_exposure([], 24, END)
    assert w.mean is None and w.coverage == 0 and not w.complete


def test_lagged_exposure_reads_exact_past_hours() -> None:
    hours = series([float(i) for i in range(30)])  # value i at END-(29-i)h
    lag = calculate_lagged_exposure(hours, END)
    assert (lag["lag_0h"], lag["lag_1h"], lag["lag_3h"], lag["lag_24h"]) == (29, 28, 26, 5)
    assert lag["change_vs_24h"] == 24


def test_lagged_exposure_ignores_future_hours() -> None:
    hours = series([10.0, 20.0]) + [
        estimate_hourly_exposure(END + timedelta(hours=1), P, 999.0, MicroEnvironment.OUTDOOR)
    ]
    assert calculate_lagged_exposure(hours, END)["lag_0h"] == 20.0


def test_profile_is_time_ordered_and_versioned() -> None:
    hours = list(reversed(series([10.0, 20.0, 30.0])))
    profile = build_exposure_profile(P, hours, END, windows=(3,))
    assert [h.exposure for h in profile.hours] == [10, 20, 30]
    assert profile.model_version.startswith("exposure_")


def test_sensitivity_scales_only_non_outdoor_factors() -> None:
    up = sensitivity_factors(1.2)
    assert up[MicroEnvironment.OUTDOOR][P] == 1.0
    assert up[MicroEnvironment.UNKNOWN][P] == 1.0
    assert up[MicroEnvironment.INDOOR_HOME][P] == pytest.approx(0.96)
    only_transit = sensitivity_factors(0.5, only=MicroEnvironment.IN_TRANSIT)
    assert only_transit[MicroEnvironment.INDOOR_HOME][P] == 0.8
    assert only_transit[MicroEnvironment.IN_TRANSIT][P] == pytest.approx(0.65)


def test_sensitivity_changes_estimate() -> None:
    base = estimate_hourly_exposure(END, P, 50.0, MicroEnvironment.INDOOR_HOME).exposure
    low = estimate_hourly_exposure(
        END, P, 50.0, MicroEnvironment.INDOOR_HOME, factors=sensitivity_factors(0.8)
    ).exposure
    assert low == pytest.approx(base * 0.8)
