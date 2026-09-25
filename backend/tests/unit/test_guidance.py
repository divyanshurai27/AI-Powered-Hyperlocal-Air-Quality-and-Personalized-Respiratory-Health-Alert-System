from datetime import UTC, datetime, timedelta

import pytest

from app.domain.guidance import (
    Level,
    avoid_windows,
    best_window,
    day_guidance,
    hourly_guidance,
    naqi_category,
    sensitivity_tier,
)
from app.domain.pollutants import Pollutant

P25, P10, O3 = Pollutant.PM25, Pollutant.PM10, Pollutant.O3
# Hour ends at :30 UTC = whole IST hours. 00:30 UTC = 06:00 IST.
START = datetime(2026, 9, 21, 0, 30, tzinfo=UTC)


def series(values, start=START):
    return [(start + timedelta(hours=i), v) for i, v in enumerate(values)]


@pytest.mark.parametrize(
    ("pollutant", "value", "category"),
    [
        (P25, 0, 0),
        (P25, 30, 0),
        (P25, 30.1, 1),
        (P25, 60, 1),
        (P25, 91, 3),
        (P25, 251, 5),
        (P10, 50, 0),
        (P10, 51, 1),
        (P10, 101, 2),
        (O3, 100, 1),
        (O3, 169, 3),
        (Pollutant.CO, 999, 0),
        (Pollutant.CO, 1500, 1),  # µg/m³ in, compared as mg/m³
    ],
)
def test_naqi_category_boundaries(pollutant, value, category) -> None:
    assert naqi_category(pollutant, value) == category


@pytest.mark.parametrize(
    ("disease", "severity", "tier"),
    [
        (None, None, "general"),
        ("asthma", "mild", "sensitive"),
        ("copd", "moderate", "sensitive"),
        ("asthma", "severe", "high"),
        ("copd", "very_severe", "high"),
    ],
)
def test_sensitivity_tiers(disease, severity, tier) -> None:
    assert sensitivity_tier(disease, severity) == tier


@pytest.mark.parametrize(
    ("tier", "pm25", "level"),
    [
        ("general", 45, Level.GOOD),  # Satisfactory
        ("sensitive", 45, Level.CAUTION),
        ("general", 75, Level.CAUTION),  # Moderate
        ("sensitive", 75, Level.LIMIT),
        ("high", 75, Level.AVOID),
        ("sensitive", 20, Level.GOOD),  # Good is good for everyone
    ],
)
def test_asthma_and_copd_get_stricter_guidance(tier, pm25, level) -> None:
    (h,) = hourly_guidance({P25: series([pm25])}, tier)
    assert h.level == level


def test_worst_pollutant_dominates() -> None:
    (h,) = hourly_guidance({P25: series([20]), P10: series([120])}, "sensitive")  # PM10 Moderate
    assert h.dominant == P10 and h.level == Level.LIMIT


def test_waking_hours_are_ist_06_to_22() -> None:
    hours = hourly_guidance({P25: series([20] * 24)}, "general")
    waking = [h for h in hours if h.waking]
    assert len(waking) == 16  # 06:00–22:00 IST


def test_best_window_is_the_cleanest_waking_stretch() -> None:
    # Entry i covers IST (5+i):00–(6+i):00; entries 5–6 (10:00–12:00 IST) are the cleanest.
    values = [80, 80, 70, 60, 50, 15, 15, 40, 80, 80, 80, 80, 80, 80, 80, 80] + [5] * 8  # night: 5
    hours = hourly_guidance({P25: series(values)}, "sensitive")
    start, end = best_window(hours)
    assert (start + timedelta(hours=5, minutes=30)).hour == 10
    assert end - start == timedelta(hours=2)


def test_best_window_ignores_night_hours() -> None:
    values = [80] * 16 + [1] * 8  # clean only at night
    start, _ = best_window(hourly_guidance({P25: series(values)}, "sensitive"))
    ist = (start + timedelta(hours=5, minutes=30)).hour
    assert 6 <= ist < 22


def test_avoid_windows_only_flag_hours_worse_than_best() -> None:
    values = [20] * 4 + [100] * 3 + [20] * 17  # Poor spike 09:00–12:00 IST
    windows = avoid_windows(hourly_guidance({P25: series(values)}, "sensitive"))
    assert len(windows) == 1
    s, e = windows[0]
    assert ((s + timedelta(hours=5, minutes=30)).hour, e - s) == (9, timedelta(hours=3))


def test_uniformly_bad_day_flags_no_avoid_window_but_has_worst_window() -> None:
    g = day_guidance({P25: series([75 + (i % 5) for i in range(24)])}, "asthma", "moderate")
    assert g.headline_level == Level.LIMIT
    assert g.avoid_windows == []
    assert g.best_window is not None and g.worst_window is not None
    assert g.best_window != g.worst_window


def test_headline_is_worst_waking_hour() -> None:
    values = [20] * 10 + [75] + [20] * 13  # one Moderate hour at 16:00 IST
    g = day_guidance({P25: series(values)}, "asthma", "mild")
    assert g.headline_level == Level.LIMIT and g.tier == "sensitive"


def test_same_air_different_people_different_guidance() -> None:
    air = {P25: series([45] * 24)}  # Satisfactory all day
    assert day_guidance(air, None, None).headline_level == Level.GOOD
    assert day_guidance(air, "asthma", "moderate").headline_level == Level.CAUTION
