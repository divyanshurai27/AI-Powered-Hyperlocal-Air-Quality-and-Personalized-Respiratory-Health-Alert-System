from datetime import UTC, datetime, timedelta

import pytest

from app.domain.observations import QualityFlag
from app.domain.spatial import (
    GeoPoint,
    LocalEstimate,
    StationReading,
    Unavailable,
    UnavailableReason,
    calculate_distance_km,
    estimate_local_air_quality,
    find_candidate_stations,
    select_representative_station,
    validate_coordinates,
)

NOW = datetime(2026, 9, 20, 12, 30, tzinfo=UTC)
FRESH = timedelta(hours=3)
USER = GeoPoint(12.9716, 77.5946)  # central Bengaluru


def reading(
    sid: str, lat: float, lon: float, value: float, age_h: float = 0.5, **kw
) -> StationReading:
    return StationReading(
        station_id=sid,
        latitude=lat,
        longitude=lon,
        concentration=value,
        timestamp=NOW - timedelta(hours=age_h),
        quality_flag=kw.pop("flag", QualityFlag.VALID),
        coverage_pct=kw.pop("coverage", 100.0),
    )


def estimate(readings, at=NOW, location=USER, **kw):
    return estimate_local_air_quality(
        location, readings, at, radius_km=kw.pop("radius_km", 10.0), max_age=FRESH, **kw
    )


# --- distance ----------------------------------------------------------------------------


def test_zero_distance() -> None:
    assert calculate_distance_km(USER, USER) == 0.0


def test_known_distance_one_degree_longitude_at_equator() -> None:
    assert calculate_distance_km(GeoPoint(0, 0), GeoPoint(0, 1)) == pytest.approx(111.195, abs=0.01)


def test_known_distance_bengaluru_to_mysuru() -> None:
    # Great-circle distance is ~128 km (road is longer).
    d = calculate_distance_km(USER, GeoPoint(12.2958, 76.6394))
    assert 125 < d < 131


def test_distance_is_symmetric() -> None:
    a, b = GeoPoint(12.9, 77.5), GeoPoint(13.1, 77.7)
    assert calculate_distance_km(a, b) == pytest.approx(calculate_distance_km(b, a))


@pytest.mark.parametrize(
    ("lat", "lon", "ok"),
    [(90, 180, True), (-90, -180, True), (90.0001, 0, False), (0, 180.0001, False)],
)
def test_boundary_coordinates(lat, lon, ok) -> None:
    assert validate_coordinates(lat, lon) is ok


# --- candidates & selection --------------------------------------------------------------


def test_candidates_sorted_nearest_first_and_radius_applied() -> None:
    far = reading("far", 13.3, 77.6, 50)  # ~37 km
    near = reading("near", 12.98, 77.60, 30)
    mid = reading("mid", 13.02, 77.60, 40)
    c = find_candidate_stations(USER, [far, mid, near], radius_km=10)
    assert [x.reading.station_id for x in c] == ["near", "mid"]


def test_empty_candidate_list() -> None:
    assert find_candidate_stations(USER, [], radius_km=10) == []
    assert select_representative_station([], NOW, FRESH) is None


def test_stale_nearest_skipped_for_representative() -> None:
    stale_near = reading("stale", 12.972, 77.595, 30, age_h=10)
    fresh_mid = reading("fresh", 13.00, 77.60, 40)
    c = find_candidate_stations(USER, [stale_near, fresh_mid], radius_km=10)
    assert select_representative_station(c, NOW, FRESH).reading.station_id == "fresh"


def test_conflicting_quality_at_same_distance_prefers_valid() -> None:
    a = reading("flagged", 12.98, 77.60, 30, flag=QualityFlag.LOW_COVERAGE)
    b = reading("valid", 12.98, 77.60, 35)
    c = find_candidate_stations(USER, [a, b], radius_km=10)
    assert select_representative_station(c, NOW, FRESH).reading.station_id == "valid"


# --- estimation --------------------------------------------------------------------------


def test_exact_station_match_uses_that_station_only() -> None:
    at_user = reading("here", USER.latitude, USER.longitude, 42.0)
    other = reading("other", 13.00, 77.60, 90.0)
    est = estimate([at_user, other])
    assert isinstance(est, LocalEstimate)
    assert est.method == "exact_station"
    assert est.concentration == 42.0
    assert [c.station_id for c in est.contributions] == ["here"]


def test_one_nearby_station_idw_equals_its_value() -> None:
    est = estimate([reading("only", 13.00, 77.60, 37.5)])
    assert est.method == "idw"
    assert est.concentration == 37.5
    assert est.contributions[0].weight == pytest.approx(1.0)


def test_multiple_stations_closer_weighs_more_and_weights_sum_to_one() -> None:
    near = reading("near", 12.985, 77.595, 20.0)  # ~1.5 km
    far = reading("far", 13.03, 77.595, 80.0)  # ~6.5 km
    est = estimate([near, far])
    w = {c.station_id: c.weight for c in est.contributions}
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["near"] > w["far"]
    assert 20.0 < est.concentration < 50.0  # pulled toward the near station


def test_max_stations_limits_contributors() -> None:
    rs = [reading(f"s{i}", 12.98 + i * 0.005, 77.60, 10 + i) for i in range(6)]
    assert len(estimate(rs, max_stations=4).contributions) == 4


def test_low_coverage_reading_is_down_weighted() -> None:
    a = reading("a", 12.99, 77.5946, 20.0)
    b = reading("b", 12.9532, 77.5946, 80.0, flag=QualityFlag.LOW_COVERAGE)  # same distance
    est = estimate([a, b])
    w = {c.station_id: c.weight for c in est.contributions}
    assert w["a"] == pytest.approx(2 * w["b"], rel=0.02)


def test_no_stations_in_radius_is_explicit_unavailable() -> None:
    est = estimate([reading("far", 13.5, 77.6, 50)])
    assert isinstance(est, Unavailable)
    assert est.reason == UnavailableReason.NO_STATIONS_IN_RADIUS


def test_all_stale_is_no_fresh_data_not_a_value() -> None:
    est = estimate([reading("old", 12.98, 77.60, 50, age_h=48)])
    assert isinstance(est, Unavailable)
    assert est.reason == UnavailableReason.NO_FRESH_DATA
    assert "older than" in est.detail


def test_stale_nearest_falls_back_to_fresh_farther_station() -> None:
    est = estimate(
        [reading("stale", 12.972, 77.595, 999, age_h=5), reading("fresh", 13.00, 77.60, 40)]
    )
    assert [c.station_id for c in est.contributions] == ["fresh"]


def test_future_reading_is_never_used() -> None:
    future = reading("future", 12.98, 77.60, 50, age_h=-1)
    est = estimate([future])
    assert isinstance(est, Unavailable)
    assert est.reason == UnavailableReason.NO_FRESH_DATA


def test_implausible_readings_excluded() -> None:
    est = estimate([reading("bad", 12.98, 77.60, 5000, flag=QualityFlag.IMPLAUSIBLE)])
    assert est.reason == UnavailableReason.NO_USABLE_DATA


def test_invalid_location() -> None:
    est = estimate([reading("a", 12.98, 77.60, 50)], location=GeoPoint(95, 77))
    assert est.reason == UnavailableReason.INVALID_LOCATION


def test_estimate_is_deterministic() -> None:
    rs = [
        reading("a", 12.98, 77.60, 20),
        reading("b", 13.01, 77.58, 60),
        reading("c", 12.95, 77.62, 35),
    ]
    assert estimate(rs) == estimate(list(reversed(rs)))
