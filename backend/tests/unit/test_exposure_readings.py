"""latest_fresh_by_hour: the batched equivalent of latest_readings + the freshness rule."""

from datetime import UTC, datetime, timedelta

from app.domain.observations import QualityFlag
from app.domain.spatial import StationReading
from app.services.exposure import latest_fresh_by_hour

T0 = datetime(2025, 6, 1, 0, 30, tzinfo=UTC)


def r(sid: str, h: int, v: float) -> StationReading:
    return StationReading(sid, 12.9, 77.6, v, T0 + timedelta(hours=h), QualityFlag.VALID)


def test_newest_fresh_reading_per_station_and_never_the_future() -> None:
    readings = [r("a", 0, 1), r("b", 0, 2), r("a", 1, 3), r("a", 5, 4)]
    grid = [T0 + timedelta(hours=h) for h in (1, 3, 4, 5)]
    out = latest_fresh_by_hour(readings, grid, timedelta(hours=3))
    val = {h: {x.station_id: x.concentration for x in out[h]} for h in grid}
    assert val[grid[0]] == {"a": 3, "b": 2}
    assert val[grid[1]] == {"a": 3, "b": 2}  # b is exactly 3 h old: still fresh
    assert val[grid[2]] == {"a": 3}  # b now 4 h old
    assert val[grid[3]] == {"a": 4}
