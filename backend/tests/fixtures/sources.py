"""Deterministic source responses shaped like the real OpenAQ v3 / Open-Meteo payloads
(captured 2026-09-24). Tests never call live APIs (PRD §48)."""

from datetime import UTC, datetime, timedelta
from typing import Any

# Bengaluru stations. BTM coordinates are OpenAQ's; the others are approximate.
BTM = {"id": 5548, "name": "BTM Layout, Bengaluru - CPCB", "lat": 12.9135218, "lon": 77.5950804}
JAYANAGAR = {
    "id": 6973,
    "name": "Jayanagar 5th Block, Bengaluru - KSPCB",
    "lat": 12.920984,
    "lon": 77.584908,
}
HEBBAL = {"id": 6984, "name": "Hebbal, Bengaluru - KSPCB", "lat": 13.029152, "lon": 77.585901}


def _dt(v: datetime | None) -> dict[str, str] | None:
    return {"utc": v.strftime("%Y-%m-%dT%H:%M:%SZ"), "local": ""} if v else None


def sensor(
    sid: int, name: str, units: str, first: datetime | None = None, last: datetime | None = None
) -> dict:
    return {
        "id": sid,
        "name": f"{name} {units}",
        "parameter": {"id": 0, "name": name, "units": units, "displayName": None},
        "datetimeFirst": _dt(first),
        "datetimeLast": _dt(last),
    }


def location(
    st: dict,
    sensors: list[dict],
    provider: str = "CPCB",
    is_monitor: bool = True,
) -> dict[str, Any]:
    return {
        "id": st["id"],
        "name": st["name"],
        "locality": None,
        "timezone": "Asia/Kolkata",
        "country": {"id": 9, "code": "IN", "name": "India"},
        "provider": {"id": 168, "name": provider},
        "isMobile": False,
        "isMonitor": is_monitor,
        "sensors": sensors,
        "coordinates": {"latitude": st["lat"], "longitude": st["lon"]},
        "datetimeFirst": _dt(datetime(2025, 2, 18, tzinfo=UTC)),
        "datetimeLast": _dt(datetime(2026, 9, 21, 15, 30, tzinfo=UTC)),
    }


def locations_payload() -> dict[str, Any]:
    first, last = datetime(2025, 2, 18, tzinfo=UTC), datetime(2026, 9, 21, 15, 30, tzinfo=UTC)
    old_first, old_last = datetime(2018, 3, 9, tzinfo=UTC), datetime(2022, 10, 16, tzinfo=UTC)
    return {
        "meta": {"found": 4},
        "results": [
            location(
                BTM,
                [
                    sensor(14635, "pm25", "µg/m³", old_first, old_last),  # legacy feed, no overlap
                    sensor(12235361, "pm25", "µg/m³", first, last),
                    sensor(12235358, "no2", "ppb", first, last),
                    sensor(12235362, "relativehumidity", "%", first, last),  # not a pollutant
                ],
            ),
            location(JAYANAGAR, [sensor(12235267, "pm25", "µg/m³", first, last)]),
            location(
                {"id": 6206921, "name": "Koramangala", "lat": 12.93, "lon": 77.62},
                [sensor(99001, "pm25", "µg/m³", first, last)],
                provider="AirGradient",
                is_monitor=False,
            ),
            {"id": 1, "name": "broken", "sensors": []},  # no coordinates: must be skipped
        ],
    }


def hour_record(
    end_utc: datetime,
    value: float | None,
    *,
    param: str = "pm25",
    units: str = "µg/m³",
    coverage: float | None = 100.0,
    flagged: bool = False,
) -> dict[str, Any]:
    return {
        "value": value,
        "flagInfo": {"hasFlags": flagged},
        "parameter": {"id": 2, "name": param, "units": units, "displayName": None},
        "period": {
            "label": "1hour",
            "interval": "01:00:00",
            "datetimeFrom": _dt(end_utc - timedelta(hours=1)),
            "datetimeTo": _dt(end_utc),
        },
        "coordinates": None,
        "summary": {"avg": value},
        "coverage": {"expectedCount": 4, "observedCount": 4, "percentComplete": coverage},
    }


T0 = datetime(2026, 9, 20, 0, 30, tzinfo=UTC)  # 06:00 IST, end of the 05:00–06:00 IST hour


def hours_payload(records: list[dict]) -> dict[str, Any]:
    return {"meta": {"found": len(records), "limit": 1000, "page": 1}, "results": records}


def openmeteo_payload(
    start: datetime, hours: int, *, null_from: int | None = None
) -> dict[str, Any]:
    times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]

    def series(base: float) -> list[float | None]:
        return [
            None if null_from is not None and i >= null_from else base + i * 0.1
            for i in range(hours)
        ]

    return {
        "latitude": 12.97,
        "longitude": 77.59,
        "timezone": "GMT",
        "hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
        "hourly": {
            "time": times,
            "temperature_2m": series(22.0),
            "relative_humidity_2m": series(70.0),
            "precipitation": series(0.0),
            "wind_speed_10m": series(2.0),
            "wind_direction_10m": series(180.0),
            "surface_pressure": series(912.0),
        },
    }
