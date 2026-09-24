"""Open-Meteo weather adapter (no API key; free for non-commercial use).

- archive: ERA5 reanalysis. Modelled, not station-measured, and typically ~5 days behind real
  time; recent hours come back null and are stored as missing, not filled.
- forecast: numerical weather forecast. Each row keeps `issued_at` so a prediction made at T
  only ever uses forecasts that existed at T (BR-04).

Times are requested in GMT, so rows sit on whole UTC hours. AQ hours end at :30 UTC; the
feature builder aligns the two by taking the latest weather row at or before the AQ timestamp.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx

from app.domain.observations import valid_coordinates
from app.ingestion.base import SourceError, request_json

SOURCE = "open-meteo"

# Open-Meteo variable → WeatherObservation column
HOURLY_VARIABLES = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative_humidity",
    "precipitation": "precipitation",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "surface_pressure": "pressure",
}


@dataclass(frozen=True)
class WeatherRecord:
    kind: str  # "reanalysis" | "forecast"
    latitude: float
    longitude: float
    timestamp: datetime
    issued_at: datetime | None
    temperature: float | None
    relative_humidity: float | None
    precipitation: float | None
    wind_speed: float | None
    wind_direction: float | None
    pressure: float | None

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, col) is None for col in HOURLY_VARIABLES.values())


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) else None


def parse_hourly(
    body: dict[str, Any],
    *,
    kind: str,
    latitude: float,
    longitude: float,
    issued_at: datetime | None,
) -> list[WeatherRecord]:
    hourly = body.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise SourceError(SOURCE, "response has no hourly.time array")
    times = hourly["time"]
    columns = {}
    for var, col in HOURLY_VARIABLES.items():
        values = hourly.get(var)
        if values is None:
            values = [None] * len(times)  # variable not offered for this model: explicit missing
        if len(values) != len(times):
            raise SourceError(
                SOURCE, f"hourly.{var} length {len(values)} != time length {len(times)}"
            )
        columns[col] = values

    records = []
    for i, t in enumerate(times):
        ts = datetime.fromisoformat(t).replace(tzinfo=UTC)
        rec = WeatherRecord(
            kind=kind,
            latitude=latitude,
            longitude=longitude,
            timestamp=ts,
            issued_at=issued_at,
            **{col: _num(values[i]) for col, values in columns.items()},
        )
        if not rec.is_empty:
            records.append(rec)
    return records


class OpenMeteoClient:
    def __init__(
        self,
        archive_url: str,
        forecast_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._archive_url = archive_url
        self._forecast_url = forecast_url
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._sleep_kwargs = {"sleep": sleep} if sleep else {}

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _point(latitude: float, longitude: float) -> tuple[float, float]:
        if not valid_coordinates(latitude, longitude):
            raise SourceError(SOURCE, f"invalid coordinates {latitude},{longitude}")
        # Rounded so repeated requests for the same place share one identity in the DB.
        return round(latitude, 4), round(longitude, 4)

    def _params(self, latitude: float, longitude: float) -> dict[str, Any]:
        return {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        }

    def fetch_archive(
        self, latitude: float, longitude: float, start: date, end: date
    ) -> list[WeatherRecord]:
        lat, lon = self._point(latitude, longitude)
        params = {
            **self._params(lat, lon),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        body = request_json(
            self._client, SOURCE, self._archive_url, params=params, **self._sleep_kwargs
        )
        return parse_hourly(body, kind="reanalysis", latitude=lat, longitude=lon, issued_at=None)

    def fetch_forecast(
        self, latitude: float, longitude: float, issued_at: datetime, hours: int = 48
    ) -> list[WeatherRecord]:
        lat, lon = self._point(latitude, longitude)
        params = {**self._params(lat, lon), "forecast_hours": hours, "past_hours": 0}
        body = request_json(
            self._client, SOURCE, self._forecast_url, params=params, **self._sleep_kwargs
        )
        issued = issued_at.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        return parse_hourly(body, kind="forecast", latitude=lat, longitude=lon, issued_at=issued)
