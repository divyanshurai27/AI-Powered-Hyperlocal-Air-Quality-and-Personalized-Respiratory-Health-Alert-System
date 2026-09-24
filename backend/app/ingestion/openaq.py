"""OpenAQ v3 adapter. OpenAQ republishes CPCB/KSPCB regulatory monitor data for India, so
observations from reference providers are CPCB ground truth accessed via OpenAQ (PRD §3).

Uses /sensors/{id}/hours: OpenAQ's hourly means, each with a coverage percentage.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from app.core.logging import get_logger
from app.domain.observations import RawObservation
from app.domain.pollutants import Pollutant, parse_pollutant
from app.ingestion.base import SourceError, request_json

logger = get_logger(__name__)

SOURCE = "openaq"
PAGE_LIMIT = 1000
# Providers that operate regulatory reference monitors. Low-cost networks (AirGradient,
# PurpleAir, ...) are stored for context but are never used as ground truth.
REFERENCE_PROVIDERS = {"CPCB", "caaqm"}

# Source-specific unit corrections, applied before validation. Each entry needs evidence.
# CO labelled "ppb" on the 2025+ CPCB feed: median 0.49 at BTM Layout (Jun–Sep 2025) vs 1260
# µg/m³ on the legacy feed (Jun–Sep 2021). Real ambient CO is ~400–1500 ppb, so the labels
# are wrong; the values match CPCB's native reporting unit, mg/m³. (Checked 2026-09-25.)
# NO2/SO2 "ppb" labels are NOT corrected: the evidence is ambiguous; see docs/data_and_methods.md.
UNIT_OVERRIDES: dict[tuple[str, str], str] = {("co", "ppb"): "mg/m³"}


@dataclass(frozen=True)
class SensorInfo:
    sensor_id: int
    pollutant: Pollutant
    unit: str
    first_utc: datetime | None
    last_utc: datetime | None

    def overlaps(self, start: datetime, end: datetime) -> bool:
        if self.first_utc is None or self.last_utc is None:
            return True  # unknown range: ask the API rather than silently skip
        return self.first_utc <= end and self.last_utc >= start


@dataclass(frozen=True)
class StationInfo:
    location_id: int
    name: str
    latitude: float
    longitude: float
    provider: str | None
    is_monitor: bool
    timezone: str | None
    first_utc: datetime | None
    last_utc: datetime | None
    sensors: list[SensorInfo] = field(default_factory=list)

    @property
    def station_id(self) -> str:
        return f"{SOURCE}:{self.location_id}"

    @property
    def is_reference(self) -> bool:
        return self.provider in REFERENCE_PROVIDERS


def _parse_utc(block: dict[str, Any] | None) -> datetime | None:
    if not block or not block.get("utc"):
        return None
    return datetime.fromisoformat(block["utc"].replace("Z", "+00:00"))


def parse_location(loc: dict[str, Any]) -> StationInfo:
    coords = loc.get("coordinates") or {}
    sensors = []
    for s in loc.get("sensors") or []:
        param = s.get("parameter") or {}
        pollutant = parse_pollutant(param.get("name"))
        if pollutant is None:
            continue  # weather channels, pm1, nox... not in the AeroGard pollutant set
        sensors.append(
            SensorInfo(
                sensor_id=int(s["id"]),
                pollutant=pollutant,
                unit=param.get("units") or "",
                first_utc=_parse_utc(s.get("datetimeFirst")),
                last_utc=_parse_utc(s.get("datetimeLast")),
            )
        )
    return StationInfo(
        location_id=int(loc["id"]),
        name=loc.get("name") or f"location {loc['id']}",
        latitude=float(coords["latitude"]),
        longitude=float(coords["longitude"]),
        provider=(loc.get("provider") or {}).get("name"),
        is_monitor=bool(loc.get("isMonitor")),
        timezone=loc.get("timezone"),
        first_utc=_parse_utc(loc.get("datetimeFirst")),
        last_utc=_parse_utc(loc.get("datetimeLast")),
        sensors=sensors,
    )


def normalize_hour(record: dict[str, Any], station: StationInfo) -> RawObservation:
    """Map one /hours record to the canonical raw shape. Missing fields become None so the
    validator rejects them with a reason instead of this function raising."""
    param = record.get("parameter") or {}
    period = record.get("period") or {}
    coverage = record.get("coverage") or {}
    value = record.get("value")
    return RawObservation(
        source=SOURCE,
        station_id=station.station_id,
        timestamp=_parse_utc(period.get("datetimeTo")),
        latitude=station.latitude,
        longitude=station.longitude,
        pollutant=param.get("name"),
        value=float(value) if isinstance(value, int | float) else None,
        unit=UNIT_OVERRIDES.get(
            (param.get("name") or "", param.get("units") or ""), param.get("units")
        ),
        coverage_pct=coverage.get("percentComplete"),
        provider_flagged=bool((record.get("flagInfo") or {}).get("hasFlags")),
    )


class OpenAQClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise SourceError(SOURCE, "OPENAQ_API_KEY is not set")
        self._client = httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )
        self._sleep_kwargs = {"sleep": sleep} if sleep else {}

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        body = request_json(self._client, SOURCE, path, params=params, **self._sleep_kwargs)
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise SourceError(SOURCE, f"unexpected response structure from {path}")
        return body

    def _paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        page = 1
        while True:
            body = self._get(path, {**params, "limit": PAGE_LIMIT, "page": page})
            results = body["results"]
            yield from results
            if len(results) < PAGE_LIMIT:
                return
            page += 1

    def list_locations(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[StationInfo]:
        params = {"coordinates": f"{latitude},{longitude}", "radius": int(radius_km * 1000)}
        stations = []
        for loc in self._paginate("/locations", params):
            try:
                stations.append(parse_location(loc))
            except (KeyError, TypeError, ValueError):
                logger.warning("openaq_location_malformed", extra={"location": str(loc)[:200]})
        return stations

    def fetch_sensor_hours(
        self, station: StationInfo, sensor: SensorInfo, start: datetime, end: datetime
    ) -> list[RawObservation]:
        params = {"datetime_from": start.isoformat(), "datetime_to": end.isoformat()}
        return [
            normalize_hour(rec, station)
            for rec in self._paginate(f"/sensors/{sensor.sensor_id}/hours", params)
        ]
