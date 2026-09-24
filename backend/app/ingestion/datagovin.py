"""data.gov.in adapter for CPCB's "Real time Air Quality Index from various locations" feed.

Role: optional live fallback (PRD §31.1). OpenAQ lags real time by days; this feed is the
current CPCB snapshot. It carries no history, so it is never used for training.

UNVERIFIED UNITS: the feed has no unit field, and its min/avg/max values have not yet been
checked against a live response (the portal returned HTTP 502 during development). Until
`UNITS_VERIFIED` is set after that check, `to_raw_observations` refuses to emit records, so
nothing from this source can reach the database with a guessed unit.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.domain.observations import RawObservation
from app.ingestion.base import SourceError, request_json

SOURCE = "cpcb_datagovin"
IST = ZoneInfo("Asia/Kolkata")
UNITS_VERIFIED = False

# Assumed units per pollutant_id, to be confirmed by `probe` before UNITS_VERIFIED is set.
ASSUMED_UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO2": "µg/m³",
    "SO2": "µg/m³",
    "OZONE": "µg/m³",
    "CO": "mg/m³",
}


def _float_or_none(v: Any) -> float | None:
    if v in (None, "", "NA", "na", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_last_update(value: str | None) -> datetime | None:
    """Feed format is 'DD-MM-YYYY HH:MM:SS' in IST."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%d-%m-%Y %H:%M:%S").replace(tzinfo=IST)
    except ValueError:
        return None


def station_id_for(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
    return f"{SOURCE}:{slug[:48]}"


def to_raw_observations(records: list[dict[str, Any]]) -> list[RawObservation]:
    if not UNITS_VERIFIED:
        raise SourceError(SOURCE, "units not verified against a live response; refusing to ingest")
    out = []
    for r in records:
        pid = (r.get("pollutant_id") or "").strip().upper()
        out.append(
            RawObservation(
                source=SOURCE,
                station_id=station_id_for(r["station"]) if r.get("station") else None,
                timestamp=parse_last_update(r.get("last_update")),
                latitude=_float_or_none(r.get("latitude")),
                longitude=_float_or_none(r.get("longitude")),
                pollutant=pid or None,
                value=_float_or_none(r.get("avg_value")),
                unit=ASSUMED_UNITS.get(pid),
            )
        )
    return out


class DataGovInClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        resource_id: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise SourceError(SOURCE, "DATA_GOV_IN_API_KEY is not set")
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/resource/{resource_id}"
        self._client = httpx.Client(timeout=timeout, transport=transport)
        # The portal is slow; fail fast and let the caller fall back rather than retrying long.
        self._kwargs: dict[str, Any] = {"max_attempts": 2, **({"sleep": sleep} if sleep else {})}

    def close(self) -> None:
        self._client.close()

    def fetch_current(self, city: str, limit: int = 500) -> list[dict[str, Any]]:
        params = {"api-key": self._api_key, "format": "json", "limit": limit, "filters[city]": city}
        body = request_json(self._client, SOURCE, self._url, params=params, **self._kwargs)
        records = body.get("records") if isinstance(body, dict) else None
        if not isinstance(records, list):
            raise SourceError(SOURCE, "response has no records array")
        return records
