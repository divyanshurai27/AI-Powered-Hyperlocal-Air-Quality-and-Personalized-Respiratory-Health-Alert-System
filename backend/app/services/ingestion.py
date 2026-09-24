"""Ingestion pipeline: fetch → normalize → validate → deduplicate → persist (PRD §8)."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.domain.observations import RawObservation, deduplicate_observations, validate_observation
from app.domain.pollutants import Pollutant
from app.ingestion import openaq, openmeteo
from app.ingestion.base import SourceError
from app.models import IngestionRun
from app.repositories.environment import (
    IngestionRunRepository,
    ObservationRepository,
    StationRepository,
    WeatherRepository,
)

logger = get_logger(__name__)

DEFAULT_POLLUTANTS = tuple(Pollutant)
CHUNK_DAYS = 90


@dataclass
class BatchStats:
    fetched: int = 0
    valid: int = 0
    rejected: int = 0
    duplicates: int = 0
    written: int = 0
    reasons: Counter = field(default_factory=Counter)


def process_raw(raw: list[RawObservation]) -> tuple[list, BatchStats]:
    """Pure validation + dedup step, shared by every AQ source."""
    stats = BatchStats(fetched=len(raw))
    valid = []
    for r in raw:
        outcome = validate_observation(r)
        if outcome.ok:
            valid.append(outcome.observation)
        else:
            stats.rejected += 1
            stats.reasons.update(reason.value for reason in outcome.reasons)
    dedup = deduplicate_observations(valid)
    stats.valid = len(dedup.observations)
    stats.duplicates = dedup.duplicates_dropped
    return dedup.observations, stats


def _apply(run: IngestionRun, stats: BatchStats) -> None:
    run.records_fetched += stats.fetched
    run.records_valid += stats.valid
    run.records_rejected += stats.rejected
    run.duplicates_dropped += stats.duplicates
    run.records_written += stats.written
    merged = Counter(run.rejection_reasons or {})
    merged.update(stats.reasons)
    run.rejection_reasons = dict(merged)


class IngestionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.runs = IngestionRunRepository(db)
        self.stations = StationRepository(db)
        self.observations = ObservationRepository(db)
        self.weather = WeatherRepository(db)

    # --- stations -------------------------------------------------------------------------

    def sync_openaq_stations(
        self, client: openaq.OpenAQClient, latitude: float, longitude: float, radius_km: float
    ) -> list[openaq.StationInfo]:
        found = client.list_locations(latitude, longitude, radius_km)
        rows = [
            {
                "id": s.station_id,
                "source": openaq.SOURCE,
                "source_station_id": str(s.location_id),
                "name": s.name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "provider": s.provider,
                "is_reference_monitor": s.is_reference,
                "timezone": s.timezone,
                "first_observed_at": s.first_utc,
                "last_observed_at": s.last_utc,
                "source_metadata": {
                    "is_monitor": s.is_monitor,
                    "sensors": [
                        {
                            "id": x.sensor_id,
                            "pollutant": x.pollutant.value,
                            "unit": x.unit,
                            "first": x.first_utc.isoformat() if x.first_utc else None,
                            "last": x.last_utc.isoformat() if x.last_utc else None,
                        }
                        for x in s.sensors
                    ],
                },
            }
            for s in found
        ]
        self.stations.upsert(rows)
        self.db.commit()
        logger.info(
            "stations_synced",
            extra={"total": len(found), "reference": sum(s.is_reference for s in found)},
        )
        return found

    # --- air quality ----------------------------------------------------------------------

    def ingest_openaq_hours(
        self,
        client: openaq.OpenAQClient,
        stations: list[openaq.StationInfo],
        start: datetime,
        end: datetime,
        pollutants: tuple[Pollutant, ...] = DEFAULT_POLLUTANTS,
    ) -> IngestionRun:
        """Backfill/refresh hourly observations for reference stations in [start, end].

        Each sensor is fetched in CHUNK_DAYS windows: OpenAQ times out (HTTP 408) on long
        ranges. Commits after each chunk, so a long backfill keeps its progress and a rerun
        only rewrites what changed. One chunk failing marks the run `partial`, not `failed`.
        """
        targets = [
            (st, se)
            for st in stations
            if st.is_reference
            for se in st.sensors
            if se.pollutant in pollutants and se.overlaps(start, end)
        ]
        run = self.runs.start(
            openaq.SOURCE,
            "hours",
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "pollutants": [p.value for p in pollutants],
                "sensors": len(targets),
            },
        )
        chunks = failures = 0
        for station, sensor in targets:
            for lo, hi in _windows(start, end, timedelta(days=CHUNK_DAYS)):
                chunks += 1
                try:
                    raw = client.fetch_sensor_hours(station, sensor, lo, hi)
                except SourceError as exc:
                    failures += 1
                    span = f"{lo:%Y-%m-%d}..{hi:%Y-%m-%d}"
                    run.errors = [
                        *run.errors,
                        f"sensor {sensor.sensor_id} ({station.name}) {span}: {exc}",
                    ]
                    logger.warning(
                        "sensor_fetch_failed", extra={"sensor": sensor.sensor_id, "error": str(exc)}
                    )
                    self.db.commit()
                    continue
                observations, stats = process_raw(raw)
                stats.written = self.observations.upsert_many(observations, run.id)
                _apply(run, stats)
                self.db.commit()
                logger.info(
                    "sensor_ingested",
                    extra={
                        "station": station.station_id,
                        "sensor": sensor.sensor_id,
                        "pollutant": sensor.pollutant.value,
                        "window": f"{lo:%Y-%m-%d}..{hi:%Y-%m-%d}",
                        "fetched": stats.fetched,
                        "written": stats.written,
                        "rejected": stats.rejected,
                    },
                )
        run.status = _status(chunks, failures)
        return self.runs.finish(run)

    # --- weather --------------------------------------------------------------------------

    def ingest_weather_archive(
        self,
        client: openmeteo.OpenMeteoClient,
        latitude: float,
        longitude: float,
        start: date,
        end: date,
        chunk_days: int = 120,
    ) -> IngestionRun:
        # The archive only serves up to yesterday (UTC); later days come from forecasts.
        end = min(end, datetime.now(UTC).date() - timedelta(days=1))
        run = self.runs.start(
            openmeteo.SOURCE,
            "archive",
            {"lat": latitude, "lon": longitude, "start": start.isoformat(), "end": end.isoformat()},
        )
        chunks = failures = 0
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            chunks += 1
            try:
                records = client.fetch_archive(latitude, longitude, cursor, chunk_end)
            except SourceError as exc:
                failures += 1
                run.errors = [*run.errors, f"{cursor}..{chunk_end}: {exc}"]
            else:
                run.records_fetched += len(records)
                run.records_valid += len(records)
                run.records_written += self.weather.upsert_many(records, openmeteo.SOURCE, run.id)
            self.db.commit()
            cursor = chunk_end + timedelta(days=1)
        run.status = _status(chunks, failures)
        return self.runs.finish(run)

    def ingest_weather_forecast(
        self,
        client: openmeteo.OpenMeteoClient,
        latitude: float,
        longitude: float,
        issued_at: datetime | None = None,
    ) -> IngestionRun:
        issued = issued_at or datetime.now(UTC)
        run = self.runs.start(openmeteo.SOURCE, "forecast", {"lat": latitude, "lon": longitude})
        try:
            records = client.fetch_forecast(latitude, longitude, issued)
        except SourceError as exc:
            run.errors = [str(exc)]
            run.status = "failed"
            return self.runs.finish(run)
        run.records_fetched = run.records_valid = len(records)
        run.records_written = self.weather.upsert_many(records, openmeteo.SOURCE, run.id)
        run.status = "success"
        return self.runs.finish(run)


def _windows(start: datetime, end: datetime, size: timedelta) -> list[tuple[datetime, datetime]]:
    """Consecutive (lo, hi] windows covering [start, end]."""
    out, lo = [], start
    while lo < end:
        hi = min(lo + size, end)
        out.append((lo, hi))
        lo = hi
    return out or [(start, end)]


def _status(attempted: int, failures: int) -> str:
    if failures == 0:
        return "success"
    return "failed" if failures >= attempted else "partial"
