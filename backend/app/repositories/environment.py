import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.domain.observations import Observation, QualityFlag
from app.domain.pollutants import Pollutant
from app.domain.spatial import StationReading
from app.ingestion.openmeteo import WeatherRecord
from app.models import AirQualityObservation, IngestionRun, Station, WeatherObservation

BATCH_SIZE = 1000


def _batches(rows: Sequence[dict[str, Any]], size: int = BATCH_SIZE) -> Iterable[Sequence[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


class IngestionRunRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def start(self, source: str, job: str, params: dict[str, Any]) -> IngestionRun:
        run = IngestionRun(source=source, job=job, params=params, status="running")
        self.db.add(run)
        self.db.commit()
        return run

    def finish(self, run: IngestionRun) -> IngestionRun:
        run.finished_at = datetime.now(UTC)
        self.db.add(run)
        self.db.commit()
        return run


class StationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        stmt = insert(Station).values(rows)
        updatable = {
            c: stmt.excluded[c]
            for c in rows[0]
            if c not in ("id", "source", "source_station_id", "created_at")
        }
        stmt = stmt.on_conflict_do_update(index_elements=[Station.id], set_=updatable)
        self.db.execute(stmt)
        return len(rows)

    def list(self, *, reference_only: bool = False) -> list[Station]:
        q = select(Station).order_by(Station.id)
        if reference_only:
            q = q.where(Station.is_reference_monitor.is_(True))
        return list(self.db.scalars(q))


class ObservationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_many(self, observations: list[Observation], run_id: uuid.UUID | None) -> int:
        """Insert canonical observations; on identity conflict, replace only if the new record
        is better (VALID beats flagged, then higher coverage). Returns rows inserted or updated.

        Re-running the same ingestion is therefore idempotent (PRD §23).
        """
        rows = [
            {
                "source": o.source,
                "station_id": o.station_id,
                "timestamp": o.timestamp,
                "latitude": o.latitude,
                "longitude": o.longitude,
                "pollutant": o.pollutant.value,
                "concentration": o.concentration,
                "unit": o.unit,
                "raw_value": o.raw_value,
                "raw_unit": o.raw_unit,
                "coverage_pct": o.coverage_pct,
                "quality_flag": o.quality_flag.value,
                "ingestion_run_id": run_id,
            }
            for o in observations
        ]
        written = 0
        table = AirQualityObservation.__table__
        for batch in _batches(rows):
            stmt = insert(AirQualityObservation).values(list(batch))
            new, old = stmt.excluded, table.c
            new_is_better = or_(
                and_(
                    old.quality_flag != QualityFlag.VALID.value,
                    new.quality_flag == QualityFlag.VALID.value,
                ),
                and_(
                    (old.quality_flag == QualityFlag.VALID.value)
                    == (new.quality_flag == QualityFlag.VALID.value),
                    func.coalesce(new.coverage_pct, -1) > func.coalesce(old.coverage_pct, -1),
                ),
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_aq_obs_identity",
                set_={
                    c: new[c]
                    for c in (
                        "concentration",
                        "raw_value",
                        "raw_unit",
                        "coverage_pct",
                        "quality_flag",
                        "ingestion_run_id",
                    )
                }
                | {"updated_at": func.now()},
                where=new_is_better,
            )
            written += len(self.db.execute(stmt.returning(AirQualityObservation.id)).all())
        return written

    def latest_readings(
        self,
        pollutant: Pollutant,
        at: datetime,
        *,
        lookback: timedelta = timedelta(days=30),
        source: str | None = None,
    ) -> list[StationReading]:
        """Newest reading per reference station at or before `at` (never after: BR-04)."""
        o = AirQualityObservation
        q = (
            select(o, Station)
            .join(Station, Station.id == o.station_id)
            .where(
                o.pollutant == pollutant,
                o.timestamp <= at,
                o.timestamp > at - lookback,
                Station.is_reference_monitor.is_(True),
            )
            .distinct(o.station_id)
            .order_by(o.station_id, o.timestamp.desc())
        )
        if source:
            q = q.where(o.source == source)
        return [
            StationReading(
                station_id=obs.station_id,
                latitude=st.latitude,
                longitude=st.longitude,
                concentration=obs.concentration,
                timestamp=obs.timestamp,
                quality_flag=QualityFlag(obs.quality_flag),
                coverage_pct=obs.coverage_pct,
            )
            for obs, st in self.db.execute(q).all()
        ]

    def history(
        self, station_id: str, pollutant: Pollutant, start: datetime, end: datetime
    ) -> list[AirQualityObservation]:
        o = AirQualityObservation
        q = (
            select(o)
            .where(o.station_id == station_id, o.pollutant == pollutant)
            .where(o.timestamp > start, o.timestamp <= end)
            .order_by(o.timestamp)
        )
        return list(self.db.scalars(q))

    def coverage_summary(self) -> list[dict[str, Any]]:
        o = AirQualityObservation
        q = (
            select(
                o.station_id,
                o.pollutant,
                func.count().label("rows"),
                func.min(o.timestamp).label("first"),
                func.max(o.timestamp).label("last"),
                func.count().filter(o.quality_flag != QualityFlag.VALID.value).label("flagged"),
            )
            .group_by(o.station_id, o.pollutant)
            .order_by(o.station_id, o.pollutant)
        )
        return [dict(r._mapping) for r in self.db.execute(q)]


class WeatherRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_many(
        self, records: list[WeatherRecord], source: str, run_id: uuid.UUID | None
    ) -> int:
        rows = [{**asdict(r), "source": source, "ingestion_run_id": run_id} for r in records]
        written = 0
        values = (
            "temperature",
            "relative_humidity",
            "precipitation",
            "wind_speed",
            "wind_direction",
            "pressure",
            "ingestion_run_id",
        )
        for batch in _batches(rows):
            stmt = insert(WeatherObservation).values(list(batch))
            stmt = stmt.on_conflict_do_update(
                constraint="uq_weather_identity", set_={c: stmt.excluded[c] for c in values}
            )
            written += len(self.db.execute(stmt.returning(WeatherObservation.id)).all())
        return written
