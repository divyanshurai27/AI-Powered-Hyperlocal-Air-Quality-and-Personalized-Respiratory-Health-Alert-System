import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, str_enum, utcnow
from app.domain.observations import QualityFlag
from app.domain.pollutants import Pollutant


class IngestionRun(UUIDPrimaryKeyMixin, Base):
    """One execution of a source adapter. Every stored row points back to the run that wrote it."""

    __tablename__ = "ingestion_runs"

    source: Mapped[str] = mapped_column(String(32))
    job: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        String(16), default="running"
    )  # running|success|partial|failed
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    records_fetched: Mapped[int] = mapped_column(Integer, default=0)
    records_valid: Mapped[int] = mapped_column(Integer, default=0)
    records_rejected: Mapped[int] = mapped_column(Integer, default=0)
    records_written: Mapped[int] = mapped_column(Integer, default=0)
    duplicates_dropped: Mapped[int] = mapped_column(Integer, default=0)
    rejection_reasons: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)
    errors: Mapped[list[str]] = mapped_column(JSONB, default=list)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class Station(TimestampMixin, Base):
    """A monitoring location. `id` is namespaced by source, e.g. 'openaq:5548'."""

    __tablename__ = "stations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    source_station_id: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(255))
    latitude: Mapped[float] = mapped_column(Double)
    longitude: Mapped[float] = mapped_column(Double)
    provider: Mapped[str | None] = mapped_column(String(64), default=None)
    # Only reference-grade regulatory monitors (CPCB/KSPCB) are ground truth (PRD §3).
    is_reference_monitor: Mapped[bool] = mapped_column(Boolean, default=False)
    timezone: Mapped[str | None] = mapped_column(String(64), default=None)
    first_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Source-specific details, e.g. OpenAQ sensor IDs per pollutant.
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (UniqueConstraint("source", "source_station_id"),)


class AirQualityObservation(Base):
    """PRD §6.1. `timestamp` is the UTC end of the averaging hour; `concentration` is µg/m³."""

    __tablename__ = "air_quality_observations"
    __table_args__ = (
        UniqueConstraint(
            "source", "station_id", "pollutant", "timestamp", name="uq_aq_obs_identity"
        ),
        Index("ix_aq_obs_pollutant_timestamp", "pollutant", "timestamp"),
        Index("ix_aq_obs_station_pollutant_timestamp", "station_id", "pollutant", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    station_id: Mapped[str] = mapped_column(ForeignKey("stations.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    latitude: Mapped[float] = mapped_column(Double)
    longitude: Mapped[float] = mapped_column(Double)
    pollutant: Mapped[Pollutant] = mapped_column(str_enum(Pollutant, "pollutant"))
    concentration: Mapped[float] = mapped_column(Double)
    unit: Mapped[str] = mapped_column(String(16))
    raw_value: Mapped[float] = mapped_column(Double)
    raw_unit: Mapped[str] = mapped_column(String(16))
    coverage_pct: Mapped[float | None] = mapped_column(Double, default=None)
    quality_flag: Mapped[QualityFlag] = mapped_column(str_enum(QualityFlag, "quality_flag"))
    ingestion_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"), default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, onupdate=utcnow
    )


class WeatherObservation(Base):
    """PRD §6.2. Hourly weather at a grid point.

    kind = 'reanalysis' (ERA5 via Open-Meteo archive: modelled past, not a station reading)
         | 'forecast'   (issued at `issued_at`; only forecasts issued ≤ T may feed a
                         prediction made at T)
    """

    __tablename__ = "weather_observations"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "kind",
            "latitude",
            "longitude",
            "timestamp",
            "issued_at",
            name="uq_weather_identity",
            postgresql_nulls_not_distinct=True,
        ),
        Index("ix_weather_observations_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(16))
    latitude: Mapped[float] = mapped_column(Double)
    longitude: Mapped[float] = mapped_column(Double)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    temperature: Mapped[float | None] = mapped_column(Double, default=None)  # °C
    relative_humidity: Mapped[float | None] = mapped_column(Double, default=None)  # %
    precipitation: Mapped[float | None] = mapped_column(Double, default=None)  # mm, preceding hour
    wind_speed: Mapped[float | None] = mapped_column(Double, default=None)  # m/s at 10 m
    wind_direction: Mapped[float | None] = mapped_column(Double, default=None)  # degrees
    pressure: Mapped[float | None] = mapped_column(Double, default=None)  # hPa, surface
    ingestion_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"), default=None
    )
    notes: Mapped[str | None] = mapped_column(Text, default=None)
