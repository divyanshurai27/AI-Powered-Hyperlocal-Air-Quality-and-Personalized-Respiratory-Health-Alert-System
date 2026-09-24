"""Canonical air-quality observation, validation and deduplication (PRD §6.1, §8).

Timestamp convention: `timestamp` is the END of the averaging period, in UTC. An hourly value
stamped 06:00 IST (00:30 UTC) is the mean of 05:00–06:00 IST, so it is fully known at
`timestamp` and can safely be used by any prediction made at or after that instant (BR-04).
"""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from app.domain.pollutants import (
    CANONICAL_UNIT,
    PLAUSIBLE_MAX_UG_M3,
    Pollutant,
    UnitConversionError,
    parse_pollutant,
    to_canonical,
)

# CPCB's data-capture criterion: an hourly mean needs ≥75% of its sub-hourly readings.
MIN_COVERAGE_PCT = 75.0


class QualityFlag(StrEnum):
    VALID = "valid"
    LOW_COVERAGE = "low_coverage"
    PROVIDER_FLAGGED = "provider_flagged"
    IMPLAUSIBLE = "implausible"


# Highest severity wins when several apply.
_FLAG_SEVERITY = [
    QualityFlag.IMPLAUSIBLE,
    QualityFlag.PROVIDER_FLAGGED,
    QualityFlag.LOW_COVERAGE,
    QualityFlag.VALID,
]


class RejectReason(StrEnum):
    MISSING_TIMESTAMP = "missing_timestamp"
    NAIVE_TIMESTAMP = "naive_timestamp"
    MISSING_POLLUTANT = "missing_pollutant"
    UNSUPPORTED_POLLUTANT = "unsupported_pollutant"
    INVALID_COORDINATES = "invalid_coordinates"
    MISSING_VALUE = "missing_value"
    NON_FINITE_VALUE = "non_finite_value"
    NEGATIVE_VALUE = "negative_value"
    INVALID_UNIT = "invalid_unit"
    MISSING_STATION = "missing_station"


@dataclass(frozen=True)
class RawObservation:
    """A source record after field mapping, before any validation or unit conversion."""

    source: str
    station_id: str | None
    timestamp: datetime | None
    latitude: float | None
    longitude: float | None
    pollutant: str | None
    value: float | None
    unit: str | None
    coverage_pct: float | None = None
    provider_flagged: bool = False


@dataclass(frozen=True)
class Observation:
    """Canonical, validated observation. `concentration` is always in µg/m³."""

    source: str
    station_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    pollutant: Pollutant
    concentration: float
    unit: str
    raw_value: float
    raw_unit: str
    coverage_pct: float | None
    quality_flag: QualityFlag

    @property
    def identity(self) -> tuple[str, str, Pollutant, datetime]:
        return (self.source, self.station_id, self.pollutant, self.timestamp)


@dataclass
class ValidationOutcome:
    observation: Observation | None = None
    reasons: list[RejectReason] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.observation is not None


def valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    if latitude is None or longitude is None:
        return False
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return False
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def normalize_timestamp(ts: datetime) -> datetime:
    """Canonical internal representation: timezone-aware UTC, no sub-second noise."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("naive timestamps are ambiguous and not accepted")
    return ts.astimezone(UTC).replace(microsecond=0)


def validate_observation(raw: RawObservation) -> ValidationOutcome:
    """Reject records that can't be trusted; flag (but keep) records that are merely doubtful."""
    reasons: list[RejectReason] = []

    if not raw.station_id:
        reasons.append(RejectReason.MISSING_STATION)

    ts: datetime | None = None
    if raw.timestamp is None:
        reasons.append(RejectReason.MISSING_TIMESTAMP)
    else:
        try:
            ts = normalize_timestamp(raw.timestamp)
        except ValueError:
            reasons.append(RejectReason.NAIVE_TIMESTAMP)

    pollutant = parse_pollutant(raw.pollutant)
    if not raw.pollutant:
        reasons.append(RejectReason.MISSING_POLLUTANT)
    elif pollutant is None:
        reasons.append(RejectReason.UNSUPPORTED_POLLUTANT)

    if not valid_coordinates(raw.latitude, raw.longitude):
        reasons.append(RejectReason.INVALID_COORDINATES)

    if raw.value is None:
        reasons.append(RejectReason.MISSING_VALUE)
    elif not math.isfinite(raw.value):
        reasons.append(RejectReason.NON_FINITE_VALUE)
    elif raw.value < 0:
        reasons.append(RejectReason.NEGATIVE_VALUE)

    concentration: float | None = None
    if pollutant is not None and raw.value is not None and not reasons:
        try:
            concentration = to_canonical(pollutant, raw.value, raw.unit or "")
        except UnitConversionError:
            reasons.append(RejectReason.INVALID_UNIT)

    if reasons:
        return ValidationOutcome(reasons=reasons)

    assert ts is not None and pollutant is not None and concentration is not None  # noqa: S101
    flags = {QualityFlag.VALID}
    if raw.coverage_pct is not None and raw.coverage_pct < MIN_COVERAGE_PCT:
        flags.add(QualityFlag.LOW_COVERAGE)
    if raw.provider_flagged:
        flags.add(QualityFlag.PROVIDER_FLAGGED)
    if concentration > PLAUSIBLE_MAX_UG_M3[pollutant]:
        flags.add(QualityFlag.IMPLAUSIBLE)
    flag = next(f for f in _FLAG_SEVERITY if f in flags)

    return ValidationOutcome(
        observation=Observation(
            source=raw.source,
            station_id=raw.station_id,  # type: ignore[arg-type]
            timestamp=ts,
            latitude=raw.latitude,  # type: ignore[arg-type]
            longitude=raw.longitude,  # type: ignore[arg-type]
            pollutant=pollutant,
            concentration=round(concentration, 4),
            unit=CANONICAL_UNIT,
            raw_value=raw.value,  # type: ignore[arg-type]
            raw_unit=raw.unit or "",
            coverage_pct=raw.coverage_pct,
            quality_flag=flag,
        )
    )


@dataclass
class DedupResult:
    observations: list[Observation]
    duplicates_dropped: int
    conflicts: int  # same identity, different concentration


def _preference(o: Observation) -> tuple[bool, float]:
    # Prefer a clean flag, then the more complete hourly mean.
    return (
        o.quality_flag == QualityFlag.VALID,
        o.coverage_pct if o.coverage_pct is not None else -1,
    )


def deduplicate_observations(observations: list[Observation]) -> DedupResult:
    """One record per (source, station, pollutant, timestamp).

    Conflict rule (documented, PRD §8.1): keep the record with a VALID flag, then the higher
    coverage; on a full tie keep the first seen. Conflicts are counted so they show up in the
    ingestion run report instead of disappearing.
    """
    kept: dict[tuple, Observation] = {}
    duplicates = conflicts = 0
    for obs in observations:
        existing = kept.get(obs.identity)
        if existing is None:
            kept[obs.identity] = obs
            continue
        duplicates += 1
        if not math.isclose(existing.concentration, obs.concentration, rel_tol=1e-9):
            conflicts += 1
        if _preference(obs) > _preference(existing):
            kept[obs.identity] = obs
    return DedupResult(list(kept.values()), duplicates, conflicts)
