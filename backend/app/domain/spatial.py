"""Hyperlocal matching: user coordinates → nearby reference-monitor readings (PRD §10).

"Hyperlocal" here means location-aware *estimation* from nearby monitors, distance, time
alignment and data quality. It is not a street-level measurement.

Method (versioned as SPATIAL_METHOD_VERSION):
  1. candidates = stations within `radius_km` (haversine great-circle distance)
  2. drop readings that are stale (older than `max_age`), from the future, or implausible
  3. if the nearest usable station is within `exact_match_km`, use it alone
  4. otherwise inverse-distance weighting (power 2) over the `max_stations` nearest usable
     stations, with low-coverage / provider-flagged readings down-weighted by 0.5
  5. if nothing usable remains, return an explicit unavailable result, never a guess
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from app.domain.observations import QualityFlag, valid_coordinates

SPATIAL_METHOD_VERSION = "spatial_idw_v1.0.0"
EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius

IDW_POWER = 2.0
DEGRADED_QUALITY_WEIGHT = 0.5
# Stops a station at ~0 km from getting infinite weight; below this we use exact match anyway.
MIN_DISTANCE_KM = 0.05


class UnavailableReason(StrEnum):
    INVALID_LOCATION = "invalid_location"
    NO_STATIONS_IN_RADIUS = "no_stations_in_radius"
    NO_FRESH_DATA = "no_fresh_data"
    NO_USABLE_DATA = "no_usable_data"


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class StationReading:
    """Latest reading of one pollutant at one station, as seen at the query time."""

    station_id: str
    latitude: float
    longitude: float
    concentration: float
    timestamp: datetime
    quality_flag: QualityFlag
    coverage_pct: float | None = None


@dataclass(frozen=True)
class Candidate:
    reading: StationReading
    distance_km: float


@dataclass(frozen=True)
class Contribution:
    station_id: str
    distance_km: float
    weight: float
    concentration: float
    timestamp: datetime
    quality_flag: QualityFlag


@dataclass(frozen=True)
class LocalEstimate:
    concentration: float
    method: str  # "exact_station" | "idw"
    method_version: str
    contributions: list[Contribution]
    oldest_input: datetime
    nearest_distance_km: float


@dataclass(frozen=True)
class Unavailable:
    reason: UnavailableReason
    detail: str


def validate_coordinates(latitude: float, longitude: float) -> bool:
    return valid_coordinates(latitude, longitude)


def calculate_distance_km(a: GeoPoint, b: GeoPoint) -> float:
    """Haversine great-circle distance on a spherical Earth (error < 0.5% for our scales)."""
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def find_candidate_stations(
    location: GeoPoint, readings: list[StationReading], radius_km: float
) -> list[Candidate]:
    """All readings within the radius, nearest first."""
    out = []
    for r in readings:
        d = calculate_distance_km(location, GeoPoint(r.latitude, r.longitude))
        if d <= radius_km:
            out.append(Candidate(r, d))
    return sorted(out, key=lambda c: (c.distance_km, c.reading.station_id))


def is_usable(reading: StationReading, at: datetime, max_age: timedelta) -> bool:
    if reading.quality_flag == QualityFlag.IMPLAUSIBLE:
        return False
    if reading.timestamp > at:  # a value from after the query time would be leakage (BR-04)
        return False
    return at - reading.timestamp <= max_age


def quality_weight(flag: QualityFlag) -> float:
    return 1.0 if flag == QualityFlag.VALID else DEGRADED_QUALITY_WEIGHT


def spatial_weight(candidate: Candidate) -> float:
    d = max(candidate.distance_km, MIN_DISTANCE_KM)
    return quality_weight(candidate.reading.quality_flag) / (d**IDW_POWER)


def select_representative_station(
    candidates: list[Candidate], at: datetime, max_age: timedelta
) -> Candidate | None:
    """Nearest usable station; ties broken by cleaner flag, then higher coverage, then recency."""
    usable = [c for c in candidates if is_usable(c.reading, at, max_age)]
    if not usable:
        return None
    return min(
        usable,
        key=lambda c: (
            round(c.distance_km, 3),
            c.reading.quality_flag != QualityFlag.VALID,
            -(c.reading.coverage_pct or 0.0),
            -c.reading.timestamp.timestamp(),
        ),
    )


def estimate_local_air_quality(
    location: GeoPoint,
    readings: list[StationReading],
    at: datetime,
    *,
    radius_km: float,
    max_age: timedelta,
    exact_match_km: float = 0.5,
    max_stations: int = 4,
) -> LocalEstimate | Unavailable:
    if not validate_coordinates(location.latitude, location.longitude):
        return Unavailable(UnavailableReason.INVALID_LOCATION, "coordinates out of range")

    candidates = find_candidate_stations(location, readings, radius_km)
    if not candidates:
        return Unavailable(
            UnavailableReason.NO_STATIONS_IN_RADIUS, f"no monitoring station within {radius_km} km"
        )

    not_implausible = [c for c in candidates if c.reading.quality_flag != QualityFlag.IMPLAUSIBLE]
    usable = [c for c in not_implausible if is_usable(c.reading, at, max_age)]
    if not usable:
        if not not_implausible:
            return Unavailable(UnavailableReason.NO_USABLE_DATA, "all nearby readings failed QC")
        past = [c.reading.timestamp for c in not_implausible if c.reading.timestamp <= at]
        detail = (
            f"newest nearby reading is {max(past).isoformat()}, older than {max_age}"
            if past
            else "no nearby reading at or before the query time"
        )
        return Unavailable(UnavailableReason.NO_FRESH_DATA, detail)

    nearest = usable[0]
    if nearest.distance_km <= exact_match_km:
        chosen, method = [nearest], "exact_station"
    else:
        chosen, method = usable[:max_stations], "idw"

    raw_weights = [spatial_weight(c) if method == "idw" else 1.0 for c in chosen]
    total = sum(raw_weights)
    contributions = [
        Contribution(
            station_id=c.reading.station_id,
            distance_km=round(c.distance_km, 3),
            weight=w / total,
            concentration=c.reading.concentration,
            timestamp=c.reading.timestamp,
            quality_flag=c.reading.quality_flag,
        )
        for c, w in zip(chosen, raw_weights, strict=True)
    ]
    value = sum(c.weight * c.concentration for c in contributions)
    return LocalEstimate(
        concentration=round(value, 2),
        method=method,
        method_version=SPATIAL_METHOD_VERSION,
        contributions=contributions,
        oldest_input=min(c.timestamp for c in contributions),
        nearest_distance_km=round(nearest.distance_km, 3),
    )
