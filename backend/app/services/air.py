from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.domain.pollutants import CANONICAL_UNIT, Pollutant
from app.domain.spatial import (
    GeoPoint,
    LocalEstimate,
    Unavailable,
    estimate_local_air_quality,
    validate_coordinates,
)
from app.repositories.environment import ObservationRepository


@dataclass(frozen=True)
class LocalAirContext:
    location: GeoPoint
    at: datetime
    unit: str
    results: dict[Pollutant, LocalEstimate | Unavailable]

    @property
    def any_available(self) -> bool:
        return any(isinstance(r, LocalEstimate) for r in self.results.values())


class AirService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.observations = ObservationRepository(db)

    def get_local_air_context(
        self,
        latitude: float,
        longitude: float,
        at: datetime | None = None,
        pollutants: tuple[Pollutant, ...] = tuple(Pollutant),
    ) -> LocalAirContext:
        settings = get_settings()
        if not validate_coordinates(latitude, longitude):
            raise AppError(
                ErrorCode.INVALID_LOCATION,
                "Latitude must be in [-90, 90] and longitude in [-180, 180].",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        now = datetime.now(UTC)
        query_at = (at or now).astimezone(UTC)
        if query_at > now + timedelta(minutes=1):
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "`at` cannot be in the future; use the forecast endpoint for future hours.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

        point = GeoPoint(latitude, longitude)
        max_age = timedelta(hours=settings.aq_max_age_hours)
        results = {}
        for pollutant in pollutants:
            readings = self.observations.latest_readings(pollutant, query_at)
            results[pollutant] = estimate_local_air_quality(
                point,
                readings,
                query_at,
                radius_km=settings.spatial_radius_km,
                max_age=max_age,
                exact_match_km=settings.spatial_exact_match_km,
                max_stations=settings.spatial_max_stations,
            )
        return LocalAirContext(point, query_at, CANONICAL_UNIT, results)
