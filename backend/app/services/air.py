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

MAX_HISTORY_HOURS = 168


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

    @staticmethod
    def _validated_at(latitude: float, longitude: float, at: datetime | None) -> datetime:
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
        return query_at

    def get_local_air_context(
        self,
        latitude: float,
        longitude: float,
        at: datetime | None = None,
        pollutants: tuple[Pollutant, ...] = tuple(Pollutant),
    ) -> LocalAirContext:
        settings = get_settings()
        query_at = self._validated_at(latitude, longitude, at)
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

    def history(
        self,
        latitude: float,
        longitude: float,
        pollutant: Pollutant,
        at: datetime | None = None,
        hours: int = 24,
    ) -> list[tuple[datetime, LocalEstimate | Unavailable]]:
        """Hourly location-matched estimates for the `hours` AQ hours up to `at`, each with
        the same freshness and spatial rules as /air/current. Gaps stay gaps (BR-05)."""
        from app.services.exposure import hour_grid, latest_fresh_by_hour

        settings = get_settings()
        if not 1 <= hours <= MAX_HISTORY_HOURS:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                f"hours must be between 1 and {MAX_HISTORY_HOURS}.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        ctx_at = self._validated_at(latitude, longitude, at)
        grid = hour_grid(ctx_at, hours)
        max_age = timedelta(hours=settings.aq_max_age_hours)
        readings = self.observations.readings_between(pollutant, grid[0] - max_age, grid[-1])
        fresh = latest_fresh_by_hour(readings, grid, max_age)
        point = GeoPoint(latitude, longitude)
        return [
            (
                hour,
                estimate_local_air_quality(
                    point,
                    fresh[hour],
                    hour,
                    radius_km=settings.spatial_radius_km,
                    max_age=max_age,
                    exact_match_km=settings.spatial_exact_match_km,
                    max_stations=settings.spatial_max_stations,
                ),
            )
            for hour in grid
        ]
