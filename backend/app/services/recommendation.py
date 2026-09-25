"""'Today for you': rule-based guidance from the 24 h forecast and the patient's profile."""

from dataclasses import dataclass
from datetime import datetime

from fastapi import status
from sqlalchemy.orm import Session

from app.core.errors import AppError, ErrorCode
from app.domain.guidance import (
    CATEGORY_HEALTH,
    CATEGORY_NAMES,
    LEVEL_TEXT,
    DayGuidance,
    day_guidance,
)
from app.domain.pollutants import Pollutant
from app.domain.spatial import Unavailable
from app.models import User
from app.services.forecast import ForecastService
from app.services.patients import PatientService

# PM2.5 is required; PM10 and O3 refine the guidance when their models are available.
# NO2/SO2 are left out until their source units are verified (docs/data_and_methods.md).
GUIDANCE_POLLUTANTS = (Pollutant.PM25, Pollutant.PM10, Pollutant.O3)


@dataclass
class TodayResult:
    place_label: str
    latitude: float
    longitude: float
    guidance: DayGuidance | None
    origin: datetime | None
    model_versions: dict[str, str]
    skipped: dict[str, str]
    disease_type: str | None
    severity: str | None
    unavailable_reason: str | None = None


class RecommendationService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def today(
        self,
        user: User,
        *,
        at: datetime | None = None,
        place: str = "home",
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> TodayResult:
        profile = PatientService(self.db).get_own_profile(user)
        if latitude is not None and longitude is not None:
            label = "Selected location"
        else:
            saved = profile.saved_place(place)
            if saved is None:
                raise AppError(
                    ErrorCode.LOCATION_NOT_SET,
                    f"No {place} location saved. Set it in your profile or pass lat/lon.",
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                )
            label, latitude, longitude = saved["label"], saved["latitude"], saved["longitude"]

        forecasts: dict[Pollutant, list[tuple[datetime, float]]] = {}
        versions: dict[str, str] = {}
        skipped: dict[str, str] = {}
        origin = None
        service = ForecastService(self.db)
        for pollutant in GUIDANCE_POLLUTANTS:
            try:
                result = service.predict_24h(latitude, longitude, pollutant, at)
            except AppError as exc:
                if exc.code == ErrorCode.INVALID_LOCATION:
                    raise
                skipped[pollutant.value] = exc.code.value
                continue
            if isinstance(result, Unavailable):
                skipped[pollutant.value] = result.reason.value
                continue
            forecasts[pollutant] = [(t, v) for _, t, v in result.horizons]
            versions[pollutant.value] = result.model_version
            origin = origin or result.origin

        common = dict(
            place_label=label,
            latitude=latitude,
            longitude=longitude,
            origin=origin,
            model_versions=versions,
            skipped=skipped,
            disease_type=profile.disease_type,
            severity=profile.severity,
        )
        if Pollutant.PM25 not in forecasts:
            return TodayResult(
                guidance=None, unavailable_reason=skipped.get("pm25", "unavailable"), **common
            )
        return TodayResult(
            guidance=day_guidance(forecasts, profile.disease_type, profile.severity), **common
        )


def describe(result: TodayResult) -> dict:
    """Plain-language fields for the API response."""
    g = result.guidance
    if g is None:
        return {}
    worst = max(
        (h for h in g.hours if h.waking), key=lambda h: (h.level, h.category), default=g.hours[0]
    )
    condition = {"asthma": "asthma", "copd": "COPD"}.get(
        result.disease_type or "", "your condition"
    )
    severity = (result.severity or "").replace("_", " ")
    tier_note = {
        "general": "No respiratory condition on your profile, so general NAQI guidance applies.",
        "sensitive": f"Stricter thresholds applied for {severity} {condition}.".replace("  ", " "),
        "high": f"Strictest thresholds applied for {severity} {condition}.",
    }[g.tier]
    return {
        "headline": LEVEL_TEXT[g.headline_level],
        "worst_category": CATEGORY_NAMES[worst.category],
        "worst_pollutant": worst.dominant.value,
        "health_note": CATEGORY_HEALTH[worst.category],
        "personalisation_note": tier_note,
    }
