from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, DbSession
from app.domain.guidance import CATEGORY_NAMES
from app.schemas.common import ErrorResponse
from app.schemas.recommendation import GuidanceHour, TimeWindow, TodayResponse
from app.services.recommendation import RecommendationService, describe

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)


@router.get("/today", response_model=TodayResponse, summary="Today's personal air-quality guidance")
def today(
    user: CurrentUser,
    db: DbSession,
    place: Literal["home", "work"] = "home",
    lat: float | None = Query(None, description="Override the saved place"),
    lon: float | None = Query(None),
    at: datetime | None = Query(None, description="Evaluate as of this instant; defaults to now"),
) -> TodayResponse:
    """Best and worst hours of the next 24 h at the patient's saved home or work location,
    from the pollutant forecast and India's NAQI categories, with stricter thresholds for
    asthma/COPD. Rule-based (see domain/guidance.py); the personal risk model is Phase 4."""
    r = RecommendationService(db).today(user, at=at, place=place, latitude=lat, longitude=lon)
    base = TodayResponse(
        available=r.guidance is not None,
        place_label=r.place_label,
        latitude=r.latitude,
        longitude=r.longitude,
        disease_type=r.disease_type,
        severity=r.severity,
        forecast_origin=r.origin,
        model_versions=r.model_versions,
        pollutants_skipped=r.skipped,
        unavailable_reason=r.unavailable_reason,
    )
    g = r.guidance
    if g is None:
        return base
    return base.model_copy(
        update={
            **describe(r),
            "policy_version": g.policy_version,
            "level": g.headline_level.name.lower(),
            "best_window": TimeWindow(start=g.best_window[0], end=g.best_window[1])
            if g.best_window
            else None,
            "worst_window": TimeWindow(start=g.worst_window[0], end=g.worst_window[1])
            if g.worst_window
            else None,
            "avoid_windows": [TimeWindow(start=a, end=b) for a, b in g.avoid_windows],
            "hours": [
                GuidanceHour(
                    time=h.time,
                    level=h.level.name.lower(),
                    category=CATEGORY_NAMES[h.category],
                    dominant_pollutant=h.dominant.value,
                    values={p.value: round(v, 1) for p, v in h.values.items()},
                    waking=h.waking,
                )
                for h in g.hours
            ],
        }
    )
