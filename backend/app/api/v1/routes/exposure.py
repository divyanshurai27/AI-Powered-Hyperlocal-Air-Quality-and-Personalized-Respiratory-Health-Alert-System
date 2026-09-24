from datetime import datetime

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentUser, DbSession
from app.domain.exposure import ExposureProfile
from app.domain.pollutants import Pollutant
from app.schemas.common import ErrorResponse
from app.schemas.exposure import (
    ExposureHourOut,
    ExposureOut,
    ExposureWindowOut,
    LocationBatchIn,
    LocationBatchOut,
)
from app.services.exposure import ExposureService, LocationFix

router = APIRouter(
    tags=["location & exposure"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)


@router.post("/location", response_model=LocationBatchOut, status_code=status.HTTP_201_CREATED)
def record_location(body: LocationBatchIn, user: CurrentUser, db: DbSession) -> LocationBatchOut:
    """Upload location fixes for the signed-in patient. Requires granted consent."""
    stored = ExposureService(db).record_locations(
        user,
        [
            LocationFix(f.timestamp, f.latitude, f.longitude, f.accuracy_m, f.activity_context)
            for f in body.fixes
        ],
    )
    return LocationBatchOut(received=len(body.fixes), stored=stored)


def _to_out(profile: ExposureProfile, include_hours: bool) -> ExposureOut:
    hours = [
        ExposureHourOut(
            hour=h.hour,
            exposure=h.exposure,
            ambient=h.ambient,
            factor=h.factor,
            microenvironment=h.microenvironment.value if h.microenvironment else None,
            assumed_outdoor=h.assumed_outdoor,
            missing_reason=h.missing_reason,
        )
        for h in profile.hours
    ]
    return ExposureOut(
        pollutant=profile.pollutant.value,
        model_version=profile.model_version,
        current=hours[-1] if hours else None,
        windows=[ExposureWindowOut(**vars(w)) for w in profile.windows.values()],
        lagged=profile.lagged,
        hours=hours if include_hours else None,
    )


@router.get("/exposure/current", response_model=ExposureOut, summary="Current estimated exposure")
def exposure_current(
    user: CurrentUser,
    db: DbSession,
    pollutant: Pollutant = Pollutant.PM25,
    at: datetime | None = Query(None, description="Evaluate as of this instant; defaults to now"),
) -> ExposureOut:
    profile = ExposureService(db).exposure_profile(user, pollutant, at, hours=72)
    return _to_out(profile, include_hours=False)


@router.get("/exposure/history", response_model=ExposureOut, summary="Hourly exposure history")
def exposure_history(
    user: CurrentUser,
    db: DbSession,
    pollutant: Pollutant = Pollutant.PM25,
    hours: int = Query(24, ge=1, le=168),
    at: datetime | None = None,
) -> ExposureOut:
    return _to_out(
        ExposureService(db).exposure_profile(user, pollutant, at, hours=hours), include_hours=True
    )
