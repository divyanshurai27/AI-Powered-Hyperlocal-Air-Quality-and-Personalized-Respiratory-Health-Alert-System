from datetime import datetime

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, DbSession
from app.domain.pollutants import Pollutant
from app.domain.spatial import LocalEstimate, Unavailable
from app.repositories.environment import StationRepository
from app.schemas.air import (
    CurrentAirResponse,
    ForecastHour,
    ForecastResponse,
    PollutantEstimate,
    StationContribution,
    StationOut,
)
from app.schemas.common import ErrorResponse
from app.services.air import AirService
from app.services.forecast import ForecastService

router = APIRouter(prefix="/air", tags=["air quality"], responses={401: {"model": ErrorResponse}})


@router.get(
    "/current",
    response_model=CurrentAirResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Location-matched current air quality",
)
def current_air(
    _: CurrentUser,
    db: DbSession,
    lat: float = Query(..., description="Latitude, WGS84"),
    lon: float = Query(..., description="Longitude, WGS84"),
    at: datetime | None = Query(
        None, description="Evaluate as of this instant (ISO 8601, with timezone). Defaults to now."
    ),
) -> CurrentAirResponse:
    """Each pollutant is either an estimate with its contributing stations, or an explicit
    `available: false` with a reason (`no_fresh_data`, `no_stations_in_radius`, ...).
    Stale data is reported as unavailable, never shown as current (BR-05)."""
    ctx = AirService(db).get_local_air_context(lat, lon, at)
    items = []
    for pollutant, result in ctx.results.items():
        if isinstance(result, LocalEstimate):
            items.append(
                PollutantEstimate(
                    pollutant=pollutant.value,
                    available=True,
                    concentration=result.concentration,
                    unit=ctx.unit,
                    method=result.method,
                    method_version=result.method_version,
                    data_age_minutes=round((ctx.at - result.oldest_input).total_seconds() / 60, 1),
                    nearest_station_km=result.nearest_distance_km,
                    contributions=[
                        StationContribution(
                            station_id=c.station_id,
                            distance_km=c.distance_km,
                            weight=round(c.weight, 4),
                            concentration=c.concentration,
                            observed_at=c.timestamp,
                            quality_flag=c.quality_flag,
                        )
                        for c in result.contributions
                    ],
                )
            )
        else:
            items.append(
                PollutantEstimate(
                    pollutant=pollutant.value,
                    available=False,
                    unit=ctx.unit,
                    unavailable_reason=result.reason.value,
                    unavailable_detail=result.detail,
                )
            )
    return CurrentAirResponse(latitude=lat, longitude=lon, at=ctx.at, pollutants=items)


@router.get("/stations", response_model=list[StationOut], summary="Known monitoring stations")
def list_stations(
    _: CurrentUser,
    db: DbSession,
    reference_only: bool = Query(True, description="Only regulatory reference monitors"),
) -> list[StationOut]:
    return [
        StationOut.model_validate(s, from_attributes=True)
        for s in StationRepository(db).list(reference_only=reference_only)
    ]


@router.get(
    "/forecast",
    response_model=ForecastResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="24-hour hourly forecast for a location",
)
def forecast_air(
    _: CurrentUser,
    db: DbSession,
    lat: float = Query(...),
    lon: float = Query(...),
    pollutant: Pollutant = Pollutant.PM25,
    at: datetime | None = Query(None, description="Issue the forecast as of this instant"),
) -> ForecastResponse:
    """Hourly T+1…T+24 from the frozen, versioned model, or an explicit unavailable state
    (AT-03). A missing model is a 503 `MODEL_NOT_AVAILABLE`, never a fake value."""
    result = ForecastService(db).predict_24h(lat, lon, pollutant, at)
    if isinstance(result, Unavailable):
        return ForecastResponse(
            latitude=lat,
            longitude=lon,
            pollutant=pollutant.value,
            available=False,
            unavailable_reason=result.reason.value,
            unavailable_detail=result.detail,
        )
    return ForecastResponse(
        latitude=lat,
        longitude=lon,
        pollutant=pollutant.value,
        available=True,
        origin=result.origin,
        model_version=result.model_version,
        feature_version=result.feature_version,
        station_weights=result.weights,
        hours=[
            ForecastHour(horizon_hours=h, target_time=t, concentration=v)
            for h, t, v in result.horizons
        ],
    )
