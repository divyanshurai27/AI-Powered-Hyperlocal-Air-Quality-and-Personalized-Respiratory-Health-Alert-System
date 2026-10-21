import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.schemas.common import ErrorResponse
from app.schemas.risk import (
    AlertSummaryOut,
    ExplanationOut,
    PersonalizationOut,
    RiskCurrentOut,
    RiskEstimateOut,
    RiskExplanationOut,
    RiskForecastOut,
    RiskHistoryItem,
    RiskHistoryOut,
    RiskHorizonOut,
)
from app.services.risk import RISK_TARGET, RiskEstimate, RiskResult, RiskService

router = APIRouter(
    prefix="/risk",
    tags=["risk"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)

AtQuery = Query(None, description="Evaluate as of this instant; defaults to now")
PersonalizeQuery = Query(True, description="Apply the bounded personal-history offset")


def _estimate(e: RiskEstimate) -> RiskEstimateOut:
    return RiskEstimateOut(
        prediction_id=e.prediction_id,
        horizon_hours=e.horizon_hours,
        target_window_start=e.target_window_start,
        target_window_end=e.target_window_end,
        risk_probability=round(e.risk_probability, 5),
        risk_level=e.risk_level.value,
        raw_probability=round(e.raw_probability, 5),
        confidence=e.confidence,
        data_quality=e.data_quality,
    )


def _header(r: RiskResult) -> dict:
    b = r.baseline
    return {
        "target": RISK_TARGET,
        "prediction_time": r.origin,
        "disease_type": r.disease_type,
        "severity": r.severity,
        "model_version": r.model_version,
        "calibration_version": r.calibration_version,
        "feature_version": r.feature_version,
        "policy_version": r.policy_version,
        "thresholds": r.thresholds,
        "personalization": PersonalizationOut(
            version=b.version,
            offset=b.offset,
            observed_events=b.observed_events,
            expected_events=b.expected_events,
            observed_days=round(b.observed_days, 2),
        )
        if b
        else None,
        "location_source": r.location_source,
        "current": _estimate(r.current),
        "explanation": ExplanationOut(**r.current.explanation) if r.current.explanation else None,
        "alert_decision": r.alert_decision.value,
        "alert": AlertSummaryOut(
            id=r.alert.id,
            state=r.alert.state.value,
            risk_level=r.alert.risk_level.value,
            message=r.alert.message,
        )
        if r.alert
        else None,
    }


@router.get("/current", response_model=RiskCurrentOut, summary="Current 24 h respiratory risk")
def risk_current(
    user: CurrentUser,
    db: DbSession,
    at: datetime | None = AtQuery,
    personalize: bool = PersonalizeQuery,
) -> RiskCurrentOut:
    """Probability of a respiratory event starting in the next 24 h, from your recent estimated
    exposure, symptom diary and profile. Requires consent, disease type and severity."""
    r = RiskService(db).assess(user, at, personalize=personalize)
    return RiskCurrentOut(**_header(r))


@router.get(
    "/forecast", response_model=RiskForecastOut, summary="Rolling 24 h respiratory risk forecast"
)
def risk_forecast(
    user: CurrentUser,
    db: DbSession,
    at: datetime | None = AtQuery,
    personalize: bool = PersonalizeQuery,
) -> RiskForecastOut:
    """The current estimate plus the 24 h risk projected at each of the next 24 hours, using
    the air-quality forecast. Hours the forecast does not cover are marked unavailable."""
    r = RiskService(db).assess(user, at, forecast=True, personalize=personalize)
    return RiskForecastOut(
        **_header(r),
        horizons=[
            RiskHorizonOut(
                horizon_hours=h.horizon_hours,
                time=h.time,
                available=h.estimate is not None,
                unavailable_reason=h.unavailable_reason,
                estimate=_estimate(h.estimate) if h.estimate else None,
            )
            for h in r.horizons
        ],
        forecast_model_versions=r.forecast_versions,
        pollutants_skipped=r.pollutants_skipped,
    )


@router.get("/history", response_model=RiskHistoryOut, summary="Stored risk estimates")
def risk_history(
    user: CurrentUser,
    db: DbSession,
    start: datetime | None = None,
    end: datetime | None = None,
    horizon_hours: int | None = Query(0, ge=0, le=24, description="Omit for all horizons"),
    limit: int = Query(200, ge=1, le=1000),
) -> RiskHistoryOut:
    rows = RiskService(db).history(user, start, end, horizon_hours, limit)
    return RiskHistoryOut(predictions=[RiskHistoryItem.model_validate(r) for r in rows])


@router.get(
    "/predictions/{prediction_id}/explanation",
    response_model=RiskExplanationOut,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Why a stored estimate came out as it did",
)
def risk_explanation(
    prediction_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> RiskExplanationOut:
    """The SHAP explanation stored with the prediction, verified against that prediction's
    model, calibration and input vector (BR-09). Another patient's id is a 404."""
    row = RiskService(db).explanation(user, prediction_id)
    return RiskExplanationOut(
        prediction_id=row.id,
        prediction_time=row.prediction_time,
        horizon_hours=row.horizon_hours,
        risk_probability=row.risk_probability,
        risk_level=row.risk_level.value,
        explanation=ExplanationOut(**row.explanation),
    )


@router.get("/debug", include_in_schema=False)
def risk_debug(user: CurrentUser, db: DbSession, at: datetime | None = AtQuery) -> dict[str, Any]:
    """PRD §33 prediction trace: development environment only."""
    if not get_settings().debug_enabled:
        raise AppError(ErrorCode.NOT_FOUND, "Not found.", status.HTTP_404_NOT_FOUND)
    return RiskService(db).debug_trace(user, at)
