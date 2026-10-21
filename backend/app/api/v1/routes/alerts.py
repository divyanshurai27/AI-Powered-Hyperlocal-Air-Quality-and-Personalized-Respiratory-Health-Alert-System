import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, DbSession
from app.domain.alerts import format_percent
from app.schemas.alert import AlertListOut, AlertOut
from app.schemas.common import ErrorResponse
from app.services.alerts import AlertService, AlertView

router = APIRouter(
    prefix="/alerts",
    tags=["alerts"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)


def _out(v: AlertView) -> AlertOut:
    a = v.alert
    return AlertOut(
        id=a.id,
        prediction_id=a.prediction_id,
        state=v.state.value,
        risk_level=a.risk_level.value,
        risk_probability=round(a.risk_probability, 5),
        risk_percent=format_percent(a.risk_probability),
        message=a.message,
        prediction_time=a.prediction_time,
        target_window_start=a.target_window_start,
        target_window_end=a.target_window_end,
        expires_at=a.expires_at,
        created_at=a.created_at,
        acknowledged_at=a.acknowledged_at,
        cancel_reason=a.cancel_reason,
        model_version=a.model_version,
        calibration_version=a.calibration_version,
        policy_version=a.policy_version,
        alert_policy_version=a.alert_policy_version,
    )


@router.get("", response_model=AlertListOut, summary="Your alerts, newest first")
def list_alerts(
    user: CurrentUser,
    db: DbSession,
    active: bool = Query(False, description="Only alerts still awaiting acknowledgement"),
    at: datetime | None = Query(
        None, description="Evaluate states as of this instant (expiry); defaults to now"
    ),
    limit: int = Query(50, ge=1, le=200),
) -> AlertListOut:
    """Alerts are raised by risk assessments (GET /risk/current or /risk/forecast) when the
    current estimate reaches High or above. Requires consent."""
    ref = (at or datetime.now(UTC)).astimezone(UTC)
    views = AlertService(db).list(user, ref, active_only=active, limit=limit)
    return AlertListOut(alerts=[_out(v) for v in views], reference_time=ref)


@router.post(
    "/{alert_id}/acknowledge",
    response_model=AlertOut,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Mark an alert as seen",
)
def acknowledge_alert(alert_id: uuid.UUID, user: CurrentUser, db: DbSession) -> AlertOut:
    """Idempotent. A withdrawn (cancelled) alert returns 409 `ALERT_NOT_OPEN`."""
    return _out(AlertService(db).acknowledge(user, alert_id))
