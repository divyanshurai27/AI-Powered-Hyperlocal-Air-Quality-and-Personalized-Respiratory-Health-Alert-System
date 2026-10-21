"""Alert engine (PRD §19, §23, BR-10, BR-11): create once, deduplicate, acknowledge, expire."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.domain.alerts import (
    ALERT_POLICY_VERSION,
    OPEN_STATES,
    AlertDecision,
    ExistingAlert,
    PredictionForAlert,
    alert_message,
    effective_state,
    evaluate_alert,
)
from app.domain.enums import AlertState, RiskLevel
from app.models import Alert, User
from app.repositories.alerts import AlertRepository
from app.repositories.audit import AuditRepository
from app.services.patients import PatientService

logger = get_logger(__name__)

MAX_LIST = 200


@dataclass(frozen=True)
class AlertView:
    alert: Alert
    state: AlertState  # effective state at the reference time


class AlertService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.alerts = AlertRepository(db)
        self.audit = AuditRepository(db)

    def evaluate(
        self,
        patient_id: uuid.UUID,
        prediction: PredictionForAlert,
        versions: dict[str, str],
        now: datetime,
    ) -> tuple[AlertDecision, Alert | None]:
        """create_alert_if_needed. Runs inside the caller's transaction (the one saving the
        prediction), so a failure leaves neither a prediction nor a partial alert behind.
        Does not commit."""
        self.alerts.lock_patient(patient_id)
        existing = self.alerts.relevant(
            patient_id, prediction.prediction_id, prediction.prediction_time
        )
        evaluation = evaluate_alert(
            prediction,
            [
                ExistingAlert(a.id, a.prediction_id, a.state, a.risk_level, a.expires_at)
                for a in existing
            ],
            now,
            timedelta(hours=get_settings().alert_max_prediction_age_hours),
        )
        if evaluation.cancel:
            self.alerts.cancel(evaluation.cancel, now)
        alert = None
        if evaluation.create:
            alert_id = self.alerts.insert_once(
                {
                    "id": uuid.uuid4(),
                    "patient_id": patient_id,
                    "prediction_id": prediction.prediction_id,
                    "state": AlertState.CREATED.value,
                    "risk_level": RiskLevel(prediction.risk_level).value,
                    "risk_probability": prediction.risk_probability,
                    "message": alert_message(prediction.risk_level, prediction.risk_probability),
                    "prediction_time": prediction.prediction_time,
                    "target_window_start": prediction.prediction_time,
                    "target_window_end": prediction.target_window_end,
                    "expires_at": prediction.target_window_end,
                    "alert_policy_version": ALERT_POLICY_VERSION,
                    **versions,
                }
            )
            decision = AlertDecision.CREATED if alert_id else AlertDecision.ALREADY_ALERTED
            if alert_id:
                self.audit.write(
                    "alert_created",
                    resource_type="alert",
                    resource_id=alert_id,
                    details={"level": RiskLevel(prediction.risk_level).value},
                )
        else:
            decision = evaluation.decision
        alert = self.alerts.for_prediction(prediction.prediction_id)
        logger.info(
            "alert_evaluated",
            extra={
                "patient_id": str(patient_id),
                "decision": decision.value,
                "cancelled": len(evaluation.cancel),
            },
        )
        return decision, alert

    def list(
        self,
        user: User,
        at: datetime | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[AlertView]:
        """get_active_alerts (active_only) or the patient's alert history, newest first."""
        profile = PatientService(self.db).require_consent(user)
        if not 1 <= limit <= MAX_LIST:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                f"limit must be between 1 and {MAX_LIST}.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        ref = (at or datetime.now(UTC)).astimezone(UTC)
        views = [
            AlertView(a, effective_state(a.state, a.expires_at, ref))
            for a in self.alerts.list(profile.id, MAX_LIST if active_only else limit)
            # As of `ref`, alerts raised later do not exist yet.
            if a.prediction_time <= ref
        ]
        if active_only:
            views = [v for v in views if v.state in OPEN_STATES]
        return views[:limit]

    def acknowledge(self, user: User, alert_id: uuid.UUID) -> AlertView:
        """Idempotent. 404 for another patient's alert (no enumeration, AT-08)."""
        profile = PatientService(self.db).require_consent(user)
        alert = self.alerts.get_for_patient(alert_id, profile.id)
        if alert is None:
            self.audit.write(
                "alert_access_denied",
                user_id=user.id,
                resource_type="alert",
                resource_id=alert_id,
                outcome="failure",
            )
            self.db.commit()
            raise AppError(ErrorCode.NOT_FOUND, "Alert not found.", status.HTTP_404_NOT_FOUND)
        now = datetime.now(UTC)
        if alert.state == AlertState.CANCELLED:
            raise AppError(
                ErrorCode.ALERT_NOT_OPEN,
                "This alert was withdrawn and cannot be acknowledged.",
                status.HTTP_409_CONFLICT,
            )
        if alert.state != AlertState.ACKNOWLEDGED:
            # An expired alert can still be acknowledged: the patient saw it late.
            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = now
            self.audit.write(
                "alert_acknowledged",
                user_id=user.id,
                resource_type="alert",
                resource_id=alert.id,
            )
            self.db.commit()
        return AlertView(alert, AlertState.ACKNOWLEDGED)
