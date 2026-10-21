import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.domain.alerts import BLOCKING_STATES
from app.domain.enums import AlertState
from app.models import Alert, PatientProfile


class AlertRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def lock_patient(self, patient_id: uuid.UUID) -> None:
        """Serialise alert evaluation per patient, so two concurrent requests cannot both
        decide that no equivalent alert exists (PRD §23 transaction boundary)."""
        self.db.execute(
            select(PatientProfile.id).where(PatientProfile.id == patient_id).with_for_update()
        )

    def relevant(
        self, patient_id: uuid.UUID, prediction_id: uuid.UUID, at: datetime
    ) -> list[Alert]:
        """Alerts that can affect a decision at `at`: this prediction's own alert, and any
        not-cancelled alert whose window is still running."""
        q = select(Alert).where(
            Alert.patient_id == patient_id,
            or_(
                Alert.prediction_id == prediction_id,
                Alert.state.in_([s.value for s in BLOCKING_STATES]) & (Alert.expires_at > at),
            ),
        )
        return list(self.db.scalars(q))

    def insert_once(self, row: dict[str, Any]) -> uuid.UUID | None:
        """BR-10: one alert per prediction. Returns the new id, or None if it already existed."""
        stmt = (
            insert(Alert)
            .values(row)
            .on_conflict_do_nothing(index_elements=["prediction_id"])
            .returning(Alert.id)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def cancel(self, ids: dict[uuid.UUID, str], at: datetime) -> None:
        for alert_id, reason in ids.items():
            self.db.execute(
                update(Alert)
                .where(Alert.id == alert_id)
                .values(state=AlertState.CANCELLED, cancelled_at=at, cancel_reason=reason)
            )

    def list(self, patient_id: uuid.UUID, limit: int = 50) -> list[Alert]:
        q = (
            select(Alert)
            .where(Alert.patient_id == patient_id)
            .order_by(Alert.created_at.desc(), Alert.prediction_time.desc())
            .limit(limit)
        )
        return list(self.db.scalars(q))

    def get_for_patient(self, alert_id: uuid.UUID, patient_id: uuid.UUID) -> Alert | None:
        """None both when the alert doesn't exist and when it belongs to someone else."""
        return self.db.scalar(
            select(Alert).where(Alert.id == alert_id, Alert.patient_id == patient_id)
        )

    def for_prediction(self, prediction_id: uuid.UUID) -> Alert | None:
        return self.db.scalar(
            select(Alert)
            .where(Alert.prediction_id == prediction_id)
            .execution_options(populate_existing=True)
        )
