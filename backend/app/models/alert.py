import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Double, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin, str_enum, utcnow
from app.domain.enums import AlertState, RiskLevel


class Alert(UUIDPrimaryKeyMixin, Base):
    """PRD §19. One alert raised from one risk prediction (unique: BR-10).

    Traceable to patient, prediction, model, policy and time (PRD §19 "alert debugging").
    The level and probability are a snapshot of what the patient was told.
    """

    __tablename__ = "alerts"
    __table_args__ = (
        Index("ix_alerts_patient_id_created_at", "patient_id", "created_at"),
        CheckConstraint(
            "risk_probability >= 0 AND risk_probability <= 1", name="probability_range"
        ),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE")
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("risk_predictions.id", ondelete="CASCADE"), unique=True
    )
    state: Mapped[AlertState] = mapped_column(
        str_enum(AlertState, "alert_state"), default=AlertState.CREATED
    )
    risk_level: Mapped[RiskLevel] = mapped_column(str_enum(RiskLevel, "risk_level"))
    risk_probability: Mapped[float] = mapped_column(Double)
    message: Mapped[str] = mapped_column(String(256))
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    target_window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    target_window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[str] = mapped_column(String(64))
    calibration_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(64))
    alert_policy_version: Mapped[str] = mapped_column(String(64))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    cancel_reason: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
