import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin, str_enum, utcnow
from app.domain.enums import DiseaseType, HealthSource, RiskLevel

# Every measurement is optional (BR-06): a record carries whatever the patient or device gave.
MEASUREMENT_COLUMNS = (
    "symptom_score",
    "rescue_inhaler_puffs",
    "exacerbation",
    "heart_rate",
    "spo2",
    "respiratory_rate",
    "steps",
    "sleep_hours",
    "pef_l_min",
)


class HealthRecord(UUIDPrimaryKeyMixin, Base):
    """PRD §6.4. One diary entry or device summary. Sensitive: patient-scoped access only.

    `timestamp` is when the record was made; diary values describe the preceding 24 h.
    """

    __tablename__ = "health_records"
    __table_args__ = (
        UniqueConstraint("patient_id", "timestamp", "source", name="uq_health_records_identity"),
        Index("ix_health_records_patient_id_timestamp", "patient_id", "timestamp"),
        CheckConstraint(f"num_nonnulls({', '.join(MEASUREMENT_COLUMNS)}) > 0", name="not_empty"),
        CheckConstraint("symptom_score BETWEEN 0 AND 10", name="symptom_score_range"),
        CheckConstraint("rescue_inhaler_puffs BETWEEN 0 AND 100", name="rescue_puffs_range"),
        CheckConstraint("heart_rate BETWEEN 25 AND 250", name="heart_rate_range"),
        CheckConstraint("spo2 BETWEEN 50 AND 100", name="spo2_range"),
        CheckConstraint("respiratory_rate BETWEEN 4 AND 80", name="respiratory_rate_range"),
        CheckConstraint("steps BETWEEN 0 AND 100000", name="steps_range"),
        CheckConstraint("sleep_hours BETWEEN 0 AND 24", name="sleep_hours_range"),
        CheckConstraint("pef_l_min BETWEEN 30 AND 1000", name="pef_range"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE")
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symptom_score: Mapped[int | None] = mapped_column(SmallInteger, default=None)  # 0–10
    rescue_inhaler_puffs: Mapped[int | None] = mapped_column(SmallInteger, default=None)
    exacerbation: Mapped[bool | None] = mapped_column(Boolean, default=None)
    heart_rate: Mapped[float | None] = mapped_column(Double, default=None)  # resting, bpm
    spo2: Mapped[float | None] = mapped_column(Double, default=None)  # %
    respiratory_rate: Mapped[float | None] = mapped_column(Double, default=None)  # breaths/min
    steps: Mapped[int | None] = mapped_column(Integer, default=None)
    sleep_hours: Mapped[float | None] = mapped_column(Double, default=None)
    pef_l_min: Mapped[float | None] = mapped_column(Double, default=None)  # peak flow, L/min
    source: Mapped[HealthSource] = mapped_column(
        str_enum(HealthSource, "health_source"), default=HealthSource.MANUAL
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class RiskPrediction(UUIDPrimaryKeyMixin, Base):
    """PRD §6.6. One risk estimate: P(event starting in the target window).

    horizon_hours = 0 is the current estimate made at `prediction_time`; 1..24 are the
    rolling forecast, projected with forecast air quality. Every row names the exact model,
    calibration, policy and feature versions (BR-08) and keeps its input vector (BR-09).
    """

    __tablename__ = "risk_predictions"
    __table_args__ = (
        UniqueConstraint(
            "patient_id",
            "prediction_time",
            "horizon_hours",
            "model_version",
            "calibration_version",
            "policy_version",
            "personalization",
            name="uq_risk_predictions_identity",
        ),
        Index("ix_risk_predictions_patient_id_prediction_time", "patient_id", "prediction_time"),
        CheckConstraint(
            "risk_probability >= 0 AND risk_probability <= 1", name="probability_range"
        ),
        CheckConstraint("raw_probability >= 0 AND raw_probability <= 1", name="raw_range"),
        CheckConstraint("horizon_hours BETWEEN 0 AND 24", name="horizon_range"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE")
    )
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    horizon_hours: Mapped[int] = mapped_column(SmallInteger)
    target_window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    target_window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_probability: Mapped[float] = mapped_column(Double)
    risk_probability: Mapped[float] = mapped_column(Double)
    risk_level: Mapped[RiskLevel] = mapped_column(str_enum(RiskLevel, "risk_level"))
    disease_type: Mapped[DiseaseType] = mapped_column(str_enum(DiseaseType, "disease_type"))
    model_version: Mapped[str] = mapped_column(String(64))
    calibration_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(64))
    feature_version: Mapped[str] = mapped_column(String(64))
    personalization: Mapped[str] = mapped_column(String(64), default="none")
    personal_offset: Mapped[float | None] = mapped_column(Double, default=None)
    confidence: Mapped[str] = mapped_column(String(16), default="normal")  # normal | low
    data_quality: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
