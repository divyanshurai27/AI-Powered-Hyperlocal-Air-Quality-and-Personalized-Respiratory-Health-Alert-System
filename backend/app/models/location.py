import uuid
from datetime import datetime

from sqlalchemy import DateTime, Double, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin, str_enum, utcnow
from app.domain.exposure import MicroEnvironment


class LocationRecord(UUIDPrimaryKeyMixin, Base):
    """PRD §6.5. One location fix for one patient. Sensitive: patient-scoped access only."""

    __tablename__ = "locations"
    __table_args__ = (
        UniqueConstraint("patient_id", "timestamp", name="uq_locations_patient_timestamp"),
        Index("ix_locations_patient_id_timestamp", "patient_id", "timestamp"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE")
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    latitude: Mapped[float] = mapped_column(Double)
    longitude: Mapped[float] = mapped_column(Double)
    accuracy_m: Mapped[float | None] = mapped_column(Double, default=None)
    activity_context: Mapped[MicroEnvironment] = mapped_column(
        str_enum(MicroEnvironment, "micro_environment"), default=MicroEnvironment.UNKNOWN
    )
    source: Mapped[str] = mapped_column(String(16), default="device")  # device | manual
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
