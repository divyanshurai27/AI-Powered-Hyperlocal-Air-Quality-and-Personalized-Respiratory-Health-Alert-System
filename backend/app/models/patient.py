import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, SmallInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, str_enum
from app.domain.enums import ConsentStatus, DiseaseType, Severity, Sex

if TYPE_CHECKING:
    from app.models.user import User


class PatientProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """PRD §6.3. `id` is the patient_id used by every patient-scoped table."""

    __tablename__ = "patient_profiles"
    __table_args__ = (
        CheckConstraint("age IS NULL OR (age >= 0 AND age <= 120)", name="age_range"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    # Clinical fields are nullable: an incomplete profile is explicit, never defaulted (BR-06).
    age: Mapped[int | None] = mapped_column(SmallInteger, default=None)
    sex: Mapped[Sex | None] = mapped_column(str_enum(Sex, "sex"), default=None)
    disease_type: Mapped[DiseaseType | None] = mapped_column(
        str_enum(DiseaseType, "disease_type"), default=None
    )
    severity: Mapped[Severity | None] = mapped_column(str_enum(Severity, "severity"), default=None)
    consent_status: Mapped[ConsentStatus] = mapped_column(
        str_enum(ConsentStatus, "consent_status"), default=ConsentStatus.PENDING
    )
    consent_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    baseline_information: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    user: Mapped["User"] = relationship(back_populates="patient_profile")

    @property
    def profile_complete(self) -> bool:
        """True once the minimum fields the risk model needs are present (PRD §14)."""
        return None not in (self.age, self.sex, self.disease_type, self.severity)
