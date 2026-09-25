import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, Double, ForeignKey, SmallInteger, String
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
        # A saved place is either fully set (both coordinates) or absent.
        CheckConstraint("(home_latitude IS NULL) = (home_longitude IS NULL)", name="home_pair"),
        CheckConstraint("(work_latitude IS NULL) = (work_longitude IS NULL)", name="work_pair"),
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

    # Saved places so each user's dashboard and guidance open on *their* area. Sensitive:
    # patient-scoped access only, and never written to the audit log.
    home_label: Mapped[str | None] = mapped_column(String(80), default=None)
    home_latitude: Mapped[float | None] = mapped_column(Double, default=None)
    home_longitude: Mapped[float | None] = mapped_column(Double, default=None)
    work_label: Mapped[str | None] = mapped_column(String(80), default=None)
    work_latitude: Mapped[float | None] = mapped_column(Double, default=None)
    work_longitude: Mapped[float | None] = mapped_column(Double, default=None)

    user: Mapped["User"] = relationship(back_populates="patient_profile")

    @property
    def profile_complete(self) -> bool:
        """True once the minimum fields the risk model needs are present (PRD §14)."""
        return None not in (self.age, self.sex, self.disease_type, self.severity)

    def saved_place(self, kind: str) -> dict[str, Any] | None:
        lat, lon = getattr(self, f"{kind}_latitude"), getattr(self, f"{kind}_longitude")
        if lat is None or lon is None:
            return None
        return {
            "label": getattr(self, f"{kind}_label") or kind.title(),
            "latitude": lat,
            "longitude": lon,
        }

    @property
    def home(self) -> dict[str, Any] | None:
        return self.saved_place("home")

    @property
    def work(self) -> dict[str, Any] | None:
        return self.saved_place("work")
