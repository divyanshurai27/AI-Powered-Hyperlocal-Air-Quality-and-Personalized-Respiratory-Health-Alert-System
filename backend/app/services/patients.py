from datetime import UTC, datetime

from fastapi import status
from sqlalchemy.orm import Session

from app.core.errors import AppError, ErrorCode
from app.models import PatientProfile, User
from app.repositories.audit import AuditRepository
from app.repositories.patients import PatientRepository
from app.schemas.patient import PatientProfileUpdate
from app.services.authorization import authorize_patient_access

# Clearing these would leave the record in an undefined state, so null is rejected.
_NON_NULLABLE_FIELDS = {"consent_status", "baseline_information"}


class PatientService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.patients = PatientRepository(db)
        self.audit = AuditRepository(db)

    def get_own_profile(self, user: User) -> PatientProfile:
        profile = self.patients.get_by_user_id(user.id)
        if profile is None:
            raise AppError(
                ErrorCode.NOT_FOUND, "Patient profile not found.", status.HTTP_404_NOT_FOUND
            )
        return authorize_patient_access(self.db, user, profile.id)

    def update_own_profile(self, user: User, update: PatientProfileUpdate) -> PatientProfile:
        profile = self.get_own_profile(user)
        changes = update.model_dump(exclude_unset=True)

        null_fields = sorted(k for k in _NON_NULLABLE_FIELDS if k in changes and changes[k] is None)
        if null_fields:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                f"These fields cannot be null: {', '.join(null_fields)}.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

        now = datetime.now(UTC)
        if "consent_status" in changes and changes["consent_status"] != profile.consent_status:
            profile.consent_updated_at = now
            self.audit.write(
                "consent_changed",
                user_id=user.id,
                resource_type="patient",
                resource_id=profile.id,
                details={"from": profile.consent_status, "to": changes["consent_status"]},
            )

        for field, value in changes.items():
            if field in ("home", "work"):
                place = value or {}
                setattr(profile, f"{field}_label", place.get("label"))
                setattr(profile, f"{field}_latitude", place.get("latitude"))
                setattr(profile, f"{field}_longitude", place.get("longitude"))
            else:
                setattr(profile, field, value)

        # Field names only: audit logs must not hold health data values.
        self.audit.write(
            "patient_profile_updated",
            user_id=user.id,
            resource_type="patient",
            resource_id=profile.id,
            details={"fields": sorted(changes)},
        )
        self.db.commit()
        self.db.refresh(profile)
        return profile
