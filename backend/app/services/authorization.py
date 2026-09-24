"""Resource-ownership checks (BR-11, PRD §20: authentication ≠ authorization).

Every patient-scoped service must call `authorize_patient_access` before touching data,
even when the route only exposes `/me`, so a future `/{patient_id}` route can't skip it.
"""

import uuid

from fastapi import status
from sqlalchemy.orm import Session

from app.core.errors import AppError, ErrorCode
from app.models import PatientProfile, User
from app.repositories.audit import AuditRepository
from app.repositories.patients import PatientRepository


def authorize_patient_access(db: Session, user: User, patient_id: uuid.UUID) -> PatientProfile:
    """Return the patient profile if `user` owns it.

    Responds 404 (not 403) for both "doesn't exist" and "belongs to someone else", so an
    attacker can't use the status code to enumerate valid patient IDs (AT-08).
    """
    profile = PatientRepository(db).get(patient_id)
    if profile is None or profile.user_id != user.id:
        AuditRepository(db).write(
            "patient_access_denied",
            user_id=user.id,
            resource_type="patient",
            resource_id=patient_id,
            outcome="failure",
        )
        db.commit()
        raise AppError(ErrorCode.NOT_FOUND, "Patient not found.", status.HTTP_404_NOT_FOUND)
    return profile
