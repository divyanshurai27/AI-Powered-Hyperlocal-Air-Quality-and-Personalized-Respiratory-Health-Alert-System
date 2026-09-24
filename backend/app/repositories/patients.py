import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import PatientProfile


class PatientRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, patient_id: uuid.UUID) -> PatientProfile | None:
        return self.db.get(PatientProfile, patient_id)

    def get_by_user_id(self, user_id: uuid.UUID) -> PatientProfile | None:
        return self.db.scalar(select(PatientProfile).where(PatientProfile.user_id == user_id))
