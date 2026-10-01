"""Health records intake and the health context for risk (PRD §14, BR-06, BR-11)."""

import uuid
from datetime import UTC, datetime, timedelta

import pandas as pd
from fastapi import status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.models import HealthRecord, User
from app.models.health import MEASUREMENT_COLUMNS
from app.repositories.audit import AuditRepository
from app.repositories.health import HealthRecordRepository
from app.schemas.health import HealthRecordIn
from app.services.patients import PatientService

MAX_LIST = 500


def records_frame(records: list[HealthRecord]) -> pd.DataFrame:
    """Health records as the frame app/ml/risk_features.health_features expects."""
    cols = ["timestamp", *MEASUREMENT_COLUMNS]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([{c: getattr(r, c) for c in cols} for r in records])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["exacerbation"] = df["exacerbation"].map({True: 1.0, False: 0.0})
    return df


class HealthService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.records = HealthRecordRepository(db)
        self.audit = AuditRepository(db)

    def record(self, user: User, records: list[HealthRecordIn]) -> int:
        profile = PatientService(self.db).require_consent(user)
        now = datetime.now(UTC)
        oldest = now - timedelta(days=get_settings().health_max_backdate_days)
        rows = []
        for r in records:
            ts = r.timestamp.astimezone(UTC)
            if ts > now + timedelta(minutes=5) or ts < oldest:
                raise AppError(
                    ErrorCode.VALIDATION_ERROR,
                    "Health record timestamps must be in the past year and not in the future.",
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                )
            rows.append(
                {"id": uuid.uuid4(), "patient_id": profile.id, **r.model_dump(), "timestamp": ts}
            )
        stored = self.records.insert_many(rows)
        # Counts only: health values never go into the audit log.
        self.audit.write(
            "health_records_recorded",
            user_id=user.id,
            resource_type="patient",
            resource_id=profile.id,
            details={"received": len(rows), "stored": stored},
        )
        self.db.commit()
        return stored

    def list(
        self,
        user: User,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[HealthRecord]:
        profile = PatientService(self.db).require_consent(user)
        if not 1 <= limit <= MAX_LIST:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                f"limit must be between 1 and {MAX_LIST}.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        end = (end or datetime.now(UTC)).astimezone(UTC)
        start = (start or end - timedelta(days=30)).astimezone(UTC)
        return self.records.between(profile.id, start, end, limit=limit, newest_first=True)

    def history_frame(self, patient_id: uuid.UUID, at: datetime, days: int = 366) -> pd.DataFrame:
        """Records dated ≤ `at` only (BR-04), for risk features and the personal baseline."""
        return records_frame(self.records.between(patient_id, at - timedelta(days=days), at))
