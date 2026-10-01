import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import HealthRecord, RiskPrediction


class HealthRecordRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def insert_many(self, rows: list[dict[str, Any]]) -> int:
        """Idempotent: a repeat of (patient, timestamp, source) is ignored. Returns rows added."""
        if not rows:
            return 0
        stmt = (
            insert(HealthRecord)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_health_records_identity")
            .returning(HealthRecord.id)
        )
        return len(self.db.execute(stmt).all())

    def between(
        self,
        patient_id: uuid.UUID,
        start: datetime,
        end: datetime,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[HealthRecord]:
        """Records with start < timestamp ≤ end, for one patient only (BR-11)."""
        order = HealthRecord.timestamp.desc() if newest_first else HealthRecord.timestamp
        q = (
            select(HealthRecord)
            .where(HealthRecord.patient_id == patient_id)
            .where(HealthRecord.timestamp > start, HealthRecord.timestamp <= end)
            .order_by(order)
        )
        if limit:
            q = q.limit(limit)
        return list(self.db.scalars(q))


class RiskPredictionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_many(self, rows: list[dict[str, Any]]) -> None:
        """BR-10: one row per prediction identity. Recomputing the same prediction (e.g. after
        a late diary entry) replaces it instead of adding a duplicate."""
        if not rows:
            return
        stmt = insert(RiskPrediction).values(rows)
        keep = {
            "id",
            "patient_id",
            "prediction_time",
            "horizon_hours",
            "model_version",
            "calibration_version",
            "policy_version",
            "personalization",
            "created_at",
        }
        stmt = stmt.on_conflict_do_update(
            constraint="uq_risk_predictions_identity",
            set_={c: stmt.excluded[c] for c in rows[0] if c not in keep},
        )
        self.db.execute(stmt)

    def history(
        self,
        patient_id: uuid.UUID,
        start: datetime,
        end: datetime,
        horizon_hours: int | None = 0,
        limit: int = 500,
    ) -> list[RiskPrediction]:
        q = (
            select(RiskPrediction)
            .where(RiskPrediction.patient_id == patient_id)
            .where(RiskPrediction.prediction_time > start, RiskPrediction.prediction_time <= end)
            .order_by(RiskPrediction.prediction_time.desc(), RiskPrediction.horizon_hours)
            .limit(limit)
        )
        if horizon_hours is not None:
            q = q.where(RiskPrediction.horizon_hours == horizon_hours)
        return list(self.db.scalars(q))
