import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import request_id_var
from app.models import AuditLog


class AuditRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def write(
        self,
        action: str,
        *,
        user_id: uuid.UUID | None = None,
        resource_type: str | None = None,
        resource_id: uuid.UUID | str | None = None,
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            outcome=outcome,
            request_id=request_id_var.get(),
            details=details or {},
        )
        self.db.add(entry)
        return entry
