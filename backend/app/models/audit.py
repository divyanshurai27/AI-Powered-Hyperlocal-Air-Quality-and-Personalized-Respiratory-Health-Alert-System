import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin, utcnow


class AuditLog(UUIDPrimaryKeyMixin, Base):
    """Append-only record of sensitive operations. Never store passwords or tokens here."""

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_user_id_created_at", "user_id", "created_at"),)

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), default=None
    )
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str | None] = mapped_column(String(64), default=None)
    resource_id: Mapped[str | None] = mapped_column(String(64), default=None)
    outcome: Mapped[str] = mapped_column(String(16), default="success")
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
