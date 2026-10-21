"""alerts and stored risk explanations

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-21 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "risk_predictions",
        sa.Column("explanation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_table(
        "alerts",
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column(
            "state",
            sa.Enum(
                "created",
                "sent",
                "acknowledged",
                "expired",
                "cancelled",
                name="alert_state",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "risk_level",
            sa.Enum(
                "Low",
                "Moderate",
                "High",
                "Severe",
                name="risk_level",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("risk_probability", sa.Double(), nullable=False),
        sa.Column("message", sa.String(length=256), nullable=False),
        sa.Column("prediction_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("calibration_version", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column("alert_policy_version", sa.String(length=64), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_reason", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "risk_probability >= 0 AND risk_probability <= 1",
            name=op.f("ck_alerts_probability_range"),
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patient_profiles.id"],
            name=op.f("fk_alerts_patient_id_patient_profiles"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["risk_predictions.id"],
            name=op.f("fk_alerts_prediction_id_risk_predictions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
        sa.UniqueConstraint("prediction_id", name=op.f("uq_alerts_prediction_id")),
    )
    op.create_index(
        "ix_alerts_patient_id_created_at", "alerts", ["patient_id", "created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_alerts_patient_id_created_at", table_name="alerts")
    op.drop_table("alerts")
    op.drop_column("risk_predictions", "explanation")
