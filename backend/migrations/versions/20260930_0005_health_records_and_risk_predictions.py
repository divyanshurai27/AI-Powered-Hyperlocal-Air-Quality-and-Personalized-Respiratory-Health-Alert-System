"""health records and risk predictions

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-30 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEASUREMENTS = (
    "symptom_score, rescue_inhaler_puffs, exacerbation, heart_rate, spo2, "
    "respiratory_rate, steps, sleep_hours, pef_l_min"
)


def upgrade() -> None:
    op.create_table(
        "health_records",
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symptom_score", sa.SmallInteger(), nullable=True),
        sa.Column("rescue_inhaler_puffs", sa.SmallInteger(), nullable=True),
        sa.Column("exacerbation", sa.Boolean(), nullable=True),
        sa.Column("heart_rate", sa.Double(), nullable=True),
        sa.Column("spo2", sa.Double(), nullable=True),
        sa.Column("respiratory_rate", sa.Double(), nullable=True),
        sa.Column("steps", sa.Integer(), nullable=True),
        sa.Column("sleep_hours", sa.Double(), nullable=True),
        sa.Column("pef_l_min", sa.Double(), nullable=True),
        sa.Column(
            "source",
            sa.Enum(
                "manual",
                "health_connect",
                "synthetic",
                name="health_source",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            f"num_nonnulls({MEASUREMENTS}) > 0", name=op.f("ck_health_records_not_empty")
        ),
        sa.CheckConstraint(
            "symptom_score BETWEEN 0 AND 10", name=op.f("ck_health_records_symptom_score_range")
        ),
        sa.CheckConstraint(
            "rescue_inhaler_puffs BETWEEN 0 AND 100",
            name=op.f("ck_health_records_rescue_puffs_range"),
        ),
        sa.CheckConstraint(
            "heart_rate BETWEEN 25 AND 250", name=op.f("ck_health_records_heart_rate_range")
        ),
        sa.CheckConstraint("spo2 BETWEEN 50 AND 100", name=op.f("ck_health_records_spo2_range")),
        sa.CheckConstraint(
            "respiratory_rate BETWEEN 4 AND 80",
            name=op.f("ck_health_records_respiratory_rate_range"),
        ),
        sa.CheckConstraint(
            "steps BETWEEN 0 AND 100000", name=op.f("ck_health_records_steps_range")
        ),
        sa.CheckConstraint(
            "sleep_hours BETWEEN 0 AND 24", name=op.f("ck_health_records_sleep_hours_range")
        ),
        sa.CheckConstraint(
            "pef_l_min BETWEEN 30 AND 1000", name=op.f("ck_health_records_pef_range")
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patient_profiles.id"],
            name=op.f("fk_health_records_patient_id_patient_profiles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_health_records")),
        sa.UniqueConstraint("patient_id", "timestamp", "source", name="uq_health_records_identity"),
    )
    op.create_index(
        "ix_health_records_patient_id_timestamp",
        "health_records",
        ["patient_id", "timestamp"],
        unique=False,
    )

    op.create_table(
        "risk_predictions",
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("prediction_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hours", sa.SmallInteger(), nullable=False),
        sa.Column("target_window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_probability", sa.Double(), nullable=False),
        sa.Column("risk_probability", sa.Double(), nullable=False),
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
        sa.Column(
            "disease_type",
            sa.Enum(
                "asthma",
                "copd",
                name="disease_type",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("calibration_version", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column("feature_version", sa.String(length=64), nullable=False),
        sa.Column("personalization", sa.String(length=64), nullable=False),
        sa.Column("personal_offset", sa.Double(), nullable=True),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("data_quality", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.CheckConstraint(
            "risk_probability >= 0 AND risk_probability <= 1",
            name=op.f("ck_risk_predictions_probability_range"),
        ),
        sa.CheckConstraint(
            "raw_probability >= 0 AND raw_probability <= 1",
            name=op.f("ck_risk_predictions_raw_range"),
        ),
        sa.CheckConstraint(
            "horizon_hours BETWEEN 0 AND 24", name=op.f("ck_risk_predictions_horizon_range")
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patient_profiles.id"],
            name=op.f("fk_risk_predictions_patient_id_patient_profiles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_risk_predictions")),
        sa.UniqueConstraint(
            "patient_id",
            "prediction_time",
            "horizon_hours",
            "model_version",
            "calibration_version",
            "policy_version",
            "personalization",
            name="uq_risk_predictions_identity",
        ),
    )
    op.create_index(
        "ix_risk_predictions_patient_id_prediction_time",
        "risk_predictions",
        ["patient_id", "prediction_time"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_risk_predictions_patient_id_prediction_time", table_name="risk_predictions")
    op.drop_table("risk_predictions")
    op.drop_index("ix_health_records_patient_id_timestamp", table_name="health_records")
    op.drop_table("health_records")
