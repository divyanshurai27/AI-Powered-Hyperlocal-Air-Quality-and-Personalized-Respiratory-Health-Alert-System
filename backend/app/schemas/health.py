import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.health import MEASUREMENT_COLUMNS


class HealthRecordIn(BaseModel):
    """A diary entry or device summary. Send only what you have: absent ≠ zero (BR-06)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(description="When recorded (ISO 8601 with timezone)")
    symptom_score: int | None = Field(default=None, ge=0, le=10, description="0 none … 10 worst")
    rescue_inhaler_puffs: int | None = Field(default=None, ge=0, le=100)
    exacerbation: bool | None = Field(
        default=None, description="A flare needing extra reliever or unscheduled care"
    )
    heart_rate: float | None = Field(default=None, ge=25, le=250, description="Resting, bpm")
    spo2: float | None = Field(default=None, ge=50, le=100, description="%")
    respiratory_rate: float | None = Field(default=None, ge=4, le=80)
    steps: int | None = Field(default=None, ge=0, le=100_000)
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    pef_l_min: float | None = Field(default=None, ge=30, le=1000, description="Peak flow")
    # Clients cannot label their own data "synthetic"; that source is for demo seeding only.
    source: Literal["manual", "health_connect"] = "manual"

    @model_validator(mode="after")
    def _has_a_measurement(self) -> "HealthRecordIn":
        if all(getattr(self, c) is None for c in MEASUREMENT_COLUMNS):
            raise ValueError("a health record needs at least one measurement")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return self


class HealthRecordBatchIn(BaseModel):
    records: list[HealthRecordIn] = Field(min_length=1, max_length=200)


class HealthRecordBatchOut(BaseModel):
    received: int
    stored: int = Field(description="New records; repeats of (timestamp, source) are ignored")


class HealthRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    timestamp: datetime
    symptom_score: int | None
    rescue_inhaler_puffs: int | None
    exacerbation: bool | None
    heart_rate: float | None
    spo2: float | None
    respiratory_rate: float | None
    steps: int | None
    sleep_hours: float | None
    pef_l_min: float | None
    source: str


class HealthRecordList(BaseModel):
    records: list[HealthRecordOut]
