import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.risk import DISCLAIMER


class AlertOut(BaseModel):
    id: uuid.UUID
    prediction_id: uuid.UUID
    state: str = Field(description="created | sent | acknowledged | expired | cancelled")
    risk_level: str
    risk_probability: float = Field(ge=0, le=1)
    risk_percent: str
    message: str = Field(description="Level and percentage only; never medication guidance")
    prediction_time: datetime
    target_window_start: datetime
    target_window_end: datetime
    expires_at: datetime
    created_at: datetime
    acknowledged_at: datetime | None
    cancel_reason: str | None
    model_version: str
    calibration_version: str
    policy_version: str
    alert_policy_version: str


class AlertListOut(BaseModel):
    alerts: list[AlertOut]
    reference_time: datetime = Field(description="Instant the states were evaluated at")
    disclaimer: str = DISCLAIMER
