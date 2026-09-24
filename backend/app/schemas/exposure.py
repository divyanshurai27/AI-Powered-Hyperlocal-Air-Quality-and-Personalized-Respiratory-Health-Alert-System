from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.exposure import MicroEnvironment


class LocationFixIn(BaseModel):
    timestamp: datetime = Field(description="When the fix was taken (ISO 8601 with timezone)")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0)
    activity_context: MicroEnvironment = MicroEnvironment.UNKNOWN


class LocationBatchIn(BaseModel):
    fixes: list[LocationFixIn] = Field(min_length=1, max_length=500)


class LocationBatchOut(BaseModel):
    received: int
    stored: int = Field(description="New fixes; exact repeats of a stored timestamp are ignored")


class ExposureHourOut(BaseModel):
    hour: datetime
    exposure: float | None
    ambient: float | None
    factor: float | None
    microenvironment: str | None
    assumed_outdoor: bool
    missing_reason: str | None


class ExposureWindowOut(BaseModel):
    window_hours: int
    mean: float | None
    maximum: float | None
    cumulative: float | None = Field(description="µg/m³·h over hours with an estimate")
    coverage: float
    complete: bool


class ExposureOut(BaseModel):
    pollutant: str
    unit: str = "µg/m³"
    model_version: str
    is_estimate: bool = True
    note: str = (
        "Estimated from nearby monitors and your location context; not a personal measurement."
    )
    current: ExposureHourOut | None
    windows: list[ExposureWindowOut]
    lagged: dict[str, float | None]
    hours: list[ExposureHourOut] | None = None
