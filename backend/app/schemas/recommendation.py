from datetime import datetime

from pydantic import BaseModel, Field


class GuidanceHour(BaseModel):
    time: datetime = Field(description="End of the forecast hour")
    level: str = Field(description="good | caution | limit | avoid")
    category: str = Field(description="NAQI category of the dominant pollutant")
    dominant_pollutant: str
    values: dict[str, float]
    waking: bool


class TimeWindow(BaseModel):
    start: datetime
    end: datetime


class TodayResponse(BaseModel):
    available: bool
    place_label: str
    latitude: float
    longitude: float
    kind: str = Field(
        default="air-quality guidance",
        description="Rule-based guidance from the forecast. Not the personal risk model.",
    )
    policy_version: str | None = None
    disease_type: str | None = None
    severity: str | None = None
    level: str | None = None
    headline: str | None = None
    worst_category: str | None = None
    worst_pollutant: str | None = None
    health_note: str | None = None
    personalisation_note: str | None = None
    best_window: TimeWindow | None = None
    worst_window: TimeWindow | None = None
    avoid_windows: list[TimeWindow] = []
    hours: list[GuidanceHour] = []
    forecast_origin: datetime | None = None
    model_versions: dict[str, str] = {}
    pollutants_skipped: dict[str, str] = {}
    unavailable_reason: str | None = None
    disclaimer: str = (
        "Informational air-quality guidance for a research prototype. "
        "It is not medical advice and does not replace your care plan."
    )
