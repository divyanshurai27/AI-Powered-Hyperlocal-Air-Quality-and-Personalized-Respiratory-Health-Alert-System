from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.observations import QualityFlag


class StationContribution(BaseModel):
    station_id: str
    distance_km: float
    weight: float = Field(description="Share of the estimate, 0–1; weights sum to 1")
    concentration: float
    observed_at: datetime
    quality_flag: QualityFlag


class PollutantEstimate(BaseModel):
    pollutant: str
    available: bool
    concentration: float | None = None
    unit: str
    method: str | None = Field(default=None, description="exact_station | idw")
    method_version: str | None = None
    data_age_minutes: float | None = Field(
        default=None, description="Age of the oldest contributing reading at query time"
    )
    nearest_station_km: float | None = None
    contributions: list[StationContribution] = []
    unavailable_reason: str | None = None
    unavailable_detail: str | None = None


class CurrentAirResponse(BaseModel):
    latitude: float
    longitude: float
    at: datetime
    source_note: str = (
        "Estimated from nearby CPCB/KSPCB reference monitors (via OpenAQ); "
        "not a direct measurement at this location."
    )
    pollutants: list[PollutantEstimate]


class StationOut(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    provider: str | None
    is_reference_monitor: bool
    last_observed_at: datetime | None


class ForecastHour(BaseModel):
    horizon_hours: int
    target_time: datetime
    concentration: float


class ForecastResponse(BaseModel):
    latitude: float
    longitude: float
    pollutant: str
    unit: str = "µg/m³"
    available: bool
    origin: datetime | None = Field(
        default=None, description="Latest observed hour the forecast starts from"
    )
    model_version: str | None = None
    feature_version: str | None = None
    station_weights: dict[str, float] = {}
    hours: list[ForecastHour] = []
    unavailable_reason: str | None = None
    unavailable_detail: str | None = None


class MapCell(BaseModel):
    latitude: float
    longitude: float
    now: float | None = Field(description="Current estimate; null where no usable station is near")
    forecast: list[float] | None = Field(default=None, description="Index i = horizon i+1 hours")
    stations: int


class CityMapResponse(BaseModel):
    pollutant: str
    unit: str = "µg/m³"
    at: datetime
    step_deg: float
    origin: datetime | None
    model_version: str | None
    cells: list[MapCell]
