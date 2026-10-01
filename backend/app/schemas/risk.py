from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "Research prototype trained on a SYNTHETIC patient cohort. This estimate is informational "
    "only: it is not a diagnosis or medical advice, and it never replaces your care plan or "
    "your clinician's instructions."
)


class PersonalizationOut(BaseModel):
    version: str
    offset: float = Field(description="Log-odds shift from your own history, bounded ±ln 2")
    observed_events: int
    expected_events: float
    observed_days: float


class RiskEstimateOut(BaseModel):
    horizon_hours: int = Field(description="0 = now; 1..24 = projected at that hour")
    target_window_start: datetime
    target_window_end: datetime
    risk_probability: float = Field(ge=0, le=1, description="Calibrated, after personalization")
    risk_level: str = Field(description="Low | Moderate | High | Severe (versioned policy)")
    raw_probability: float = Field(ge=0, le=1, description="Model output before calibration")
    confidence: str = Field(description="normal | low (see data_quality.reasons)")
    data_quality: dict


class RiskHorizonOut(BaseModel):
    horizon_hours: int
    time: datetime
    available: bool
    unavailable_reason: str | None = None
    estimate: RiskEstimateOut | None = None


class RiskHeader(BaseModel):
    target: str
    prediction_time: datetime
    disease_type: str
    severity: str | None
    model_version: str
    calibration_version: str
    feature_version: str
    policy_version: str
    thresholds: dict[str, float]
    personalization: PersonalizationOut | None
    location_source: str = Field(description="device | saved_home | none")
    label_source: str = "synthetic cohort (not clinical outcomes)"
    disclaimer: str = DISCLAIMER


class RiskCurrentOut(RiskHeader):
    current: RiskEstimateOut


class RiskForecastOut(RiskHeader):
    current: RiskEstimateOut
    horizons: list[RiskHorizonOut]
    forecast_model_versions: dict[str, str]
    pollutants_skipped: dict[str, str]
    assumption: str = (
        "Projected with the 24 h air-quality forecast at your current place and setting; "
        "symptoms and weather are held at their latest values."
    )


class RiskHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prediction_time: datetime
    horizon_hours: int
    target_window_start: datetime
    target_window_end: datetime
    risk_probability: float
    risk_level: str
    confidence: str
    model_version: str
    calibration_version: str
    policy_version: str
    personalization: str


class RiskHistoryOut(BaseModel):
    predictions: list[RiskHistoryItem]
    disclaimer: str = DISCLAIMER
