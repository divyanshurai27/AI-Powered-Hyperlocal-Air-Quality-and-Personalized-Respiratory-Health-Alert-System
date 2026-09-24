from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

INSECURE_SECRETS = {"", "change-me"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AeroGard"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    database_url: str = (
        "postgresql+psycopg://aerogard:aerogard_dev_password@localhost:5433/aerogard"
    )
    test_database_url: str = (
        "postgresql+psycopg://aerogard:aerogard_dev_password@localhost:5433/aerogard_test"
    )

    jwt_secret_key: str = "change-me"  # noqa: S105  (dev default; rejected in production)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, gt=0)
    refresh_token_expire_days: int = Field(default=7, gt=0)

    # --- External sources ---
    openaq_api_key: str = ""
    openaq_base_url: str = "https://api.openaq.org/v3"
    data_gov_in_api_key: str = ""
    data_gov_in_base_url: str = "https://api.data.gov.in"
    # "Real time Air Quality Index from various locations" (CPCB)
    data_gov_in_aqi_resource_id: str = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    http_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Study area (Bengaluru) ---
    study_area_name: str = "Bengaluru"
    study_area_latitude: float = 12.9716
    study_area_longitude: float = 77.5946
    # OpenAQ caps location search at 25 km.
    study_area_radius_km: float = Field(default=25.0, gt=0, le=25)

    # --- Freshness and spatial matching (BR-05, PRD §10) ---
    aq_max_age_hours: float = Field(default=3.0, gt=0)
    weather_max_age_hours: float = Field(default=3.0, gt=0)
    spatial_radius_km: float = Field(default=10.0, gt=0)
    spatial_exact_match_km: float = Field(default=0.5, ge=0)
    spatial_max_stations: int = Field(default=4, ge=1)
    # BR-07: a location fix older than this no longer represents where the patient is.
    location_max_age_hours: float = Field(default=2.0, gt=0)

    # --- Forecast artifacts (PRD §39) ---
    aq_forecast_artifact_dir: str = "ml/artifacts/aq_forecast"
    aq_forecast_model_version: str = "1.1.0"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def debug_enabled(self) -> bool:
        """Debug-only surfaces (prediction trace, verbose errors) are never on in production."""
        return self.app_env == "development"

    @model_validator(mode="after")
    def _reject_insecure_secret_in_production(self) -> "Settings":
        if self.is_production and self.jwt_secret_key in INSECURE_SECRETS:
            raise ValueError("JWT_SECRET_KEY must be set to a strong secret in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
