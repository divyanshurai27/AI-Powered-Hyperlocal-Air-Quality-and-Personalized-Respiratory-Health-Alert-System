"""Import every model here so Alembic autogenerate sees the full metadata."""

from app.models.alert import Alert
from app.models.audit import AuditLog
from app.models.environment import (
    AirQualityObservation,
    IngestionRun,
    Station,
    WeatherObservation,
)
from app.models.health import HealthRecord, RiskPrediction
from app.models.location import LocationRecord
from app.models.patient import PatientProfile
from app.models.user import RefreshToken, User

__all__ = [
    "AirQualityObservation",
    "Alert",
    "AuditLog",
    "HealthRecord",
    "IngestionRun",
    "LocationRecord",
    "PatientProfile",
    "RefreshToken",
    "RiskPrediction",
    "Station",
    "User",
    "WeatherObservation",
]
