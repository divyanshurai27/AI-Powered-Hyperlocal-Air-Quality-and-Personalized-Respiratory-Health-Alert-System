import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ConsentStatus, DiseaseType, Severity, Sex


class SavedPlace(BaseModel):
    label: str = Field(default="Home", min_length=1, max_length=80)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class PatientProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    patient_id: uuid.UUID = Field(validation_alias="id")
    age: int | None
    sex: Sex | None
    disease_type: DiseaseType | None
    severity: Severity | None
    consent_status: ConsentStatus
    consent_updated_at: datetime | None
    baseline_information: dict[str, Any]
    profile_complete: bool
    home: SavedPlace | None
    work: SavedPlace | None
    created_at: datetime
    updated_at: datetime


class PatientProfileUpdate(BaseModel):
    """PATCH semantics: only fields present in the request body are changed."""

    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=0, le=120)
    sex: Sex | None = None
    disease_type: DiseaseType | None = None
    severity: Severity | None = None
    consent_status: ConsentStatus | None = None
    baseline_information: dict[str, Any] | None = None
    home: SavedPlace | None = None  # null clears it
    work: SavedPlace | None = None
