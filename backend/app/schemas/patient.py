import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ConsentStatus, DiseaseType, Severity, Sex


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
