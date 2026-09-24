from enum import StrEnum


class UserRole(StrEnum):
    PATIENT = "patient"
    # Elevated roles are deliberately not granted anywhere yet (PRD §50).
    RESEARCHER = "researcher"


class DiseaseType(StrEnum):
    ASTHMA = "asthma"
    COPD = "copd"


class Severity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    VERY_SEVERE = "very_severe"


class Sex(StrEnum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class ConsentStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"


class RiskLevel(StrEnum):
    """BR-02: the only valid alert levels. Thresholds live in versioned policy, not here."""

    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    SEVERE = "Severe"
