import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.enums import RiskLevel
from app.models import PatientProfile


def test_production_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(app_env="production", jwt_secret_key="change-me")


def test_production_accepts_real_secret() -> None:
    s = Settings(app_env="production", jwt_secret_key="a-long-random-secret-value")
    assert s.is_production
    assert not s.debug_enabled


def test_risk_levels_are_exactly_the_br02_set() -> None:
    assert {level.value for level in RiskLevel} == {"Low", "Moderate", "High", "Severe"}


def test_profile_incomplete_until_clinical_fields_set() -> None:
    p = PatientProfile(age=40, sex="female", disease_type="asthma", severity=None)
    assert not p.profile_complete
    p.severity = "mild"
    assert p.profile_complete
