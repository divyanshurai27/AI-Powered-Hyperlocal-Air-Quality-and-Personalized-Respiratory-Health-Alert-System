"""Health records (PRD §14) and the risk API (PRD §15, §24): consent, isolation (BR-11),
validation (BR-06), probability bounds (BR-01), versioning (BR-08), idempotency (BR-10)
and explicit unavailable horizons (BR-03)."""

# ruff: noqa: F811  (pytest fixtures imported from another module)
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.ml.calibration import fit_disease_calibrators
from app.ml.risk import SharedHeadRiskModel, save_risk_artifact
from app.ml.risk_features import RISK_FEATURE_NAMES
from app.models import AuditLog, RiskPrediction
from app.services import forecast as forecast_module
from app.services import risk as risk_module
from tests.integration.test_exposure_and_forecast_api import (  # noqa: F401
    BTM,
    T,
    forecast_artifact,
    seed_recent_history,
)
from tests.unit.test_risk_model import FAST, _meta, toy

ME = "/api/v1/patients/me"
RECORDS = "/api/v1/health/records"
CURRENT = "/api/v1/risk/current"
FORECAST = "/api/v1/risk/forecast"
HISTORY = "/api/v1/risk/history"
HOME = {"label": "Home", "latitude": BTM[0], "longitude": BTM[1]}
NOW = datetime.now(UTC)


def setup_patient(client, s, **profile) -> None:
    body = {
        "consent_status": "granted",
        "age": 45,
        "sex": "female",
        "disease_type": "asthma",
        "severity": "moderate",
        "home": HOME,
    } | profile
    r = client.patch(ME, headers=s["headers"], json=body)
    assert r.status_code == 200, r.text


def diary(days_ago: float, **values) -> dict:
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return {"timestamp": ts, "symptom_score": 3, "rescue_inhaler_puffs": 1} | values


@pytest.fixture
def risk_artifact(tmp_path, monkeypatch):
    X, y, d = toy(3000)
    Xv, yv, dv = toy(1500, seed=1)
    model = SharedHeadRiskModel(RISK_FEATURE_NAMES, FAST).fit(X, y, d, Xv, yv, dv)
    cals = fit_disease_calibrators(model.predict_proba(Xv, dv), yv, dv)
    meta = _meta(stratum_daily_rates={"asthma|moderate": 0.02, "copd|severe": 0.02})
    save_risk_artifact(model, cals, meta, tmp_path / "test")
    monkeypatch.setattr(get_settings(), "risk_artifact_dir", str(tmp_path))
    monkeypatch.setattr(get_settings(), "risk_model_version", "test")
    risk_module.get_risk_model.cache_clear()
    yield
    risk_module.get_risk_model.cache_clear()


# --- health records ------------------------------------------------------------------------


def test_health_records_need_consent(client, register_and_login) -> None:
    s = register_and_login()
    r = client.post(RECORDS, headers=s["headers"], json={"records": [diary(1)]})
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "CONSENT_REQUIRED"


def test_health_records_round_trip_and_idempotent(client, register_and_login, db) -> None:
    s = register_and_login()
    setup_patient(client, s)
    batch = {"records": [diary(2), diary(1, exacerbation=True, spo2=95.5)]}
    first = client.post(RECORDS, headers=s["headers"], json=batch)
    assert first.status_code == 201, first.text
    assert first.json() == {"received": 2, "stored": 2}
    again = client.post(RECORDS, headers=s["headers"], json=batch)
    assert again.json() == {"received": 2, "stored": 0}
    rows = client.get(RECORDS, headers=s["headers"]).json()["records"]
    assert len(rows) == 2
    assert rows[0]["exacerbation"] is True and rows[0]["spo2"] == 95.5  # newest first
    assert rows[1]["spo2"] is None  # absent stays absent (BR-06)
    audit = db.scalars(select(AuditLog).where(AuditLog.action == "health_records_recorded")).all()
    assert all(set(a.details) == {"received", "stored"} for a in audit)


@pytest.mark.parametrize(
    "record",
    [
        {"timestamp": NOW.isoformat()},  # no measurement at all
        diary(-1),  # tomorrow
        diary(400),  # older than the backfill window
        diary(1, symptom_score=11),
        diary(1, spo2=30),
        diary(1, source="synthetic"),  # clients cannot claim synthetic data
        {"timestamp": "2026-09-01T10:00:00", "symptom_score": 2},  # naive timestamp
    ],
)
def test_invalid_health_records_are_rejected(client, register_and_login, record) -> None:
    s = register_and_login()
    setup_patient(client, s)
    r = client.post(RECORDS, headers=s["headers"], json={"records": [record]})
    assert r.status_code == 422, r.text


def test_health_records_are_patient_isolated(client, register_and_login) -> None:
    alice, bob = register_and_login(), register_and_login("bob@example.com")
    setup_patient(client, alice)
    setup_patient(client, bob)
    client.post(RECORDS, headers=alice["headers"], json={"records": [diary(1)]})
    assert client.get(RECORDS, headers=bob["headers"]).json()["records"] == []


# --- risk ---------------------------------------------------------------------------------


def test_risk_needs_disease_and_severity(client, register_and_login, risk_artifact) -> None:
    s = register_and_login()
    client.patch(ME, headers=s["headers"], json={"consent_status": "granted"})
    r = client.get(CURRENT, headers=s["headers"])
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "PROFILE_INCOMPLETE"


def test_missing_risk_model_is_503(client, register_and_login, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(get_settings(), "risk_artifact_dir", str(tmp_path / "none"))
    risk_module.get_risk_model.cache_clear()
    s = register_and_login()
    setup_patient(client, s)
    r = client.get(CURRENT, headers=s["headers"])
    risk_module.get_risk_model.cache_clear()
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "MODEL_NOT_AVAILABLE"


def test_current_risk_is_bounded_versioned_and_stored_once(
    client, register_and_login, db, risk_artifact
) -> None:
    seed_recent_history(db)
    s = register_and_login()
    setup_patient(client, s)
    client.post(RECORDS, headers=s["headers"], json={"records": [diary(1), diary(0.5)]})
    params = {"at": T.isoformat()}
    r = client.get(CURRENT, headers=s["headers"], params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    est = body["current"]
    assert 0 <= est["risk_probability"] <= 1 and 0 <= est["raw_probability"] <= 1
    assert est["risk_level"] in {"Low", "Moderate", "High", "Severe"}
    assert est["horizon_hours"] == 0
    for key in ("model_version", "calibration_version", "feature_version", "policy_version"):
        assert body[key]
    assert body["location_source"] == "saved_home"
    assert "location_from_saved_home" in est["data_quality"]["reasons"]
    assert "SYNTHETIC" in body["disclaimer"]
    assert body["personalization"]["observed_days"] >= 0

    client.get(CURRENT, headers=s["headers"], params=params)  # same prediction again
    assert db.scalar(select(func.count()).select_from(RiskPrediction)) == 1  # BR-10
    history = client.get(HISTORY, headers=s["headers"], params={"end": NOW.isoformat()})
    assert len(history.json()["predictions"]) == 1


def test_personalization_can_be_switched_off(client, register_and_login, db, risk_artifact) -> None:
    seed_recent_history(db)
    s = register_and_login()
    setup_patient(client, s)
    r = client.get(CURRENT, headers=s["headers"], params={"personalize": False})
    assert r.status_code == 200, r.text
    assert r.json()["personalization"] is None


def test_no_location_lowers_confidence_and_blocks_forecast(
    client, register_and_login, db, risk_artifact
) -> None:
    seed_recent_history(db)
    s = register_and_login()
    setup_patient(client, s, home=None)
    cur = client.get(CURRENT, headers=s["headers"], params={"at": T.isoformat()}).json()
    assert cur["location_source"] == "none"
    assert cur["current"]["confidence"] == "low"
    assert "no_location" in cur["current"]["data_quality"]["reasons"]
    r = client.get(FORECAST, headers=s["headers"], params={"at": T.isoformat()})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "LOCATION_NOT_SET"


def test_device_location_is_preferred(client, register_and_login, db, risk_artifact) -> None:
    seed_recent_history(db)
    s = register_and_login()
    setup_patient(client, s)
    fix = {
        "timestamp": (T - timedelta(minutes=20)).isoformat(),
        "latitude": BTM[0],
        "longitude": BTM[1],
        "activity_context": "indoor_home",
    }
    assert (
        client.post("/api/v1/location", headers=s["headers"], json={"fixes": [fix]}).status_code
        == 201
    )
    body = client.get(CURRENT, headers=s["headers"], params={"at": T.isoformat()}).json()
    assert body["location_source"] == "device"


def test_forecast_has_24_explicit_horizons(
    client, register_and_login, db, risk_artifact, forecast_artifact
) -> None:
    seed_recent_history(db)
    s = register_and_login()
    setup_patient(client, s)
    r = client.get(FORECAST, headers=s["headers"], params={"at": T.isoformat()})
    assert r.status_code == 200, r.text
    body = r.json()
    hz = body["horizons"]
    assert [h["horizon_hours"] for h in hz] == list(range(1, 25))
    available = [h for h in hz if h["available"]]
    assert available, body
    for h in hz:
        if h["available"]:
            e = h["estimate"]
            assert 0 <= e["risk_probability"] <= 1
            assert "exposure_projected_from_aq_forecast" in e["data_quality"]["reasons"]
        else:
            assert h["estimate"] is None and h["unavailable_reason"]
    # PM10/O3 have no test model: reported as skipped, not silently filled.
    assert set(body["pollutants_skipped"]) == {"pm10", "o3"}
    stored = db.scalar(select(func.count()).select_from(RiskPrediction))
    assert stored == 1 + len(available)


def test_forecast_without_pm25_forecast_marks_every_horizon_unavailable(
    client, register_and_login, db, risk_artifact, monkeypatch, tmp_path
) -> None:
    seed_recent_history(db)
    monkeypatch.setattr(get_settings(), "aq_forecast_artifact_dir", str(tmp_path / "none"))
    forecast_module.get_forecast_model.cache_clear()
    s = register_and_login()
    setup_patient(client, s)
    r = client.get(FORECAST, headers=s["headers"], params={"at": T.isoformat()})
    forecast_module.get_forecast_model.cache_clear()
    assert r.status_code == 200, r.text
    hz = r.json()["horizons"]
    assert len(hz) == 24 and not any(h["available"] for h in hz)
    assert all(h["unavailable_reason"].startswith("pm25_forecast_") for h in hz)


def test_risk_history_is_patient_isolated(client, register_and_login, db, risk_artifact) -> None:
    seed_recent_history(db)
    alice, bob = register_and_login(), register_and_login("bob@example.com")
    setup_patient(client, alice)
    setup_patient(client, bob)
    client.get(CURRENT, headers=alice["headers"], params={"at": T.isoformat()})
    r = client.get(HISTORY, headers=bob["headers"], params={"end": NOW.isoformat()})
    assert r.json()["predictions"] == []


def test_demo_seed_creates_a_usable_synthetic_account(client, db) -> None:
    from app.models import HealthRecord, LocationRecord
    from app.services.demo_seed import seed_demo_patient
    from tests.fixtures.cohort import small_cohort

    cohort = small_cohort(5, 30)
    end = cohort.hours[-1].to_pydatetime()
    out = seed_demo_patient(db, cohort, 1, "demo@example.com", "correct-horse-1", end, days=10)
    assert out["locations"] > 100 and out["health_records"] > 0
    sources = set(db.scalars(select(HealthRecord.source)).all())
    assert sources == {"synthetic"}
    assert db.scalar(select(func.count()).select_from(LocationRecord)) == out["locations"]
    login = client.post(
        "/api/v1/auth/login", json={"email": "demo@example.com", "password": "correct-horse-1"}
    )
    me = client.get(ME, headers={"Authorization": f"Bearer {login.json()['access_token']}"})
    assert me.json()["profile_complete"] and me.json()["consent_status"] == "granted"
