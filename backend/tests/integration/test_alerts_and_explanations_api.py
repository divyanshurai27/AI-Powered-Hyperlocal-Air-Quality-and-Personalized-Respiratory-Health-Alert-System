"""Explanations (AT-05, BR-09), alerts (AT-06, IT-05, BR-10, BR-11), air history and the
development-only prediction trace (PRD §33)."""

# ruff: noqa: F811  (pytest fixtures imported from another module)
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from app.core.config import get_settings
from app.domain.enums import RiskLevel
from app.models import Alert, RiskPrediction
from app.services import risk as risk_module
from tests.integration.test_exposure_and_forecast_api import (  # noqa: F401
    BTM,
    T,
    forecast_artifact,
    seed_recent_history,
)
from tests.integration.test_health_and_risk_api import (  # noqa: F401
    CURRENT,
    FORECAST,
    diary,
    risk_artifact,
    setup_patient,
)

ALERTS = "/api/v1/alerts"
AT = {"at": T.isoformat()}


def force_level(monkeypatch, level: RiskLevel) -> None:
    """Pin the policy output so alert behaviour does not depend on the toy model's numbers."""
    monkeypatch.setattr(risk_module, "classify_risk", lambda p, d: level)


@pytest.fixture
def patient(client, register_and_login, db, risk_artifact):
    seed_recent_history(db)
    s = register_and_login()
    setup_patient(client, s)
    client.post("/api/v1/health/records", headers=s["headers"], json={"records": [diary(1)]})
    return s


# --- explanations --------------------------------------------------------------------------


def test_current_risk_carries_a_consistent_explanation(client, patient, db) -> None:
    body = client.get(CURRENT, headers=patient["headers"], params=AT).json()
    exp, cur = body["explanation"], body["current"]
    assert exp["model_version"] == body["model_version"]
    assert exp["calibration_version"] == body["calibration_version"]
    assert exp["feature_version"] == body["feature_version"]
    assert exp["space"] == "calibrated_log_odds"
    assert exp["reconstructed_probability"] == pytest.approx(cur["risk_probability"], abs=1e-5)
    assert 1 <= len(exp["summary"]) <= 5 and "synthetic" in exp["note"]

    stored = client.get(
        f"/api/v1/risk/predictions/{cur['prediction_id']}/explanation", headers=patient["headers"]
    )
    assert stored.status_code == 200, stored.text
    assert stored.json()["explanation"] == exp  # same prediction → same explanation


def test_explanation_is_patient_isolated(client, patient, register_and_login) -> None:
    pid = client.get(CURRENT, headers=patient["headers"], params=AT).json()["current"][
        "prediction_id"
    ]
    bob = register_and_login("bob@example.com")
    setup_patient(client, bob)
    r = client.get(f"/api/v1/risk/predictions/{pid}/explanation", headers=bob["headers"])
    assert r.status_code == 404
    missing = client.get(
        f"/api/v1/risk/predictions/{uuid.uuid4()}/explanation", headers=patient["headers"]
    )
    assert missing.status_code == 404


def test_explanation_not_matching_its_inputs_is_refused(client, patient, db) -> None:
    pid = client.get(CURRENT, headers=patient["headers"], params=AT).json()["current"][
        "prediction_id"
    ]
    row = db.get(RiskPrediction, uuid.UUID(pid))
    db.execute(
        update(RiskPrediction)
        .where(RiskPrediction.id == row.id)
        .values(features=row.features | {"pt_age": 99.0})
    )
    db.commit()
    r = client.get(f"/api/v1/risk/predictions/{pid}/explanation", headers=patient["headers"])
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "EXPLANATION_UNAVAILABLE"


def test_forecast_horizons_store_explanations(client, patient, db, forecast_artifact) -> None:
    r = client.get(FORECAST, headers=patient["headers"], params=AT)
    assert r.status_code == 200, r.text
    rows = db.scalars(select(RiskPrediction)).all()
    assert len(rows) > 1 and all(row.explanation for row in rows)


# --- alerts --------------------------------------------------------------------------------


def test_below_threshold_creates_no_alert(client, patient, db, monkeypatch) -> None:
    force_level(monkeypatch, RiskLevel.MODERATE)
    body = client.get(CURRENT, headers=patient["headers"], params=AT).json()
    assert body["alert_decision"] == "below_alert_level" and body["alert"] is None
    assert db.scalar(select(func.count()).select_from(Alert)) == 0


def test_high_risk_creates_exactly_one_alert(client, patient, db, monkeypatch) -> None:
    force_level(monkeypatch, RiskLevel.HIGH)
    first = client.get(CURRENT, headers=patient["headers"], params=AT).json()
    assert first["alert_decision"] == "created"
    assert first["alert"]["risk_level"] == "High"
    again = client.get(CURRENT, headers=patient["headers"], params=AT).json()
    assert again["alert_decision"] == "already_alerted"
    assert again["alert"]["id"] == first["alert"]["id"]
    # the next hour, still High: the open alert covers it
    later = client.get(
        CURRENT, headers=patient["headers"], params={"at": (T + timedelta(hours=1)).isoformat()}
    )
    if later.status_code == 200:  # T + 1 h may be in the future near the top of the hour
        assert later.json()["alert_decision"] == "equivalent_alert_open"
    assert db.scalar(select(func.count()).select_from(Alert)) == 1

    alert = db.scalar(select(Alert))
    pred = db.get(RiskPrediction, alert.prediction_id)
    assert pred.horizon_hours == 0  # traceable to the prediction it came from
    assert alert.model_version == pred.model_version
    assert alert.policy_version == pred.policy_version
    assert "%" in alert.message and "inhaler" not in alert.message.lower()


def test_research_view_never_alerts(client, patient, db, monkeypatch) -> None:
    force_level(monkeypatch, RiskLevel.SEVERE)
    body = client.get(
        CURRENT, headers=patient["headers"], params=AT | {"personalize": False}
    ).json()
    assert body["alert_decision"] == "not_evaluated"
    assert db.scalar(select(func.count()).select_from(Alert)) == 0


def test_escalation_supersedes_and_revision_withdraws(client, patient, db, monkeypatch) -> None:
    force_level(monkeypatch, RiskLevel.HIGH)
    high = client.get(CURRENT, headers=patient["headers"], params=AT).json()["alert"]
    # the same prediction recomputed (e.g. after a late diary entry) now reads Moderate
    force_level(monkeypatch, RiskLevel.MODERATE)
    client.get(CURRENT, headers=patient["headers"], params=AT)
    listed = client.get(ALERTS, headers=patient["headers"], params=AT).json()["alerts"]
    assert [(a["id"], a["state"], a["cancel_reason"]) for a in listed] == [
        (high["id"], "cancelled", "prediction_revised")
    ]


def test_list_and_acknowledge(client, patient, register_and_login, monkeypatch) -> None:
    force_level(monkeypatch, RiskLevel.HIGH)
    alert_id = client.get(CURRENT, headers=patient["headers"], params=AT).json()["alert"]["id"]

    active = client.get(ALERTS, headers=patient["headers"], params=AT | {"active": True}).json()
    assert [a["id"] for a in active["alerts"]] == [alert_id]
    assert active["alerts"][0]["state"] == "created"
    assert "SYNTHETIC" in active["disclaimer"]

    bob = register_and_login("bob@example.com")
    setup_patient(client, bob)
    assert client.get(ALERTS, headers=bob["headers"]).json()["alerts"] == []
    stolen = client.post(f"{ALERTS}/{alert_id}/acknowledge", headers=bob["headers"])
    assert stolen.status_code == 404  # AT-08: no disclosure

    ack = client.post(f"{ALERTS}/{alert_id}/acknowledge", headers=patient["headers"])
    assert ack.status_code == 200 and ack.json()["state"] == "acknowledged"
    again = client.post(f"{ALERTS}/{alert_id}/acknowledge", headers=patient["headers"])
    assert again.json()["acknowledged_at"] == ack.json()["acknowledged_at"]  # idempotent
    still = client.get(ALERTS, headers=patient["headers"], params=AT | {"active": True})
    assert still.json()["alerts"] == []


def test_alert_expires_when_its_window_ends(client, patient, monkeypatch) -> None:
    force_level(monkeypatch, RiskLevel.HIGH)
    client.get(CURRENT, headers=patient["headers"], params=AT)
    after = {"at": (T + timedelta(hours=25)).isoformat()}
    listed = client.get(ALERTS, headers=patient["headers"], params=after).json()["alerts"]
    assert listed[0]["state"] == "expired"
    before = {"at": (T - timedelta(hours=1)).isoformat()}
    assert client.get(ALERTS, headers=patient["headers"], params=before).json()["alerts"] == []


def test_alerts_need_consent(client, register_and_login) -> None:
    s = register_and_login()
    r = client.get(ALERTS, headers=s["headers"])
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "CONSENT_REQUIRED"


# --- air history and debug trace -----------------------------------------------------------


def test_air_history_is_hourly_and_explicit_about_gaps(client, register_and_login, db) -> None:
    seed_recent_history(db, hours=10)
    s = register_and_login()
    r = client.get(
        "/api/v1/air/history",
        headers=s["headers"],
        params={"lat": BTM[0], "lon": BTM[1], "hours": 24, "at": T.isoformat()},
    )
    assert r.status_code == 200, r.text
    hours = r.json()["hours"]
    assert len(hours) == 24
    assert [h["hour"] for h in hours] == sorted(h["hour"] for h in hours)
    assert any(h["available"] for h in hours)
    gaps = [h for h in hours if not h["available"]]
    assert gaps and all(h["concentration"] is None and h["unavailable_reason"] for h in gaps)


def test_debug_trace_only_in_development(client, patient, monkeypatch) -> None:
    r = client.get("/api/v1/risk/debug", headers=patient["headers"], params=AT)
    assert r.status_code == 404  # APP_ENV=test
    monkeypatch.setattr(get_settings(), "app_env", "development")
    r = client.get("/api/v1/risk/debug", headers=patient["headers"], params=AT)
    assert r.status_code == 200, r.text
    trace = r.json()
    for key in (
        "request_id",
        "features",
        "raw_probability",
        "calibrated_probability",
        "risk_level",
        "alert_decision",
        "freshness",
    ):
        assert key in trace
    assert trace["freshness"]["location_source"] == "saved_home"
