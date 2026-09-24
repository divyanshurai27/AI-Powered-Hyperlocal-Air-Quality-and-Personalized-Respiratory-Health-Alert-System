import uuid

import pytest
from sqlalchemy import select

from app.core.errors import AppError
from app.models import AuditLog, User
from app.services.authorization import authorize_patient_access

ME = "/api/v1/patients/me"


def test_get_profile_requires_auth(client) -> None:
    r = client.get(ME)
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "NOT_AUTHENTICATED"


def test_garbage_bearer_token_rejected(client) -> None:
    r = client.get(ME, headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "INVALID_TOKEN"


def test_new_profile_is_incomplete_and_pending_consent(client, register_and_login) -> None:
    s = register_and_login()
    r = client.get(ME, headers=s["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == s["patient_id"]
    assert body["profile_complete"] is False
    assert body["consent_status"] == "pending"


def test_at01_complete_profile_and_grant_consent(client, register_and_login, db) -> None:
    """AT-01: register → login → create profile → consent → save health information."""
    s = register_and_login()
    r = client.patch(
        ME,
        headers=s["headers"],
        json={
            "age": 34,
            "sex": "female",
            "disease_type": "asthma",
            "severity": "moderate",
            "consent_status": "granted",
            "baseline_information": {"exacerbations_last_12m": 2, "uses_rescue_inhaler": True},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["profile_complete"] is True
    assert body["consent_status"] == "granted"
    assert body["consent_updated_at"] is not None

    actions = set(db.scalars(select(AuditLog.action)).all())
    assert {"consent_changed", "patient_profile_updated"} <= actions
    update_log = db.scalar(select(AuditLog).where(AuditLog.action == "patient_profile_updated"))
    assert "34" not in str(update_log.details)  # audit holds field names, not health values


def test_patch_is_partial(client, register_and_login) -> None:
    s = register_and_login()
    client.patch(ME, headers=s["headers"], json={"age": 60, "disease_type": "copd"})
    body = client.patch(ME, headers=s["headers"], json={"severity": "severe"}).json()
    assert (body["age"], body["disease_type"], body["severity"]) == (60, "copd", "severe")


@pytest.mark.parametrize(
    "payload",
    [
        {"age": -1},
        {"age": 121},
        {"disease_type": "flu"},
        {"severity": "extreme"},
        {"consent_status": None},
        {"patient_id": str(uuid.uuid4())},  # cannot re-point the profile
        {"user_id": str(uuid.uuid4())},
    ],
)
def test_patch_rejects_invalid_input(client, register_and_login, payload) -> None:
    s = register_and_login()
    r = client.patch(ME, headers=s["headers"], json=payload)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "VALIDATION_ERROR"


# --- IT-06 / AT-08: patient isolation --------------------------------------------------


def test_each_user_only_sees_their_own_profile(client, register_and_login) -> None:
    a = register_and_login("a@example.com")
    b = register_and_login("b@example.com")
    client.patch(ME, headers=b["headers"], json={"age": 70, "disease_type": "copd"})

    a_view = client.get(ME, headers=a["headers"]).json()
    assert a_view["patient_id"] == a["patient_id"]
    assert a_view["age"] is None and a_view["disease_type"] is None


def test_patient_id_in_path_is_not_an_access_route(client, register_and_login) -> None:
    a = register_and_login("a@example.com")
    b = register_and_login("b@example.com")
    client.patch(ME, headers=b["headers"], json={"age": 70})
    r = client.get(f"/api/v1/patients/{b['patient_id']}", headers=a["headers"])
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "NOT_FOUND"


def test_authorize_patient_access_denies_other_users_patient(
    client, register_and_login, db
) -> None:
    register_and_login("a@example.com")
    b = register_and_login("b@example.com")
    user_a = db.scalar(select(User).where(User.email == "a@example.com"))

    with pytest.raises(AppError) as exc:
        authorize_patient_access(db, user_a, uuid.UUID(b["patient_id"]))
    assert exc.value.status_code == 404  # same as "doesn't exist": no ID enumeration

    denied = db.scalar(select(AuditLog).where(AuditLog.action == "patient_access_denied"))
    assert denied.resource_id == b["patient_id"]


def test_authorize_patient_access_unknown_id_is_also_404(client, register_and_login, db) -> None:
    register_and_login("a@example.com")
    user_a = db.scalar(select(User))
    with pytest.raises(AppError) as exc:
        authorize_patient_access(db, user_a, uuid.uuid4())
    assert exc.value.status_code == 404
