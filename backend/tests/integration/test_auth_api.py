from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditLog, PatientProfile, User

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh"
LOGOUT = "/api/v1/auth/logout"


def test_register_creates_user_and_empty_profile(client, db: Session) -> None:
    r = client.post(REGISTER, json={"email": "  Alice@Example.COM ", "password": "correct-horse-1"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["user"]["email"] == "alice@example.com"
    assert "password" not in r.text and "password_hash" not in r.text

    user = db.scalar(select(User))
    assert user.password_hash.startswith("$argon2")
    profile = db.scalar(select(PatientProfile))
    assert str(profile.id) == body["patient_id"]
    assert profile.consent_status == "pending"
    assert profile.disease_type is None  # never defaulted (BR-06)


def test_register_duplicate_email_conflict(client) -> None:
    client.post(REGISTER, json={"email": "a@example.com", "password": "correct-horse-1"})
    r = client.post(REGISTER, json={"email": "A@example.com", "password": "another-pass-2"})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "EMAIL_ALREADY_REGISTERED"


def test_register_validation_uses_error_contract(client) -> None:
    r = client.post(REGISTER, json={"email": "not-an-email", "password": "short"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "VALIDATION_ERROR"
    assert detail["request_id"]
    assert {tuple(e["loc"]) for e in detail["errors"]} >= {("body", "email"), ("body", "password")}


def test_login_success_returns_token_pair(client, register_and_login) -> None:
    session = register_and_login()
    assert session["access"] and session["refresh"]


def test_login_wrong_password_and_unknown_email_look_identical(
    client, register_and_login, db
) -> None:
    register_and_login()
    wrong = client.post(LOGIN, json={"email": "alice@example.com", "password": "nope-nope"})
    unknown = client.post(LOGIN, json={"email": "ghost@example.com", "password": "nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert (
        wrong.json()["detail"]["code"] == unknown.json()["detail"]["code"] == "INVALID_CREDENTIALS"
    )
    assert wrong.json()["detail"]["message"] == unknown.json()["detail"]["message"]

    failures = db.scalars(select(AuditLog).where(AuditLog.action == "login_failed")).all()
    assert len(failures) == 2


def test_refresh_rotates_and_old_token_is_single_use(client, register_and_login) -> None:
    s = register_and_login()
    first = client.post(REFRESH, json={"refresh_token": s["refresh"]})
    assert first.status_code == 200
    new_refresh = first.json()["refresh_token"]
    assert new_refresh != s["refresh"]

    replay = client.post(REFRESH, json={"refresh_token": s["refresh"]})
    assert replay.status_code == 401
    assert replay.json()["detail"]["code"] == "INVALID_TOKEN"


def test_refresh_token_reuse_revokes_all_sessions(client, register_and_login, db) -> None:
    s = register_and_login()
    rotated = client.post(REFRESH, json={"refresh_token": s["refresh"]}).json()["refresh_token"]
    client.post(REFRESH, json={"refresh_token": s["refresh"]})  # replay of the stolen token

    # The legitimately rotated token is now dead too.
    assert client.post(REFRESH, json={"refresh_token": rotated}).status_code == 401
    assert db.scalar(select(AuditLog).where(AuditLog.action == "refresh_token_reuse_detected"))


def test_access_token_rejected_on_refresh_endpoint(client, register_and_login) -> None:
    s = register_and_login()
    assert client.post(REFRESH, json={"refresh_token": s["access"]}).status_code == 401


def test_logout_revokes_refresh_token(client, register_and_login) -> None:
    s = register_and_login()
    r = client.post(LOGOUT, json={"refresh_token": s["refresh"]}, headers=s["headers"])
    assert r.status_code == 204
    assert client.post(REFRESH, json={"refresh_token": s["refresh"]}).status_code == 401


def test_logout_cannot_revoke_another_users_token(client, register_and_login) -> None:
    a = register_and_login("a@example.com")
    b = register_and_login("b@example.com")
    client.post(LOGOUT, json={"refresh_token": b["refresh"]}, headers=a["headers"])
    assert client.post(REFRESH, json={"refresh_token": b["refresh"]}).status_code == 200


def test_logout_requires_authentication(client, register_and_login) -> None:
    s = register_and_login()
    r = client.post(LOGOUT, json={"refresh_token": s["refresh"]})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "NOT_AUTHENTICATED"
