import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import get_settings
from app.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_password_hash_is_argon2_and_verifies() -> None:
    h = hash_password("s3cret-pass")
    assert h.startswith("$argon2")
    assert verify_password("s3cret-pass", h)
    assert not verify_password("wrong", h)


def test_verify_password_rejects_garbage_hash() -> None:
    assert not verify_password("anything", "not-a-hash")


def test_same_password_hashes_differently() -> None:
    assert hash_password("x" * 10) != hash_password("x" * 10)


def test_access_token_round_trip() -> None:
    uid = uuid.uuid4()
    token, payload = create_access_token(uid)
    decoded = decode_token(token, "access")
    assert decoded.sub == uid
    assert decoded.jti == payload.jti


def test_refresh_token_cannot_be_used_as_access_token() -> None:
    token, _ = create_refresh_token(uuid.uuid4())
    with pytest.raises(TokenError):
        decode_token(token, "access")


def test_access_token_cannot_be_used_as_refresh_token() -> None:
    token, _ = create_access_token(uuid.uuid4())
    with pytest.raises(TokenError):
        decode_token(token, "refresh")


def test_expired_token_rejected() -> None:
    s = get_settings()
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "type": "access",
            "jti": "x",
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),
        },
        s.jwt_secret_key,
        algorithm=s.jwt_algorithm,
    )
    with pytest.raises(TokenError):
        decode_token(token, "access")


def test_token_signed_with_other_key_rejected() -> None:
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "type": "access",
            "jti": "x",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        "attacker-controlled-key-that-is-long-enough-0123",
        algorithm="HS256",
    )
    with pytest.raises(TokenError):
        decode_token(token, "access")


def test_alg_none_token_rejected() -> None:
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "type": "access",
            "jti": "x",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        key=None,
        algorithm="none",
    )
    with pytest.raises(TokenError):
        decode_token(token, "access")


def test_non_uuid_subject_rejected() -> None:
    s = get_settings()
    token = jwt.encode(
        {
            "sub": "1 OR 1=1",
            "type": "access",
            "jti": "x",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        s.jwt_secret_key,
        algorithm=s.jwt_algorithm,
    )
    with pytest.raises(TokenError):
        decode_token(token, "access")
