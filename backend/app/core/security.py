"""Password hashing (Argon2) and JWT access/refresh tokens (PRD §20)."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import get_settings

TokenType = Literal["access", "refresh"]

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


@dataclass(frozen=True)
class TokenPayload:
    sub: uuid.UUID
    type: TokenType
    jti: str
    exp: datetime


class TokenError(Exception):
    pass


def _create_token(
    user_id: uuid.UUID, token_type: TokenType, ttl: timedelta
) -> tuple[str, TokenPayload]:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = TokenPayload(sub=user_id, type=token_type, jti=uuid.uuid4().hex, exp=now + ttl)
    token = jwt.encode(
        {
            "sub": str(payload.sub),
            "type": payload.type,
            "jti": payload.jti,
            "iat": now,
            "exp": payload.exp,
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return token, payload


def create_access_token(user_id: uuid.UUID) -> tuple[str, TokenPayload]:
    ttl = timedelta(minutes=get_settings().access_token_expire_minutes)
    return _create_token(user_id, "access", ttl)


def create_refresh_token(user_id: uuid.UUID) -> tuple[str, TokenPayload]:
    ttl = timedelta(days=get_settings().refresh_token_expire_days)
    return _create_token(user_id, "refresh", ttl)


def decode_token(token: str, expected_type: TokenType) -> TokenPayload:
    """Verify signature, expiry and token type. Raises TokenError on any failure."""
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "type", "jti", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc

    if claims.get("type") != expected_type:
        raise TokenError(f"expected {expected_type} token")
    try:
        sub = uuid.UUID(claims["sub"])
    except (ValueError, TypeError) as exc:
        raise TokenError("invalid subject") from exc

    return TokenPayload(
        sub=sub,
        type=claims["type"],
        jti=claims["jti"],
        exp=datetime.fromtimestamp(claims["exp"], UTC),
    )
