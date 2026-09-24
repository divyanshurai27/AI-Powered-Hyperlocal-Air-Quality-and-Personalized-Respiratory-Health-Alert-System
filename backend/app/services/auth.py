"""Registration, login, token refresh/rotation and logout (PRD §20)."""

from datetime import UTC, datetime

from fastapi import status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from app.models import PatientProfile, User
from app.repositories.audit import AuditRepository
from app.repositories.users import RefreshTokenRepository, UserRepository
from app.schemas.auth import TokenPair

logger = get_logger(__name__)

# Verified against when the email is unknown, so response time doesn't reveal which emails exist.
_DUMMY_HASH = hash_password("aerogard-timing-equalizer")


def _invalid_credentials() -> AppError:
    return AppError(
        ErrorCode.INVALID_CREDENTIALS,
        "Email or password is incorrect.",
        status.HTTP_401_UNAUTHORIZED,
    )


def _invalid_token() -> AppError:
    return AppError(
        ErrorCode.INVALID_TOKEN,
        "Refresh token is invalid, expired or revoked.",
        status.HTTP_401_UNAUTHORIZED,
    )


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.users = UserRepository(db)
        self.tokens = RefreshTokenRepository(db)
        self.audit = AuditRepository(db)

    def register(self, email: str, password: str) -> tuple[User, PatientProfile]:
        if self.users.get_by_email(email) is not None:
            raise self._email_taken()
        try:
            user, profile = self.users.add_with_profile(email, hash_password(password))
            self.audit.write(
                "user_registered", user_id=user.id, resource_type="user", resource_id=user.id
            )
            self.db.commit()
        except IntegrityError:
            # Lost a race with a concurrent registration of the same email.
            self.db.rollback()
            raise self._email_taken() from None
        logger.info("user_registered", extra={"user_id": str(user.id)})
        return user, profile

    def login(self, email: str, password: str) -> TokenPair:
        user = self.users.get_by_email(email)
        if user is None:
            verify_password(password, _DUMMY_HASH)
            self._audit_failed_login(None, "unknown_email")
            raise _invalid_credentials()
        if not verify_password(password, user.password_hash):
            self._audit_failed_login(user, "bad_password")
            raise _invalid_credentials()
        if not user.is_active:
            self._audit_failed_login(user, "inactive")
            raise _invalid_credentials()

        if password_needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)

        pair = self._issue_pair(user)
        self.audit.write(
            "login_succeeded", user_id=user.id, resource_type="user", resource_id=user.id
        )
        self.db.commit()
        return pair

    def refresh(self, refresh_token: str) -> TokenPair:
        try:
            payload = decode_token(refresh_token, "refresh")
        except TokenError:
            raise _invalid_token() from None

        record = self.tokens.get_by_jti_for_update(payload.jti)
        now = datetime.now(UTC)
        if record is None or record.user_id != payload.sub:
            raise _invalid_token()

        if record.revoked_at is not None:
            # A revoked token being replayed suggests it was stolen: kill every session.
            self.tokens.revoke_all_for_user(record.user_id, now)
            self.audit.write(
                "refresh_token_reuse_detected",
                user_id=record.user_id,
                resource_type="user",
                resource_id=record.user_id,
                outcome="failure",
            )
            self.db.commit()
            raise _invalid_token()

        user = self.users.get(record.user_id)
        if record.expires_at <= now or user is None or not user.is_active:
            raise _invalid_token()

        record.revoked_at = now
        pair = self._issue_pair(user)
        self.db.commit()
        return pair

    def logout(self, user: User, refresh_token: str) -> None:
        """Revoke the given refresh token. Idempotent; never reveals whether the token existed."""
        try:
            payload = decode_token(refresh_token, "refresh")
        except TokenError:
            return
        record = self.tokens.get_by_jti_for_update(payload.jti)
        if record is None or record.user_id != user.id or record.revoked_at is not None:
            self.db.rollback()
            return
        record.revoked_at = datetime.now(UTC)
        self.audit.write("logout", user_id=user.id, resource_type="user", resource_id=user.id)
        self.db.commit()

    def _issue_pair(self, user: User) -> TokenPair:
        access, _ = create_access_token(user.id)
        refresh, refresh_payload = create_refresh_token(user.id)
        self.tokens.add(user.id, refresh_payload.jti, refresh_payload.exp)
        return TokenPair(
            access_token=access,
            refresh_token=refresh,
            expires_in=get_settings().access_token_expire_minutes * 60,
        )

    def _audit_failed_login(self, user: User | None, reason: str) -> None:
        self.audit.write(
            "login_failed",
            user_id=user.id if user else None,
            resource_type="user",
            resource_id=user.id if user else None,
            outcome="failure",
            details={"reason": reason},
        )
        self.db.commit()

    @staticmethod
    def _email_taken() -> AppError:
        return AppError(
            ErrorCode.EMAIL_ALREADY_REGISTERED,
            "An account with this email already exists.",
            status.HTTP_409_CONFLICT,
        )
