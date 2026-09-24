import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import PatientProfile, RefreshToken, User


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, user_id: uuid.UUID) -> User | None:
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).where(User.email == email))

    def add_with_profile(self, email: str, password_hash: str) -> tuple[User, PatientProfile]:
        user = User(email=email, password_hash=password_hash)
        profile = PatientProfile(user=user)
        self.db.add_all([user, profile])
        self.db.flush()
        return user, profile


class RefreshTokenRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, user_id: uuid.UUID, jti: str, expires_at: datetime) -> RefreshToken:
        token = RefreshToken(user_id=user_id, jti=jti, expires_at=expires_at)
        self.db.add(token)
        self.db.flush()
        return token

    def get_by_jti_for_update(self, jti: str) -> RefreshToken | None:
        # Row lock so two concurrent refreshes of the same token can't both succeed.
        return self.db.scalar(select(RefreshToken).where(RefreshToken.jti == jti).with_for_update())

    def revoke_all_for_user(self, user_id: uuid.UUID, now: datetime) -> None:
        self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
