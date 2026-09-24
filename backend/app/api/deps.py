from typing import Annotated

from fastapi import Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.errors import AppError, ErrorCode
from app.core.security import TokenError, decode_token
from app.db.session import get_db
from app.models import User
from app.repositories.users import UserRepository

_bearer = HTTPBearer(auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]

_WWW_AUTH = {"WWW-Authenticate": "Bearer"}


def get_current_user(
    db: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None:
        raise AppError(
            ErrorCode.NOT_AUTHENTICATED,
            "Authentication required.",
            status.HTTP_401_UNAUTHORIZED,
            headers=_WWW_AUTH,
        )
    try:
        payload = decode_token(credentials.credentials, "access")
    except TokenError:
        raise AppError(
            ErrorCode.INVALID_TOKEN,
            "Access token is invalid or expired.",
            status.HTTP_401_UNAUTHORIZED,
            headers=_WWW_AUTH,
        ) from None

    user = UserRepository(db).get(payload.sub)
    if user is None or not user.is_active:
        raise AppError(
            ErrorCode.INVALID_TOKEN,
            "Access token is invalid or expired.",
            status.HTTP_401_UNAUTHORIZED,
            headers=_WWW_AUTH,
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
