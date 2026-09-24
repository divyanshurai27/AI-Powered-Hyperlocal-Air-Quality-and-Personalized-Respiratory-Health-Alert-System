from fastapi import APIRouter, status

from app.api.deps import CurrentUser, DbSession
from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenPair,
    UserOut,
)
from app.schemas.common import ErrorResponse
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
)
def register(body: RegisterRequest, db: DbSession) -> RegisterResponse:
    user, profile = AuthService(db).register(body.email, body.password)
    return RegisterResponse(user=UserOut.model_validate(user), patient_id=profile.id)


@router.post("/login", response_model=TokenPair, responses={401: {"model": ErrorResponse}})
def login(body: LoginRequest, db: DbSession) -> TokenPair:
    return AuthService(db).login(body.email, body.password)


@router.post("/refresh", response_model=TokenPair, responses={401: {"model": ErrorResponse}})
def refresh(body: RefreshRequest, db: DbSession) -> TokenPair:
    return AuthService(db).refresh(body.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(body: LogoutRequest, user: CurrentUser, db: DbSession) -> None:
    AuthService(db).logout(user, body.refresh_token)
