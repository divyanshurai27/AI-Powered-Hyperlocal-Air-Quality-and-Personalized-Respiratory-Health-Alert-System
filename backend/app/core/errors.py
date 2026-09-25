"""Machine-readable error contract (PRD §49).

Every error response has the shape:
    {"detail": {"code": "...", "message": "...", "request_id": "..."}}
Stack traces are never returned to the client.
"""

from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)


class ErrorCode(StrEnum):
    # Platform
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    # Auth
    EMAIL_ALREADY_REGISTERED = "EMAIL_ALREADY_REGISTERED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    NOT_AUTHENTICATED = "NOT_AUTHENTICATED"
    INVALID_TOKEN = "INVALID_TOKEN"  # noqa: S105
    UNAUTHORIZED_RESOURCE = "UNAUTHORIZED_RESOURCE"
    CONSENT_REQUIRED = "CONSENT_REQUIRED"
    # Domain (used from Phase 2 onwards)
    AQ_SOURCE_UNAVAILABLE = "AQ_SOURCE_UNAVAILABLE"
    AQ_DATA_STALE = "AQ_DATA_STALE"
    INVALID_LOCATION = "INVALID_LOCATION"
    LOCATION_NOT_SET = "LOCATION_NOT_SET"
    WEATHER_UNAVAILABLE = "WEATHER_UNAVAILABLE"
    HEALTH_DATA_UNAVAILABLE = "HEALTH_DATA_UNAVAILABLE"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    FEATURE_CONTRACT_MISMATCH = "FEATURE_CONTRACT_MISMATCH"
    PREDICTION_FAILED = "PREDICTION_FAILED"
    ALERT_DUPLICATE = "ALERT_DUPLICATE"


class AppError(Exception):
    """Raise from services; the handler turns it into the error contract."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        extra: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = extra or {}
        self.headers = headers


def error_body(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "detail": {"code": code, "message": message, "request_id": request_id_var.get(), **extra}
    }


_HTTP_STATUS_CODES = {
    status.HTTP_401_UNAUTHORIZED: ErrorCode.NOT_AUTHENTICATED,
    status.HTTP_403_FORBIDDEN: ErrorCode.UNAUTHORIZED_RESOURCE,
    status.HTTP_404_NOT_FOUND: ErrorCode.NOT_FOUND,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, **exc.extra),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=error_body(
                ErrorCode.VALIDATION_ERROR, "Request validation failed.", errors=errors
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_STATUS_CODES.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, message),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(OperationalError)
    async def _db_unavailable(_: Request, exc: OperationalError) -> JSONResponse:
        logger.error("database_unavailable", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_body(
                ErrorCode.DATABASE_UNAVAILABLE, "Database is temporarily unavailable."
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(ErrorCode.INTERNAL_ERROR, "An unexpected error occurred."),
        )
