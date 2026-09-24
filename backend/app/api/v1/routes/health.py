from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import __version__
from app.core.config import get_settings
from app.db.session import check_database
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health() -> JSONResponse:
    db_ok = check_database()
    body = HealthResponse(
        status="ok" if db_ok else "degraded",
        database="ok" if db_ok else "unavailable",
        version=__version__,
        environment=get_settings().app_env,
    )
    return JSONResponse(status_code=200 if db_ok else 503, content=body.model_dump())
