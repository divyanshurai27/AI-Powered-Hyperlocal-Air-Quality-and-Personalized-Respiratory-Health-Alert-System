import re
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.v1.router import api_router
from app.api.v1.routes import health
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger, request_id_var

logger = get_logger("app.request")

# Accept a client-supplied ID only if it is short and safe to echo into logs/headers.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=__version__,
        description=(
            "Research prototype. Risk outputs are informational only and are not "
            "medical advice, diagnosis or medication guidance (BR-12)."
        ),
        # Interactive docs are a development convenience only.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

    register_exception_handlers(app)
    app.include_router(health.router)  # unversioned /health for probes
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    # Web demo client. Same-origin, so it needs no CORS; it only calls the public API.
    app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="web")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/app/")

    return app


app = create_app()
