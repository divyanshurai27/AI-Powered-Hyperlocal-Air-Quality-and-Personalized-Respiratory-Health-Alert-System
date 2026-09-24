import os

# Must run before any `app` import so cached settings/engine point at the test DB.
os.environ["APP_ENV"] = "test"
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-for-production-use-0123456789")

from app.core.config import Settings  # noqa: E402

os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or Settings().test_database_url

from collections.abc import Callable, Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.base import Base  # noqa: E402
from app.db.session import check_database, get_engine, get_sessionmaker  # noqa: E402
from app.main import app  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    return cfg


@pytest.fixture(scope="session")
def migrated_db() -> Iterator[None]:
    """Build the schema through Alembic (not create_all), so every run also tests migrations."""
    if not check_database():
        pytest.skip(
            "Test database unreachable. Start it with `docker compose up -d db` "
            f"(DATABASE_URL={os.environ['DATABASE_URL']})."
        )
    cfg = _alembic_config()
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield


@pytest.fixture
def db(migrated_db: None) -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
        tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
        with get_engine().begin() as conn:
            conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def register_and_login(client: TestClient) -> Callable[..., dict]:
    """Create a user and return {'patient_id', 'access', 'refresh', 'headers'}."""

    def _make(email: str = "alice@example.com", password: str = "correct-horse-1") -> dict:
        reg = client.post("/api/v1/auth/register", json={"email": email, "password": password})
        assert reg.status_code == 201, reg.text
        tokens = client.post("/api/v1/auth/login", json={"email": email, "password": password})
        assert tokens.status_code == 200, tokens.text
        body = tokens.json()
        return {
            "patient_id": reg.json()["patient_id"],
            "access": body["access_token"],
            "refresh": body["refresh_token"],
            "headers": {"Authorization": f"Bearer {body['access_token']}"},
        }

    return _make
