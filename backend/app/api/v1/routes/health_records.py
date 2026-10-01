from datetime import datetime

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentUser, DbSession
from app.schemas.common import ErrorResponse
from app.schemas.health import (
    HealthRecordBatchIn,
    HealthRecordBatchOut,
    HealthRecordList,
    HealthRecordOut,
)
from app.services.health import HealthService

router = APIRouter(
    prefix="/health/records",
    tags=["health records"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)


@router.post("", response_model=HealthRecordBatchOut, status_code=status.HTTP_201_CREATED)
def add_records(
    body: HealthRecordBatchIn, user: CurrentUser, db: DbSession
) -> HealthRecordBatchOut:
    """Upload symptom-diary entries or device summaries. Requires granted consent."""
    stored = HealthService(db).record(user, body.records)
    return HealthRecordBatchOut(received=len(body.records), stored=stored)


@router.get("", response_model=HealthRecordList)
def list_records(
    user: CurrentUser,
    db: DbSession,
    start: datetime | None = Query(None, description="Default: 30 days before `end`"),
    end: datetime | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> HealthRecordList:
    """Your own records, newest first."""
    rows = HealthService(db).list(user, start, end, limit)
    return HealthRecordList(records=[HealthRecordOut.model_validate(r) for r in rows])
