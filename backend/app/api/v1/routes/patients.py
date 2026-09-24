from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.schemas.common import ErrorResponse
from app.schemas.patient import PatientProfileOut, PatientProfileUpdate
from app.services.patients import PatientService

router = APIRouter(
    prefix="/patients",
    tags=["patients"],
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)


@router.get("/me", response_model=PatientProfileOut)
def get_my_profile(user: CurrentUser, db: DbSession) -> PatientProfileOut:
    return PatientProfileOut.model_validate(PatientService(db).get_own_profile(user))


@router.patch("/me", response_model=PatientProfileOut, responses={422: {"model": ErrorResponse}})
def update_my_profile(
    body: PatientProfileUpdate, user: CurrentUser, db: DbSession
) -> PatientProfileOut:
    return PatientProfileOut.model_validate(PatientService(db).update_own_profile(user, body))
