from fastapi import APIRouter

from app.api.v1.routes import air, auth, exposure, health, patients, recommendations

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(patients.router)
api_router.include_router(air.router)
api_router.include_router(exposure.router)
api_router.include_router(recommendations.router)
