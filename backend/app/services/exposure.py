"""Location intake and personal exposure estimation (PRD §13, BR-07, BR-11)."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.domain.exposure import (
    ExposureProfile,
    MicroEnvironment,
    build_exposure_profile,
    estimate_hourly_exposure,
)
from app.domain.observations import valid_coordinates
from app.domain.pollutants import Pollutant
from app.domain.spatial import GeoPoint, LocalEstimate, estimate_local_air_quality
from app.models import LocationRecord, PatientProfile, User
from app.repositories.audit import AuditRepository
from app.repositories.environment import ObservationRepository
from app.services.patients import PatientService

MAX_BATCH = 500
MAX_BACKDATE = timedelta(days=7)
MAX_EXPOSURE_WINDOW_HOURS = 168


@dataclass(frozen=True)
class LocationFix:
    timestamp: datetime
    latitude: float
    longitude: float
    accuracy_m: float | None = None
    activity_context: MicroEnvironment = MicroEnvironment.UNKNOWN
    source: str = "device"


def _hour_grid(end: datetime, hours: int) -> list[datetime]:
    """AQ hour ends (:30 UTC, i.e. whole IST hours) covering (end - hours, end]."""
    anchor = end.replace(minute=30, second=0, microsecond=0)
    if anchor > end:
        anchor -= timedelta(hours=1)
    return [anchor - timedelta(hours=i) for i in reversed(range(hours))]


class ExposureService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.observations = ObservationRepository(db)
        self.audit = AuditRepository(db)

    # --- locations ------------------------------------------------------------------------

    def _patient_with_consent(self, user: User) -> PatientProfile:
        profile = PatientService(self.db).get_own_profile(user)  # ownership check (BR-11)
        if profile.consent_status != "granted":
            raise AppError(
                ErrorCode.CONSENT_REQUIRED,
                "Location and exposure data need consent. Grant it via PATCH /patients/me.",
                status.HTTP_403_FORBIDDEN,
            )
        return profile

    def record_locations(self, user: User, fixes: list[LocationFix]) -> int:
        """Store location fixes idempotently: the same (patient, timestamp) is kept once."""
        profile = self._patient_with_consent(user)
        if not fixes or len(fixes) > MAX_BATCH:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                f"Send between 1 and {MAX_BATCH} location fixes.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        now = datetime.now(UTC)
        rows = []
        for f in fixes:
            ts = f.timestamp.astimezone(UTC)
            if not valid_coordinates(f.latitude, f.longitude):
                raise AppError(
                    ErrorCode.INVALID_LOCATION,
                    "Coordinates out of range.",
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                )
            if ts > now + timedelta(minutes=5) or ts < now - MAX_BACKDATE:
                raise AppError(
                    ErrorCode.VALIDATION_ERROR,
                    "Location timestamps must be within the last 7 days and not in the future.",
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                )
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "patient_id": profile.id,
                    "timestamp": ts,
                    "latitude": f.latitude,
                    "longitude": f.longitude,
                    "accuracy_m": f.accuracy_m,
                    "activity_context": f.activity_context.value,
                    "source": f.source,
                }
            )
        stmt = (
            insert(LocationRecord)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_locations_patient_timestamp")
        )
        stored = len(self.db.execute(stmt.returning(LocationRecord.id)).all())
        # Count only: location values never go into the audit log.
        self.audit.write(
            "locations_recorded",
            user_id=user.id,
            resource_type="patient",
            resource_id=profile.id,
            details={"received": len(rows), "stored": stored},
        )
        self.db.commit()
        return stored

    def _locations(
        self, patient_id: uuid.UUID, start: datetime, end: datetime
    ) -> list[LocationRecord]:
        q = (
            select(LocationRecord)
            .where(LocationRecord.patient_id == patient_id, LocationRecord.timestamp <= end)
            .where(LocationRecord.timestamp > start)
            .order_by(LocationRecord.timestamp)
        )
        return list(self.db.scalars(q))

    def get_current_valid_location(
        self, patient_id: uuid.UUID, at: datetime
    ) -> LocationRecord | None:
        """BR-07: the latest fix at or before `at`, only if within the freshness window."""
        max_age = timedelta(hours=get_settings().location_max_age_hours)
        fixes = self._locations(patient_id, at - max_age, at)
        return fixes[-1] if fixes else None

    # --- exposure -------------------------------------------------------------------------

    def exposure_profile(
        self, user: User, pollutant: Pollutant, at: datetime | None = None, hours: int = 24
    ) -> ExposureProfile:
        profile = self._patient_with_consent(user)
        if not 1 <= hours <= MAX_EXPOSURE_WINDOW_HOURS:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                f"hours must be between 1 and {MAX_EXPOSURE_WINDOW_HOURS}.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        settings = get_settings()
        end = (at or datetime.now(UTC)).astimezone(UTC)
        loc_age = timedelta(hours=settings.location_max_age_hours)
        grid = _hour_grid(end, hours)
        fixes = self._locations(profile.id, grid[0] - loc_age, end)

        estimates = []
        idx = -1
        for hour in grid:
            while idx + 1 < len(fixes) and fixes[idx + 1].timestamp <= hour:
                idx += 1
            fix = fixes[idx] if idx >= 0 and hour - fixes[idx].timestamp <= loc_age else None
            if fix is None:
                estimates.append(
                    estimate_hourly_exposure(hour, pollutant, None, None, location_available=False)
                )
                continue
            local = estimate_local_air_quality(
                GeoPoint(fix.latitude, fix.longitude),
                self.observations.latest_readings(pollutant, hour),
                hour,
                radius_km=settings.spatial_radius_km,
                max_age=timedelta(hours=settings.aq_max_age_hours),
                exact_match_km=settings.spatial_exact_match_km,
                max_stations=settings.spatial_max_stations,
            )
            ambient = local.concentration if isinstance(local, LocalEstimate) else None
            estimates.append(
                estimate_hourly_exposure(
                    hour, pollutant, ambient, MicroEnvironment(fix.activity_context)
                )
            )
        return build_exposure_profile(pollutant, estimates, grid[-1])
