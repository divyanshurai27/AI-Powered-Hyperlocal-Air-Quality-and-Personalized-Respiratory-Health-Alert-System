"""Personal respiratory-risk estimate and 24 h rolling risk forecast (PRD §15, §24).

Pipeline per request, all as of the origin T (the AQ hour at or before `at`):

    location (latest fresh fix; else the saved home) ─┐
    exposure: last 96 h of estimated PM2.5/PM10/O3 ───┤
    weather ≤ T, health records ≤ T, profile ─────────┴─► risk features (one definition,
    app/ml/risk_features.py) → shared-trunk model → per-disease calibration → personal
    offset (optional, bounded) → versioned policy → Low / Moderate / High / Severe

Rolling forecast (BR-03): for h = 1..24 the same model is applied at T+h, with exposure
after T PROJECTED from the 24 h AQ forecast at the patient's current place and current
microenvironment. Health and weather features are held at their values at T (nothing
after T is known), so a horizon answers "if you stay where you are and nothing else
changes, what would your 24 h risk look like at T+h". Horizons the AQ forecast does not
cover are returned as unavailable, never filled.

The model was trained on a SYNTHETIC cohort (app/ml/cohort.py); outputs are research
estimates, never medical advice or medication guidance (BR-12).
"""

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import status
from sqlalchemy.orm import Session

from app.core.config import REPO_ROOT, get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.domain.enums import DiseaseType, RiskLevel
from app.domain.exposure import MIN_WINDOW_COVERAGE, MicroEnvironment, microenvironment_factor
from app.domain.pollutants import Pollutant
from app.domain.risk_policy import (
    PERSONALIZATION_VERSION,
    RISK_POLICY_VERSION,
    RISK_THRESHOLDS,
    PersonalBaseline,
    apply_personal_offset,
    classify_risk,
    estimate_personal_baseline,
)
from app.domain.spatial import GeoPoint, Unavailable
from app.ml.calibration import DiseaseCalibrators
from app.ml.risk import (
    RiskArtifactError,
    RiskArtifactMetadata,
    SharedHeadRiskModel,
    load_risk_artifact,
)
from app.ml.risk_features import (
    HEALTH_FEATURES,
    RISK_FEATURE_NAMES,
    WEATHER_FEATURES,
    build_risk_features,
    personal_history,
)
from app.models import PatientProfile, User
from app.repositories.audit import AuditRepository
from app.repositories.health import RiskPredictionRepository
from app.services.exposure import ExposureService, hour_grid
from app.services.forecast import ForecastService
from app.services.health import HealthService
from app.services.patients import PatientService

logger = get_logger(__name__)

RISK_TARGET = "respiratory event starting within the next 24 h"
RISK_POLLUTANTS = (Pollutant.PM25, Pollutant.PM10, Pollutant.O3)
EXPOSURE_HISTORY_HOURS = 96  # 72 h window + the 24 h change feature's 30 h reach, with slack
HORIZONS = range(1, 25)


@lru_cache(maxsize=2)
def get_risk_model() -> tuple[SharedHeadRiskModel, DiseaseCalibrators, RiskArtifactMetadata]:
    s = get_settings()
    directory = REPO_ROOT / Path(s.risk_artifact_dir) / s.risk_model_version
    if not (directory / "metadata.json").exists():
        raise AppError(
            ErrorCode.MODEL_NOT_AVAILABLE,
            f"No risk model (version {s.risk_model_version}).",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        return load_risk_artifact(directory)
    except RiskArtifactError as exc:
        logger.error("risk_artifact_rejected", extra={"error": str(exc)})
        raise AppError(
            ErrorCode.FEATURE_CONTRACT_MISMATCH,
            "The risk model does not match the current feature contract.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        ) from exc


@dataclass
class RiskEstimate:
    horizon_hours: int
    prediction_time: datetime
    target_window_start: datetime
    target_window_end: datetime
    raw_probability: float
    risk_probability: float
    risk_level: RiskLevel
    confidence: str
    data_quality: dict[str, Any]
    features: dict[str, float | None]


@dataclass
class HorizonResult:
    horizon_hours: int
    time: datetime
    estimate: RiskEstimate | None
    unavailable_reason: str | None = None


@dataclass
class RiskResult:
    disease_type: str
    severity: str | None
    origin: datetime
    model_version: str
    calibration_version: str
    feature_version: str
    policy_version: str
    personalization: str
    baseline: PersonalBaseline | None
    location_source: str
    current: RiskEstimate
    horizons: list[HorizonResult] = field(default_factory=list)
    forecast_versions: dict[str, str] = field(default_factory=dict)
    pollutants_skipped: dict[str, str] = field(default_factory=dict)

    @property
    def thresholds(self) -> dict[str, float]:
        moderate, high, severe = RISK_THRESHOLDS[DiseaseType(self.disease_type)]
        return {"Moderate": moderate, "High": high, "Severe": severe}


def _clean(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 5)


@dataclass
class _Context:
    profile: PatientProfile
    origin: datetime
    point: GeoPoint | None
    environment: MicroEnvironment
    location_source: str
    exposure: pd.DataFrame
    weather: pd.DataFrame
    health: pd.DataFrame


class RiskService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.exposure = ExposureService(db)
        self.predictions = RiskPredictionRepository(db)
        self.audit = AuditRepository(db)

    # --- context -------------------------------------------------------------------------

    def _profile(self, user: User) -> PatientProfile:
        profile = PatientService(self.db).require_consent(user)
        missing = [f for f in ("disease_type", "severity") if getattr(profile, f) is None]
        if missing:
            raise AppError(
                ErrorCode.PROFILE_INCOMPLETE,
                f"Set {', '.join(missing)} in your profile to get a personal risk estimate.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        return profile

    def _origin(self, at: datetime | None) -> datetime:
        now = datetime.now(UTC)
        query_at = (at or now).astimezone(UTC)
        if query_at > now + timedelta(minutes=1):
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "`at` cannot be in the future.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        return hour_grid(query_at, 1)[0]

    def _context(self, user: User, at: datetime | None) -> _Context:
        profile = self._profile(user)
        origin = self._origin(at)
        grid = hour_grid(origin, EXPOSURE_HISTORY_HOURS)

        fix = self.exposure.get_current_valid_location(profile.id, origin)
        any_fix = self.exposure.has_locations(profile.id, grid[0], origin)
        home = profile.home
        home_point = GeoPoint(home["latitude"], home["longitude"]) if home else None
        if fix is not None:
            point, env, source = (
                GeoPoint(fix.latitude, fix.longitude),
                MicroEnvironment(fix.activity_context),
                "device",
            )
        elif home_point is not None:
            point, env, source = home_point, MicroEnvironment.UNKNOWN, "saved_home"
        else:
            point, env, source = None, MicroEnvironment.UNKNOWN, "none"
        # A saved home stands in for the whole history only when the device never reported;
        # otherwise unlogged hours stay missing, exactly as in the training data.
        fallback = home_point if not any_fix else None

        exposure = pd.DataFrame(
            {
                p.value: [
                    h.exposure if h.exposure is not None else np.nan
                    for h in self.exposure.hourly_estimates(profile.id, p, grid, fallback=fallback)
                ]
                for p in RISK_POLLUTANTS
            },
            index=pd.DatetimeIndex(grid),
            dtype=float,
        )
        weather = ForecastService(self.db).weather_frame(origin - timedelta(hours=30), origin)
        health = HealthService(self.db).history_frame(profile.id, origin)
        return _Context(profile, origin, point, env, source, exposure, weather, health)

    # --- prediction ----------------------------------------------------------------------

    def _baseline(self, ctx: _Context, meta: RiskArtifactMetadata) -> PersonalBaseline | None:
        rate = meta.stratum_daily_rates.get(f"{ctx.profile.disease_type}|{ctx.profile.severity}")
        if rate is None:
            return None
        events, diary_days = personal_history(ctx.health, pd.Timestamp(ctx.origin))
        return estimate_personal_baseline(events, diary_days, rate)

    @staticmethod
    def _quality(row: pd.Series, location_source: str, projected: bool) -> tuple[str, dict]:
        reasons = []
        if not row["exp_pm25_coverage_24h"] >= MIN_WINDOW_COVERAGE:
            reasons.append("incomplete_recent_exposure")
        if not row["hx_diary_days_7d"] > 0:
            reasons.append("no_symptom_diary_in_7_days")
        if location_source == "none":
            reasons.append("no_location")
        low = bool(reasons)
        if math.isnan(row["wx_temperature"]):
            reasons.append("weather_unavailable")
        if location_source == "saved_home":
            reasons.append("location_from_saved_home")
        if projected:
            reasons.append("exposure_projected_from_aq_forecast")
        return ("low" if low else "normal"), {
            "reasons": reasons,
            "location_source": location_source,
            "pm25_exposure_coverage_24h": _clean(row["exp_pm25_coverage_24h"]),
            "diary_days_7d": _clean(row["hx_diary_days_7d"]),
        }

    def _estimate(
        self,
        X: pd.DataFrame,
        ctx: _Context,
        baseline: PersonalBaseline | None,
        projected: bool,
    ) -> list[RiskEstimate]:
        model, calibrators, _ = get_risk_model()
        disease = np.full(len(X), str(ctx.profile.disease_type))
        raw = model.predict_proba(X[RISK_FEATURE_NAMES], disease)
        calibrated = calibrators.calibrate(raw, disease)
        out = []
        for i, (t, row) in enumerate(X.iterrows()):
            p = apply_personal_offset(float(calibrated[i]), baseline)
            confidence, quality = self._quality(row, ctx.location_source, projected)
            horizon = int(round((t - pd.Timestamp(ctx.origin)).total_seconds() / 3600))
            out.append(
                RiskEstimate(
                    horizon_hours=horizon,
                    prediction_time=ctx.origin,
                    target_window_start=t.to_pydatetime(),
                    target_window_end=(t + pd.Timedelta(hours=24)).to_pydatetime(),
                    raw_probability=float(np.clip(raw[i], 0, 1)),
                    risk_probability=p,
                    risk_level=classify_risk(p, str(ctx.profile.disease_type)),
                    confidence=confidence,
                    data_quality=quality,
                    features={k: _clean(v) for k, v in row[RISK_FEATURE_NAMES].items()},
                )
            )
        return out

    def _projected_horizons(
        self, ctx: _Context, base_row: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[int], dict[str, str], dict[str, str], dict[int, str]]:
        """Feature rows for T+1..T+24 with exposure projected from the AQ forecast."""
        forecasts: dict[str, dict[datetime, float]] = {}
        versions: dict[str, str] = {}
        skipped: dict[str, str] = {}
        service = ForecastService(self.db)
        for p in RISK_POLLUTANTS:
            try:
                f = service.predict_24h(ctx.point.latitude, ctx.point.longitude, p, ctx.origin)
            except AppError as exc:
                if exc.code == ErrorCode.INVALID_LOCATION:
                    raise
                skipped[p.value] = exc.code.value
                continue
            if isinstance(f, Unavailable):
                skipped[p.value] = f.reason.value
                continue
            forecasts[p.value] = {t: v for _, t, v in f.horizons}
            versions[p.value] = f.model_version

        times = [ctx.origin + timedelta(hours=h) for h in HORIZONS]
        missing: dict[int, str] = {}
        if "pm25" not in forecasts:
            reason = f"pm25_forecast_{skipped.get('pm25', 'unavailable').lower()}"
            return pd.DataFrame(), [], versions, skipped, dict.fromkeys(HORIZONS, reason)

        future = pd.DataFrame(index=pd.DatetimeIndex(times), columns=ctx.exposure.columns)
        for p in RISK_POLLUTANTS:
            factor = microenvironment_factor(ctx.environment, p)
            by_time = forecasts.get(p.value, {})
            future[p.value] = [by_time.get(t, np.nan) * factor for t in times]
        exposure = pd.concat([ctx.exposure, future.astype(float)])
        available = [h for h, t in zip(HORIZONS, times, strict=True) if t in forecasts["pm25"]]
        missing = {h: "beyond_aq_forecast" for h in HORIZONS if h not in available}
        if not available:
            return pd.DataFrame(), [], versions, skipped, missing
        origins = pd.DatetimeIndex([ctx.origin + timedelta(hours=h) for h in available])
        X = build_risk_features(
            self._profile_dict(ctx.profile), exposure, ctx.weather, ctx.health, origins
        )
        # Nothing after T is known: hold health and weather at their values at T.
        for col in HEALTH_FEATURES + WEATHER_FEATURES:
            X[col] = base_row[col].iloc[0]
        return X, available, versions, skipped, missing

    @staticmethod
    def _profile_dict(profile: PatientProfile) -> dict:
        return {
            "age": profile.age,
            "sex": profile.sex,
            "disease_type": profile.disease_type,
            "severity": profile.severity,
        }

    # --- public ---------------------------------------------------------------------------

    def assess(
        self,
        user: User,
        at: datetime | None = None,
        *,
        forecast: bool = False,
        personalize: bool = True,
    ) -> RiskResult:
        _, calibrators, meta = get_risk_model()  # fail fast with MODEL_NOT_AVAILABLE
        ctx = self._context(user, at)
        if forecast and ctx.point is None:
            raise AppError(
                ErrorCode.LOCATION_NOT_SET,
                "No recent location and no saved home: set your home in your profile.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        baseline = self._baseline(ctx, meta) if personalize else None
        X0 = build_risk_features(
            self._profile_dict(ctx.profile),
            ctx.exposure,
            ctx.weather,
            ctx.health,
            pd.DatetimeIndex([ctx.origin]),
        )
        current = self._estimate(X0, ctx, baseline, projected=False)[0]
        result = RiskResult(
            disease_type=str(ctx.profile.disease_type),
            severity=ctx.profile.severity,
            origin=ctx.origin,
            model_version=meta.model_version,
            calibration_version=calibrators.version,
            feature_version=meta.feature_version,
            policy_version=RISK_POLICY_VERSION,
            personalization=PERSONALIZATION_VERSION if baseline else "none",
            baseline=baseline,
            location_source=ctx.location_source,
            current=current,
        )
        if forecast:
            X, available, versions, skipped, missing = self._projected_horizons(ctx, X0)
            estimates = (
                {e.horizon_hours: e for e in self._estimate(X, ctx, baseline, projected=True)}
                if available
                else {}
            )
            result.forecast_versions, result.pollutants_skipped = versions, skipped
            result.horizons = [
                HorizonResult(h, ctx.origin + timedelta(hours=h), estimates.get(h), missing.get(h))
                for h in HORIZONS
            ]
        self._save(user, ctx.profile.id, result)
        return result

    def _save(self, user: User, patient_id: uuid.UUID, result: RiskResult) -> None:
        estimates = [result.current] + [h.estimate for h in result.horizons if h.estimate]
        rows = [
            {
                "id": uuid.uuid4(),
                "patient_id": patient_id,
                "prediction_time": e.prediction_time,
                "horizon_hours": e.horizon_hours,
                "target_window_start": e.target_window_start,
                "target_window_end": e.target_window_end,
                "raw_probability": e.raw_probability,
                "risk_probability": e.risk_probability,
                "risk_level": e.risk_level.value,
                "disease_type": result.disease_type,
                "model_version": result.model_version,
                "calibration_version": result.calibration_version,
                "policy_version": result.policy_version,
                "feature_version": result.feature_version,
                "personalization": result.personalization,
                "personal_offset": result.baseline.offset if result.baseline else None,
                "confidence": e.confidence,
                "data_quality": e.data_quality,
                "features": e.features,
            }
            for e in estimates
        ]
        self.predictions.upsert_many(rows)
        # Levels and counts only; probabilities and features stay in the patient-scoped table.
        self.audit.write(
            "risk_assessed",
            user_id=user.id,
            resource_type="patient",
            resource_id=patient_id,
            details={"predictions": len(rows), "model_version": result.model_version},
        )
        self.db.commit()

    def history(
        self,
        user: User,
        start: datetime | None = None,
        end: datetime | None = None,
        horizon_hours: int | None = 0,
        limit: int = 200,
    ):
        profile = PatientService(self.db).require_consent(user)
        end = (end or datetime.now(UTC)).astimezone(UTC)
        start = (start or end - timedelta(days=30)).astimezone(UTC)
        return self.predictions.history(profile.id, start, end, horizon_hours, limit)
