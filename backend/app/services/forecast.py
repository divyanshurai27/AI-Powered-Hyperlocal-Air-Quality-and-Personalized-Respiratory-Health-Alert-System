"""24 h AQ forecast for a location (PRD §12, §24 `forecast_service.predict_24h`).

Per usable nearby station: build the feature row at the station's latest observed hour
(origin), run the frozen artifact, then combine stations with the same inverse-distance
weights used for current AQ.

Weather at inference: training used ERA5 reanalysis at t, which is published ~5 days late.
Live requests therefore fall back to the latest forecast issued at or before t for hour t.
That is a documented train/serve difference; a historical `at` uses reanalysis like training.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import REPO_ROOT, get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.domain.observations import valid_coordinates
from app.domain.pollutants import Pollutant
from app.domain.spatial import GeoPoint, LocalEstimate, Unavailable, estimate_local_air_quality
from app.ml.dataset import load_observations, station_coordinates
from app.ml.features import (
    HORIZONS,
    LAGS_HOURS,
    build_station_frames,
    feature_names,
    generate_features,
)
from app.ml.forecasting import (
    ArtifactContractError,
    DirectGBMForecaster,
    ForecastArtifactMetadata,
    load_model_artifact,
)
from app.models import WeatherObservation
from app.repositories.environment import ObservationRepository

logger = get_logger(__name__)

HISTORY_HOURS = max(LAGS_HOURS) + 26  # longest lag + rolling window slack


@dataclass(frozen=True)
class StationForecast:
    station_id: str
    origin: datetime
    values: list[float]  # index i → horizon i+1


@dataclass(frozen=True)
class LocationForecast:
    pollutant: Pollutant
    origin: datetime
    model_version: str
    feature_version: str
    method: str
    weights: dict[str, float]
    horizons: list[tuple[int, datetime, float]]  # (h, target_time, µg/m³)
    stations: list[StationForecast]


@lru_cache(maxsize=8)
def get_forecast_model(
    pollutant: Pollutant,
) -> tuple[DirectGBMForecaster, ForecastArtifactMetadata]:
    """Load once per process (PRD §46: keep inference models loaded) and verify the contract."""
    s = get_settings()
    directory = (
        REPO_ROOT / Path(s.aq_forecast_artifact_dir) / pollutant.value / s.aq_forecast_model_version
    )
    if not (directory / "metadata.json").exists():
        raise AppError(
            ErrorCode.MODEL_NOT_AVAILABLE,
            f"No forecast model for {pollutant.value} (version {s.aq_forecast_model_version}).",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        return load_model_artifact(directory, feature_names(pollutant))
    except ArtifactContractError as exc:
        logger.error("forecast_artifact_rejected", extra={"error": str(exc)})
        raise AppError(
            ErrorCode.FEATURE_CONTRACT_MISMATCH,
            "The forecast model does not match the current feature contract.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        ) from exc


class ForecastService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.observations = ObservationRepository(db)

    def _weather(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Reanalysis where it exists; for later hours, the latest forecast issued ≤ that hour."""
        w = WeatherObservation
        cols = [
            "timestamp",
            "temperature",
            "relative_humidity",
            "precipitation",
            "wind_speed",
            "wind_direction",
            "pressure",
        ]
        rows = self.db.execute(
            select(*(getattr(w, c) for c in cols), w.kind, w.issued_at)
            .where(w.timestamp > start, w.timestamp <= end)
            .where(
                (w.kind == "reanalysis") | ((w.kind == "forecast") & (w.issued_at <= w.timestamp))
            )
            .order_by(w.timestamp)
        ).all()
        df = pd.DataFrame(rows, columns=[*cols, "kind", "issued_at"])
        if df.empty:
            return pd.DataFrame(columns=cols[1:], index=pd.DatetimeIndex([], tz="UTC"))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["rank"] = (df["kind"] == "reanalysis").astype(int)
        df = df.sort_values(["timestamp", "rank", "issued_at"]).drop_duplicates(
            "timestamp", keep="last"
        )
        return df.set_index("timestamp")[cols[1:]]

    def station_forecast(
        self, station_id: str, pollutant: Pollutant, at: datetime
    ) -> StationForecast | None:
        """None when the station can't be forecast honestly (no recent observed value)."""
        model, _ = get_forecast_model(pollutant)
        start = at - timedelta(hours=HISTORY_HOURS)
        obs = load_observations(self.db, start=start, end=at, station_ids=[station_id])
        frames = build_station_frames(obs)
        if station_id not in frames or pollutant.value not in frames[station_id]:
            return None
        frame = frames[station_id]
        observed = frame[f"{pollutant.value}__observed"]
        if not observed.any():
            return None
        origin = observed[observed].index.max()
        if at - origin.to_pydatetime() > timedelta(hours=get_settings().aq_max_age_hours):
            return None
        lat, lon = station_coordinates(self.db)[station_id]
        weather = self._weather(start - timedelta(hours=26), origin.to_pydatetime())
        row = generate_features(frame.loc[:origin], weather, pollutant, lat, lon).loc[[origin]]
        pred = model.predict(row).iloc[0].to_numpy(dtype=float)
        # PRD §12 acceptance gate.
        if pred.shape != (len(HORIZONS),) or not np.isfinite(pred).all():
            logger.error("forecast_rejected", extra={"station": station_id, "shape": pred.shape})
            return None
        return StationForecast(
            station_id, origin.to_pydatetime(), [round(float(v), 2) for v in pred]
        )

    def predict_24h(
        self, latitude: float, longitude: float, pollutant: Pollutant, at: datetime | None = None
    ) -> LocationForecast | Unavailable:
        if not valid_coordinates(latitude, longitude):
            raise AppError(
                ErrorCode.INVALID_LOCATION,
                "Coordinates out of range.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        s = get_settings()
        now = datetime.now(UTC)
        query_at = (at or now).astimezone(UTC)
        if query_at > now + timedelta(minutes=1):
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "`at` cannot be in the future.",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        _, meta = get_forecast_model(pollutant)  # fail fast with MODEL_NOT_AVAILABLE

        current = estimate_local_air_quality(
            GeoPoint(latitude, longitude),
            self.observations.latest_readings(pollutant, query_at),
            query_at,
            radius_km=s.spatial_radius_km,
            max_age=timedelta(hours=s.aq_max_age_hours),
            exact_match_km=s.spatial_exact_match_km,
            max_stations=s.spatial_max_stations,
        )
        if isinstance(current, Unavailable):
            return current

        forecasts = []
        for c in current.contributions:
            f = self.station_forecast(c.station_id, pollutant, query_at)
            if f is not None:
                forecasts.append((c.weight, f))
        if not forecasts:
            raise AppError(
                ErrorCode.PREDICTION_FAILED,
                "No nearby station has enough recent data to forecast from.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        # Horizons only line up across stations that share an origin; keep the most recent one.
        origin = max(f.origin for _, f in forecasts)
        forecasts = [(w, f) for w, f in forecasts if f.origin == origin]
        total = sum(w for w, _ in forecasts)
        weights = {f.station_id: w / total for w, f in forecasts}
        horizons = [
            (
                h,
                origin + timedelta(hours=h),
                round(sum(weights[f.station_id] * f.values[h - 1] for _, f in forecasts), 2),
            )
            for h in HORIZONS
        ]
        return LocationForecast(
            pollutant=pollutant,
            origin=origin,
            model_version=meta.model_version,
            feature_version=meta.feature_version,
            method=current.method if isinstance(current, LocalEstimate) else "idw",
            weights={k: round(v, 4) for k, v in weights.items()},
            horizons=horizons,
            stations=[f for _, f in forecasts],
        )
