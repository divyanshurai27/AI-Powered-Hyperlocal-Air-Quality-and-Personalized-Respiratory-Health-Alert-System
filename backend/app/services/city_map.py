"""City-wide grid of current and forecast concentrations for Bengaluru.

Each station is forecast once; every grid cell then reuses the same inverse-distance
weights as a point query (domain/spatial), so a cell's value equals what /air/current and
/air/forecast would return at that cell's centre. Cells with no usable station within the
matching radius are returned as unavailable, never extrapolated.
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError
from app.domain.pollutants import Pollutant
from app.domain.spatial import GeoPoint, LocalEstimate, estimate_local_air_quality, is_usable
from app.repositories.environment import ObservationRepository
from app.services.forecast import ForecastService, StationForecast, get_forecast_model

GRID_STEP_DEG = 0.02  # ≈ 2.2 km
GRID_PADDING_DEG = 0.06
CACHE_SECONDS = 600
_cache: dict[tuple, tuple[float, "CityGrid"]] = {}


@dataclass
class Cell:
    latitude: float
    longitude: float
    now: float | None
    forecast: list[float] | None  # index i → horizon i+1
    stations: int = 0


@dataclass
class CityGrid:
    pollutant: Pollutant
    at: datetime
    step_deg: float
    origin: datetime | None
    model_version: str | None
    cells: list[Cell] = field(default_factory=list)


class CityMapService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def grid(
        self, pollutant: Pollutant, at: datetime | None = None, with_forecast: bool = True
    ) -> CityGrid:
        now = datetime.now(UTC)
        query_at = (at or now).astimezone(UTC)
        bucket = query_at.replace(second=0, microsecond=0, minute=(query_at.minute // 5) * 5)
        key = (pollutant, bucket, with_forecast)
        hit = _cache.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        result = self._compute(pollutant, query_at, with_forecast)
        _cache[key] = (time.monotonic() + CACHE_SECONDS, result)
        return result

    def _compute(self, pollutant: Pollutant, at: datetime, with_forecast: bool) -> CityGrid:
        s = get_settings()
        max_age = timedelta(hours=s.aq_max_age_hours)
        readings = ObservationRepository(self.db).latest_readings(pollutant, at)

        forecasts: dict[str, StationForecast] = {}
        model_version = None
        if with_forecast:
            service = ForecastService(self.db)
            try:
                model_version = get_forecast_model(pollutant)[1].model_version
            except AppError:  # no model for this pollutant: serve current values only
                model_version = None
            if model_version:
                for r in readings:
                    if is_usable(r, at, max_age):
                        f = service.station_forecast(r.station_id, pollutant, at)
                        if f is not None:
                            forecasts[r.station_id] = f
        origin = max((f.origin for f in forecasts.values()), default=None)
        aligned = {k: f for k, f in forecasts.items() if f.origin == origin}

        grid = CityGrid(pollutant, at, GRID_STEP_DEG, origin, model_version)
        if not readings:
            return grid
        lats = [r.latitude for r in readings]
        lons = [r.longitude for r in readings]
        lat = min(lats) - GRID_PADDING_DEG
        while lat <= max(lats) + GRID_PADDING_DEG:
            lon = min(lons) - GRID_PADDING_DEG
            while lon <= max(lons) + GRID_PADDING_DEG:
                est = estimate_local_air_quality(
                    GeoPoint(lat, lon),
                    readings,
                    at,
                    radius_km=s.spatial_radius_km,
                    max_age=max_age,
                    exact_match_km=s.spatial_exact_match_km,
                    max_stations=s.spatial_max_stations,
                )
                cell = Cell(round(lat, 4), round(lon, 4), None, None)
                if isinstance(est, LocalEstimate):
                    cell.now = est.concentration
                    cell.stations = len(est.contributions)
                    ws = [
                        (c.weight, aligned[c.station_id])
                        for c in est.contributions
                        if c.station_id in aligned
                    ]
                    total = sum(w for w, _ in ws)
                    if total > 0:
                        cell.forecast = [
                            round(sum(w * f.values[h] for w, f in ws) / total, 1)
                            for h in range(len(ws[0][1].values))
                        ]
                grid.cells.append(cell)
                lon += GRID_STEP_DEG
            lat += GRID_STEP_DEG
        return grid
