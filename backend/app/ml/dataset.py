"""Read stored observations into pandas for feature building and training."""

from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AirQualityObservation, Station, WeatherObservation


def load_observations(
    db: Session,
    start: datetime | None = None,
    end: datetime | None = None,
    station_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Reference-monitor observations as a long frame, ordered by station and time."""
    o = AirQualityObservation
    q = (
        select(o.station_id, o.pollutant, o.timestamp, o.concentration, o.quality_flag)
        .join(Station, Station.id == o.station_id)
        .where(Station.is_reference_monitor.is_(True))
        .order_by(o.station_id, o.pollutant, o.timestamp)
    )
    if start is not None:
        q = q.where(o.timestamp >= start)
    if end is not None:
        q = q.where(o.timestamp <= end)
    if station_ids:
        q = q.where(o.station_id.in_(station_ids))
    df = pd.DataFrame(
        db.execute(q).all(),
        columns=["station_id", "pollutant", "timestamp", "concentration", "quality_flag"],
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["pollutant"] = df["pollutant"].astype(str)
        df["quality_flag"] = df["quality_flag"].astype(str)
    return df


def load_weather(
    db: Session,
    start: datetime | None = None,
    end: datetime | None = None,
    kind: str = "reanalysis",
) -> pd.DataFrame:
    """Hourly weather indexed by UTC timestamp. Only `reanalysis` is used for training."""
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
    q = select(*(getattr(w, c) for c in cols)).where(w.kind == kind).order_by(w.timestamp)
    if start is not None:
        q = q.where(w.timestamp >= start)
    if end is not None:
        q = q.where(w.timestamp <= end)
    df = pd.DataFrame(db.execute(q).all(), columns=cols)
    if df.empty:
        return df.set_index(pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def station_coordinates(db: Session) -> dict[str, tuple[float, float]]:
    rows = db.execute(select(Station.id, Station.latitude, Station.longitude)).all()
    return {sid: (lat, lon) for sid, lat, lon in rows}
