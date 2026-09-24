"""IT-03 (forecast → exposure mapping) and AT-03 (24 h forecast or explicit unavailable)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.core.config import get_settings
from app.domain.pollutants import Pollutant
from app.ml.features import build_forecasting_matrix, build_station_frames, feature_names
from app.ml.forecasting import (
    DirectGBMForecaster,
    ForecastArtifactMetadata,
    GBMConfig,
    save_model_artifact,
)
from app.models import AirQualityObservation, Station, WeatherObservation
from app.services import forecast as forecast_module
from tests.unit.test_forecast_features import synthetic_obs, synthetic_weather

LOC = "/api/v1/location"
EXP_CURRENT = "/api/v1/exposure/current"
EXP_HISTORY = "/api/v1/exposure/history"
FORECAST = "/api/v1/air/forecast"

NOW = datetime.now(UTC)
T = NOW.replace(minute=30, second=0, microsecond=0) - timedelta(hours=1)  # a recent AQ hour end
BTM = (12.9135218, 77.5950804)


def consent(client, s) -> None:
    r = client.patch(
        "/api/v1/patients/me", headers=s["headers"], json={"consent_status": "granted"}
    )
    assert r.status_code == 200


def seed_station(db, sid="openaq:5548", lat=BTM[0], lon=BTM[1]) -> None:
    db.add(
        Station(
            id=sid,
            source="openaq",
            source_station_id=sid.split(":")[1],
            name=sid,
            latitude=lat,
            longitude=lon,
            provider="CPCB",
            is_reference_monitor=True,
        )
    )
    db.flush()


def seed_obs(
    db, sid, values_by_hour: dict[datetime, float], pollutant="pm25", lat=BTM[0], lon=BTM[1]
) -> None:
    for ts, v in values_by_hour.items():
        db.add(
            AirQualityObservation(
                source="openaq",
                station_id=sid,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                pollutant=pollutant,
                concentration=v,
                unit="µg/m³",
                raw_value=v,
                raw_unit="µg/m³",
                coverage_pct=100.0,
                quality_flag="valid",
            )
        )


# --- location intake ---------------------------------------------------------------------


def test_location_upload_requires_consent(client, register_and_login) -> None:
    s = register_and_login()
    r = client.post(
        LOC,
        headers=s["headers"],
        json={"fixes": [{"timestamp": NOW.isoformat(), "latitude": 12.97, "longitude": 77.59}]},
    )
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "CONSENT_REQUIRED"


def test_location_upload_is_idempotent(client, register_and_login) -> None:
    s = register_and_login()
    consent(client, s)
    body = {
        "fixes": [
            {
                "timestamp": (NOW - timedelta(minutes=m)).isoformat(),
                "latitude": 12.97,
                "longitude": 77.59,
            }
            for m in (0, 10)
        ]
    }
    assert client.post(LOC, headers=s["headers"], json=body).json() == {"received": 2, "stored": 2}
    assert client.post(LOC, headers=s["headers"], json=body).json() == {"received": 2, "stored": 0}


@pytest.mark.parametrize(
    "fix",
    [
        {"latitude": 91, "longitude": 77.59},
        {"latitude": 12.97, "longitude": 181},
        {"latitude": 12.97, "longitude": 77.59, "timestamp_offset_h": 2},  # future
        {"latitude": 12.97, "longitude": 77.59, "timestamp_offset_h": -24 * 8},  # too old
        {"latitude": 12.97, "longitude": 77.59, "activity_context": "on_the_moon"},
    ],
)
def test_invalid_location_fixes_rejected(client, register_and_login, fix) -> None:
    s = register_and_login()
    consent(client, s)
    offset = fix.pop("timestamp_offset_h", 0)
    fix["timestamp"] = (NOW + timedelta(hours=offset)).isoformat()
    r = client.post(LOC, headers=s["headers"], json={"fixes": [fix]})
    assert r.status_code == 422


# --- exposure (IT-03) --------------------------------------------------------------------


def test_exposure_maps_location_and_time_to_matching_air(client, register_and_login, db) -> None:
    seed_station(db)
    seed_obs(db, "openaq:5548", {T - timedelta(hours=i): 40.0 + i for i in range(6)})
    db.commit()
    s = register_and_login()
    consent(client, s)
    fixes = [
        {
            "timestamp": (T - timedelta(hours=3, minutes=5)).isoformat(),
            "latitude": BTM[0],
            "longitude": BTM[1],
            "activity_context": "outdoor",
        },
        {
            "timestamp": (T - timedelta(minutes=5)).isoformat(),
            "latitude": BTM[0],
            "longitude": BTM[1],
            "activity_context": "indoor_home",
        },
    ]
    assert client.post(LOC, headers=s["headers"], json={"fixes": fixes}).status_code == 201

    body = client.get(
        EXP_HISTORY, headers=s["headers"], params={"hours": 6, "at": T.isoformat()}
    ).json()
    hours = {h["hour"]: h for h in body["hours"]}
    now_h = hours[T.isoformat().replace("+00:00", "Z")]
    assert now_h["ambient"] == 40.0 and now_h["microenvironment"] == "indoor_home"
    assert now_h["exposure"] == pytest.approx(32.0)  # 40 × 0.8
    earlier = hours[(T - timedelta(hours=3)).isoformat().replace("+00:00", "Z")]
    assert earlier["ambient"] == 43.0 and earlier["exposure"] == 43.0  # outdoor
    assert body["is_estimate"] is True and "not a personal measurement" in body["note"]


def test_hours_without_recent_location_are_missing(client, register_and_login, db) -> None:
    seed_station(db)
    seed_obs(db, "openaq:5548", {T - timedelta(hours=i): 40.0 for i in range(12)})
    db.commit()
    s = register_and_login()
    consent(client, s)
    fix = {
        "timestamp": (T - timedelta(minutes=10)).isoformat(),
        "latitude": BTM[0],
        "longitude": BTM[1],
    }
    client.post(LOC, headers=s["headers"], json={"fixes": [fix]})
    body = client.get(
        EXP_HISTORY, headers=s["headers"], params={"hours": 12, "at": T.isoformat()}
    ).json()
    old = body["hours"][0]
    assert old["exposure"] is None and old["missing_reason"] == "location_unavailable"
    assert body["hours"][-1]["assumed_outdoor"] is True  # activity unknown
    w12 = next(w for w in body["windows"] if w["window_hours"] == 6)
    assert not w12["complete"]


def test_exposure_is_patient_isolated(client, register_and_login, db) -> None:
    seed_station(db)
    seed_obs(db, "openaq:5548", {T: 40.0})
    db.commit()
    a, b = register_and_login("a@example.com"), register_and_login("b@example.com")
    consent(client, a)
    consent(client, b)
    client.post(
        LOC,
        headers=a["headers"],
        json={
            "fixes": [
                {
                    "timestamp": (T - timedelta(minutes=5)).isoformat(),
                    "latitude": BTM[0],
                    "longitude": BTM[1],
                }
            ]
        },
    )
    b_view = client.get(EXP_CURRENT, headers=b["headers"], params={"at": T.isoformat()}).json()
    assert b_view["current"]["exposure"] is None  # B never sees A's location-derived exposure
    assert b_view["current"]["missing_reason"] == "location_unavailable"


def test_exposure_requires_consent(client, register_and_login) -> None:
    s = register_and_login()
    assert (
        client.get(EXP_CURRENT, headers=s["headers"]).json()["detail"]["code"] == "CONSENT_REQUIRED"
    )


# --- forecast (AT-03) --------------------------------------------------------------------


@pytest.fixture
def forecast_artifact(tmp_path, monkeypatch):
    """Train a tiny model on synthetic data and point the service at it."""
    obs = synthetic_obs(hours=24 * 15)
    frames = build_station_frames(obs)
    coords = {"s1": (12.91, 77.59), "s2": (13.03, 77.58)}
    X, Y, _, _ = build_forecasting_matrix(
        frames, synthetic_weather(24 * 16), Pollutant.PM25, coords
    )
    t = X.index.get_level_values("t")
    cut = t.min() + pd.Timedelta(days=10)
    model = DirectGBMForecaster(
        feature_names(Pollutant.PM25), GBMConfig(n_estimators=30, min_child_samples=5)
    )
    model.fit(X[t < cut], Y[t < cut], X[t >= cut], Y[t >= cut])
    meta = ForecastArtifactMetadata(
        model_name=model.name,
        model_version="aq_pm25_vtest",
        pollutant="pm25",
        feature_version="aq_features_v1.0.0",
        feature_names=feature_names(Pollutant.PM25),
        horizons=list(range(1, 25)),
        training_dataset="synthetic",
        training_start="",
        training_end="",
        validation_start="",
        validation_end="",
        test_start="",
        test_end="",
        hyperparameters={},
        validation_metrics={},
        test_metrics={},
    )
    save_model_artifact(model, meta, tmp_path / "pm25" / "test")
    monkeypatch.setattr(get_settings(), "aq_forecast_artifact_dir", str(tmp_path))
    monkeypatch.setattr(get_settings(), "aq_forecast_model_version", "test")
    forecast_module.get_forecast_model.cache_clear()
    yield
    forecast_module.get_forecast_model.cache_clear()


def seed_recent_history(db, hours=80) -> None:
    seed_station(db)
    rng = np.random.default_rng(1)
    seed_obs(
        db,
        "openaq:5548",
        {
            T - timedelta(hours=i): 35 + 10 * np.sin(i / 24 * 2 * np.pi) + rng.normal(0, 1)
            for i in range(hours)
        },
    )
    for i in range(hours + 30):
        ts = T.replace(minute=0) - timedelta(hours=i)
        db.add(
            WeatherObservation(
                source="open-meteo",
                kind="reanalysis",
                latitude=12.97,
                longitude=77.59,
                timestamp=ts,
                temperature=24.0,
                relative_humidity=70.0,
                precipitation=0.0,
                wind_speed=2.0,
                wind_direction=200.0,
                pressure=912.0,
            )
        )
    db.commit()


def test_forecast_returns_24_hourly_finite_values(
    client, register_and_login, db, forecast_artifact
) -> None:
    seed_recent_history(db)
    s = register_and_login()
    r = client.get(FORECAST, headers=s["headers"], params={"lat": 12.97, "lon": 77.59})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["model_version"] == "aq_pm25_vtest"
    assert [h["horizon_hours"] for h in body["hours"]] == list(range(1, 25))
    origin = datetime.fromisoformat(body["origin"])
    assert origin == T
    for h in body["hours"]:
        assert datetime.fromisoformat(h["target_time"]) == origin + timedelta(
            hours=h["horizon_hours"]
        )
        assert np.isfinite(h["concentration"]) and h["concentration"] >= 0
    assert body["station_weights"] == {"openaq:5548": 1.0}


def test_forecast_unavailable_state_when_data_is_stale(
    client, register_and_login, db, forecast_artifact
) -> None:
    seed_station(db)
    seed_obs(db, "openaq:5548", {T - timedelta(days=3): 40.0})
    db.commit()
    s = register_and_login()
    body = client.get(FORECAST, headers=s["headers"], params={"lat": 12.97, "lon": 77.59}).json()
    assert body["available"] is False
    assert body["unavailable_reason"] == "no_fresh_data"
    assert body["hours"] == []


def test_missing_model_is_503_not_a_fake_forecast(
    client, register_and_login, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(get_settings(), "aq_forecast_artifact_dir", str(tmp_path / "nothing-here"))
    forecast_module.get_forecast_model.cache_clear()
    s = register_and_login()
    r = client.get(FORECAST, headers=s["headers"], params={"lat": 12.97, "lon": 77.59})
    forecast_module.get_forecast_model.cache_clear()
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "MODEL_NOT_AVAILABLE"
