# ruff: noqa: F811  (pytest fixtures imported from another module)
from datetime import timedelta

from app.services import city_map
from tests.integration.test_exposure_and_forecast_api import (  # noqa: F401
    BTM,
    T,
    forecast_artifact,
    seed_obs,
    seed_recent_history,
    seed_station,
)

ME = "/api/v1/patients/me"
TODAY = "/api/v1/recommendations/today"
MAP = "/api/v1/air/map"
HOME = {"label": "Home", "latitude": BTM[0], "longitude": BTM[1]}


# --- saved places ------------------------------------------------------------------------


def test_save_and_clear_home_and_work(client, register_and_login) -> None:
    s = register_and_login()
    body = client.patch(
        ME,
        headers=s["headers"],
        json={"home": HOME, "work": {"label": "Office", "latitude": 12.97, "longitude": 77.60}},
    ).json()
    assert body["home"] == HOME and body["work"]["label"] == "Office"
    cleared = client.patch(ME, headers=s["headers"], json={"work": None}).json()
    assert cleared["work"] is None and cleared["home"] == HOME


def test_saved_place_validation(client, register_and_login) -> None:
    s = register_and_login()
    r = client.patch(
        ME, headers=s["headers"], json={"home": {"label": "x", "latitude": 95, "longitude": 77}}
    )
    assert r.status_code == 422


def test_saved_places_are_private(client, register_and_login) -> None:
    a, b = register_and_login("a@example.com"), register_and_login("b@example.com")
    client.patch(ME, headers=a["headers"], json={"home": HOME})
    assert client.get(ME, headers=b["headers"]).json()["home"] is None


# --- recommendations ---------------------------------------------------------------------


def test_today_needs_a_saved_place(client, register_and_login) -> None:
    s = register_and_login()
    r = client.get(TODAY, headers=s["headers"])
    assert r.status_code == 422 and r.json()["detail"]["code"] == "LOCATION_NOT_SET"


def test_today_guidance_for_home(client, register_and_login, db, forecast_artifact) -> None:
    seed_recent_history(db)
    s = register_and_login()
    client.patch(
        ME,
        headers=s["headers"],
        json={"home": HOME, "disease_type": "asthma", "severity": "moderate"},
    )
    r = client.get(TODAY, headers=s["headers"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["available"] is True and d["kind"] == "air-quality guidance"
    assert d["level"] in {"good", "caution", "limit", "avoid"}
    assert len(d["hours"]) == 24
    assert d["best_window"] is not None
    assert "asthma" in d["personalisation_note"]
    assert d["model_versions"] == {"pm25": "aq_pm25_vtest"}
    # PM10/O3 have no model in this test setup: skipped explicitly, not faked.
    assert set(d["pollutants_skipped"]) == {"pm10", "o3"}
    assert "not medical advice" in d["disclaimer"]
    text = (d["headline"] + d["health_note"]).lower()
    assert "safe" not in text and "inhaler" not in text and "medication" not in text


def test_today_is_unavailable_when_air_data_is_stale(
    client, register_and_login, forecast_artifact
) -> None:
    s = register_and_login()
    client.patch(ME, headers=s["headers"], json={"home": HOME})
    d = client.get(TODAY, headers=s["headers"]).json()
    assert d["available"] is False and d["hours"] == []
    assert d["unavailable_reason"]


# --- city map ----------------------------------------------------------------------------


def test_city_map_grid(client, register_and_login, db, forecast_artifact) -> None:
    city_map._cache.clear()
    seed_recent_history(db)
    # A second station ~30 km away with only a stale reading: it widens the grid, but the
    # cells around it have no fresh data and must stay empty.
    far = (13.20, 77.60)
    seed_station(db, "openaq:9999", *far)
    seed_obs(db, "openaq:9999", {T - timedelta(days=2): 40.0}, lat=far[0], lon=far[1])
    db.commit()
    s = register_and_login()
    d = client.get(MAP, headers=s["headers"]).json()
    cells = d["cells"]
    with_data = [c for c in cells if c["now"] is not None]
    assert with_data, "cells near the station should have values"
    assert all(len(c["forecast"]) == 24 for c in with_data)
    assert d["model_version"] == "aq_pm25_vtest"
    # Cells far from the only station are gaps, not extrapolated values.
    assert any(c["now"] is None and c["forecast"] is None for c in cells)


def test_city_map_requires_auth(client) -> None:
    assert client.get(MAP).status_code == 401
