from datetime import UTC, datetime, timedelta

import pytest

from app.models import AirQualityObservation, Station

CURRENT = "/api/v1/air/current"
AT = datetime(2026, 9, 20, 12, 30, tzinfo=UTC)

STATIONS = [
    ("openaq:5548", "BTM Layout", 12.9135218, 77.5950804),
    ("openaq:6984", "Hebbal", 13.029152, 77.585901),
]


@pytest.fixture
def seeded(db):
    for sid, name, lat, lon in STATIONS:
        db.add(
            Station(
                id=sid,
                source="openaq",
                source_station_id=sid.split(":")[1],
                name=name,
                latitude=lat,
                longitude=lon,
                provider="CPCB",
                is_reference_monitor=True,
            )
        )
    db.add(
        Station(
            id="openaq:9",
            source="openaq",
            source_station_id="9",
            name="low-cost",
            latitude=12.9716,
            longitude=77.5946,
            provider="AirGradient",
            is_reference_monitor=False,
        )
    )
    db.flush()

    def obs(sid, lat, lon, value, ts, pollutant="pm25", flag="valid"):
        db.add(
            AirQualityObservation(
                source="openaq",
                station_id=sid,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                pollutant=pollutant,
                concentration=value,
                unit="µg/m³",
                raw_value=value,
                raw_unit="µg/m³",
                coverage_pct=100.0,
                quality_flag=flag,
            )
        )

    obs("openaq:5548", 12.9135218, 77.5950804, 20.0, AT - timedelta(hours=1))
    obs(
        "openaq:5548", 12.9135218, 77.5950804, 99.0, AT + timedelta(hours=1)
    )  # future: must be ignored
    obs("openaq:6984", 13.029152, 77.585901, 40.0, AT - timedelta(hours=1))
    obs("openaq:9", 12.9716, 77.5946, 500.0, AT - timedelta(minutes=30))  # low-cost: ignored
    db.commit()


def pm25(body):
    return next(p for p in body["pollutants"] if p["pollutant"] == "pm25")


def test_requires_authentication(client) -> None:
    assert client.get(CURRENT, params={"lat": 12.97, "lon": 77.59}).status_code == 401


def test_exact_station_match(client, seeded, register_and_login) -> None:
    s = register_and_login()
    r = client.get(
        CURRENT,
        params={"lat": 12.9135218, "lon": 77.5950804, "at": AT.isoformat()},
        headers=s["headers"],
    )
    assert r.status_code == 200, r.text
    p = pm25(r.json())
    assert p["available"] and p["method"] == "exact_station"
    assert p["concentration"] == 20.0  # not the future 99.0
    assert p["data_age_minutes"] == 60.0
    assert "not a direct measurement" in r.json()["source_note"]


def test_between_stations_uses_idw_and_ignores_low_cost_sensor(
    client, seeded, register_and_login
) -> None:
    s = register_and_login()
    r = client.get(
        CURRENT, params={"lat": 12.9716, "lon": 77.5946, "at": AT.isoformat()}, headers=s["headers"]
    )
    p = pm25(r.json())
    assert p["method"] == "idw"
    assert {c["station_id"] for c in p["contributions"]} == {"openaq:5548", "openaq:6984"}
    assert 20.0 < p["concentration"] < 40.0
    assert sum(c["weight"] for c in p["contributions"]) == pytest.approx(1.0, abs=1e-3)


def test_pollutant_with_no_data_is_explicitly_unavailable(
    client, seeded, register_and_login
) -> None:
    s = register_and_login()
    body = client.get(
        CURRENT, params={"lat": 12.97, "lon": 77.59, "at": AT.isoformat()}, headers=s["headers"]
    ).json()
    o3 = next(p for p in body["pollutants"] if p["pollutant"] == "o3")
    assert o3["available"] is False and o3["concentration"] is None
    assert o3["unavailable_reason"] == "no_stations_in_radius"


def test_stale_data_reported_as_stale_not_current(client, seeded, register_and_login) -> None:
    s = register_and_login()
    later = AT + timedelta(hours=12)
    p = pm25(
        client.get(
            CURRENT,
            params={"lat": 12.97, "lon": 77.59, "at": later.isoformat()},
            headers=s["headers"],
        ).json()
    )
    assert p["available"] is False
    assert p["unavailable_reason"] == "no_fresh_data"


def test_far_away_location_has_no_stations(client, seeded, register_and_login) -> None:
    s = register_and_login()
    p = pm25(
        client.get(
            CURRENT, params={"lat": 28.61, "lon": 77.21, "at": AT.isoformat()}, headers=s["headers"]
        ).json()
    )
    assert p["unavailable_reason"] == "no_stations_in_radius"


@pytest.mark.parametrize("params", [{"lat": 95, "lon": 77}, {"lat": 12, "lon": 200}])
def test_invalid_location_error_code(client, register_and_login, params) -> None:
    s = register_and_login()
    r = client.get(CURRENT, params=params, headers=s["headers"])
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_LOCATION"


def test_future_at_rejected(client, register_and_login) -> None:
    s = register_and_login()
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    r = client.get(CURRENT, params={"lat": 12.97, "lon": 77.59, "at": future}, headers=s["headers"])
    assert r.status_code == 422


def test_stations_endpoint_lists_reference_monitors(client, seeded, register_and_login) -> None:
    s = register_and_login()
    ids = {x["id"] for x in client.get("/api/v1/air/stations", headers=s["headers"]).json()}
    assert ids == {"openaq:5548", "openaq:6984"}
