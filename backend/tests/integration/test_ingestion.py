from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from sqlalchemy import func, select

from app.ingestion import openaq, openmeteo
from app.models import AirQualityObservation, IngestionRun, Station, WeatherObservation
from app.services.ingestion import IngestionService
from tests.fixtures.sources import (
    BTM,
    T0,
    hour_record,
    hours_payload,
    locations_payload,
    openmeteo_payload,
)

START, END = T0 - timedelta(hours=1), T0 + timedelta(hours=5)


def make_openaq(hours_by_sensor: dict[int, list | int]) -> openaq.OpenAQClient:
    """hours_by_sensor: sensor id → list of records, or an HTTP status to fail with."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/locations"):
            return httpx.Response(200, json=locations_payload())
        sensor_id = int(req.url.path.split("/")[-2])
        spec = hours_by_sensor.get(sensor_id, [])
        if isinstance(spec, int):
            return httpx.Response(spec)
        return httpx.Response(200, json=hours_payload(spec))

    return openaq.OpenAQClient(
        "k",
        "https://api.openaq.test/v3",
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


PM25_BTM, NO2_BTM, PM25_JAY = 12235361, 12235358, 12235267


def standard_hours() -> dict[int, list]:
    return {
        PM25_BTM: [
            hour_record(T0, 15.3),
            hour_record(T0 + timedelta(hours=1), 18.0, coverage=50.0),
            hour_record(T0 + timedelta(hours=2), None),  # rejected: missing value
            hour_record(T0 + timedelta(hours=2), 17.0),
            hour_record(T0, 15.3),  # exact duplicate
        ],
        NO2_BTM: [hour_record(T0, 10.0, param="no2", units="ppb")],
        PM25_JAY: [
            hour_record(T0, 22.0),
            hour_record(T0 + timedelta(hours=1), -5.0),
        ],  # negative rejected
    }


def ingest(db, client):
    svc = IngestionService(db)
    stations = svc.sync_openaq_stations(client, 12.97, 77.59, 25)
    return svc.ingest_openaq_hours(client, stations, START, END)


def test_it01_source_fixture_to_exact_canonical_row(db) -> None:
    run = ingest(db, make_openaq(standard_hours()))

    assert run.status == "success"
    assert run.records_fetched == 8
    assert run.records_rejected == 2
    assert run.rejection_reasons == {"missing_value": 1, "negative_value": 1}
    assert run.duplicates_dropped == 1
    assert run.records_written == 5

    row = db.scalar(
        select(AirQualityObservation).where(
            AirQualityObservation.station_id == "openaq:5548",
            AirQualityObservation.pollutant == "no2",
        )
    )
    assert row.source == "openaq"
    assert row.timestamp == T0
    assert (row.latitude, row.longitude) == (BTM["lat"], BTM["lon"])
    assert row.concentration == pytest.approx(18.816, abs=1e-3)
    assert (row.unit, row.raw_value, row.raw_unit) == ("µg/m³", 10.0, "ppb")
    assert row.quality_flag == "valid"
    assert row.ingestion_run_id == run.id

    flagged = db.scalar(
        select(AirQualityObservation).where(
            AirQualityObservation.timestamp == T0 + timedelta(hours=1),
            AirQualityObservation.station_id == "openaq:5548",
        )
    )
    assert flagged.quality_flag == "low_coverage"


def test_every_persisted_row_satisfies_ingestion_invariants(db) -> None:
    """PRD §8.2: every row has source, station, timestamp, pollutant, value and unit."""
    ingest(db, make_openaq(standard_hours()))
    rows = db.scalars(select(AirQualityObservation)).all()
    assert rows
    for r in rows:
        assert r.source and r.station_id and r.pollutant and r.unit
        assert r.timestamp.tzinfo is not None
        assert r.concentration is not None and r.concentration >= 0


def test_stations_synced_with_reference_flag(db) -> None:
    ingest(db, make_openaq(standard_hours()))
    stations = {s.id: s for s in db.scalars(select(Station))}
    assert set(stations) == {"openaq:5548", "openaq:6973", "openaq:6206921"}
    assert stations["openaq:5548"].is_reference_monitor
    assert not stations["openaq:6206921"].is_reference_monitor
    assert {x["id"] for x in stations["openaq:5548"].source_metadata["sensors"]} == {
        14635,
        12235361,
        12235358,
    }


def test_low_cost_sensor_data_is_never_ingested_as_ground_truth(db) -> None:
    hours = standard_hours() | {99001: [hour_record(T0, 999.0)]}
    ingest(db, make_openaq(hours))
    count = db.scalar(
        select(func.count()).where(AirQualityObservation.station_id == "openaq:6206921")
    )
    assert count == 0


def test_rerun_is_idempotent(db) -> None:
    client = make_openaq(standard_hours())
    ingest(db, client)
    before = db.scalar(select(func.count()).select_from(AirQualityObservation))
    second = ingest(db, client)
    after = db.scalar(select(func.count()).select_from(AirQualityObservation))
    assert before == after == 5
    assert second.records_written == 0  # nothing better arrived, nothing rewritten


def test_better_revision_replaces_worse_record(db) -> None:
    ingest(db, make_openaq({PM25_BTM: [hour_record(T0 + timedelta(hours=1), 18.0, coverage=50.0)]}))
    run = ingest(
        db, make_openaq({PM25_BTM: [hour_record(T0 + timedelta(hours=1), 19.5, coverage=100.0)]})
    )
    row = db.scalar(
        select(AirQualityObservation).where(
            AirQualityObservation.timestamp == T0 + timedelta(hours=1)
        )
    )
    assert (row.concentration, row.coverage_pct, row.quality_flag) == (19.5, 100.0, "valid")
    assert run.records_written == 1


def test_worse_revision_does_not_overwrite(db) -> None:
    ingest(db, make_openaq({PM25_BTM: [hour_record(T0, 15.3, coverage=100.0)]}))
    ingest(db, make_openaq({PM25_BTM: [hour_record(T0, 99.0, coverage=25.0)]}))
    row = db.scalar(
        select(AirQualityObservation).where(AirQualityObservation.station_id == "openaq:5548")
    )
    assert row.concentration == 15.3


def test_partial_source_failure_keeps_other_sensors(db) -> None:
    hours = standard_hours() | {PM25_JAY: 500}
    run = ingest(db, make_openaq(hours))
    assert run.status == "partial"
    assert len(run.errors) == 1 and "12235267" in run.errors[0]
    assert (
        db.scalar(select(func.count()).where(AirQualityObservation.station_id == "openaq:5548"))
        == 4
    )
    persisted = db.get(IngestionRun, run.id)
    assert persisted.finished_at is not None


def test_total_source_failure_is_failed_and_writes_nothing(db) -> None:
    run = ingest(db, make_openaq({PM25_BTM: 503, NO2_BTM: 503, PM25_JAY: 503}))
    assert run.status == "failed"
    assert db.scalar(select(func.count()).select_from(AirQualityObservation)) == 0


def test_weather_archive_ingest_and_idempotency(db) -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    client = openmeteo.OpenMeteoClient(
        "https://a.test",
        "https://f.test",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=openmeteo_payload(start, 24))
        ),
    )
    svc = IngestionService(db)
    run = svc.ingest_weather_archive(client, 12.9716, 77.5946, date(2026, 9, 1), date(2026, 9, 1))
    svc.ingest_weather_archive(client, 12.9716, 77.5946, date(2026, 9, 1), date(2026, 9, 1))
    assert run.status == "success"
    assert db.scalar(select(func.count()).select_from(WeatherObservation)) == 24


def test_forecasts_from_different_issue_times_coexist(db) -> None:
    start = datetime(2026, 9, 24, 18, tzinfo=UTC)
    client = openmeteo.OpenMeteoClient(
        "https://a.test",
        "https://f.test",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=openmeteo_payload(start, 48))
        ),
    )
    svc = IngestionService(db)
    svc.ingest_weather_forecast(
        client, 12.97, 77.59, issued_at=datetime(2026, 9, 24, 17, tzinfo=UTC)
    )
    svc.ingest_weather_forecast(
        client, 12.97, 77.59, issued_at=datetime(2026, 9, 24, 18, tzinfo=UTC)
    )
    # Same target hours, two issue times: both kept, so later we can use only what existed at T.
    assert db.scalar(select(func.count()).select_from(WeatherObservation)) == 96


def test_long_ranges_are_fetched_in_chunks(db, monkeypatch) -> None:
    from app.services import ingestion as ingestion_module

    monkeypatch.setattr(ingestion_module, "CHUNK_DAYS", 1)
    windows = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/locations"):
            return httpx.Response(200, json=locations_payload())
        if req.url.path.endswith("/12235361/hours"):
            windows.append((req.url.params["datetime_from"], req.url.params["datetime_to"]))
        return httpx.Response(200, json=hours_payload([]))

    client = openaq.OpenAQClient(
        "k",
        "https://api.openaq.test/v3",
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )
    svc = IngestionService(db)
    stations = svc.sync_openaq_stations(client, 12.97, 77.59, 25)
    start = datetime(2026, 9, 1, tzinfo=UTC)
    svc.ingest_openaq_hours(client, stations, start, start + timedelta(days=3))
    assert len(windows) == 3
    assert windows[0][0].startswith("2026-09-01") and windows[-1][1].startswith("2026-09-04")


def test_http_408_is_retried(db) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/locations"):
            return httpx.Response(200, json=locations_payload())
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(408, json={"detail": "Connection timed out"})
        return httpx.Response(200, json=hours_payload([hour_record(T0, 15.3)]))

    client = openaq.OpenAQClient(
        "k",
        "https://api.openaq.test/v3",
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )
    svc = IngestionService(db)
    stations = [
        s for s in svc.sync_openaq_stations(client, 12.97, 77.59, 25) if s.location_id == 5548
    ]
    stations[0].sensors[:] = [x for x in stations[0].sensors if x.sensor_id == 12235361]
    run = svc.ingest_openaq_hours(client, stations, START, END)
    assert run.status == "success" and run.records_written == 1
