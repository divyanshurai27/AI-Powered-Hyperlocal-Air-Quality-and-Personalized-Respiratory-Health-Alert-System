from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from app.domain.pollutants import Pollutant
from app.ingestion import datagovin, openaq, openmeteo
from app.ingestion.base import SourceError, SourceUnavailable
from tests.fixtures.sources import (
    BTM,
    T0,
    hour_record,
    hours_payload,
    locations_payload,
    openmeteo_payload,
)

NO_SLEEP = {"sleep": lambda _s: None}


def openaq_client(handler) -> openaq.OpenAQClient:
    return openaq.OpenAQClient(
        "test-key", "https://api.openaq.test/v3", transport=httpx.MockTransport(handler), **NO_SLEEP
    )


def btm_station() -> openaq.StationInfo:
    return openaq.parse_location(locations_payload()["results"][0])


# --- OpenAQ ------------------------------------------------------------------------------


def test_list_locations_parses_and_skips_malformed() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["key"] = req.headers.get("X-API-Key")
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=locations_payload())

    stations = openaq_client(handler).list_locations(12.97, 77.59, 25)
    assert seen["key"] == "test-key"
    assert seen["params"]["radius"] == "25000"
    assert [s.location_id for s in stations] == [5548, 6973, 6206921]  # "broken" skipped

    btm = stations[0]
    assert btm.station_id == "openaq:5548"
    assert btm.is_reference
    assert {s.pollutant for s in btm.sensors} == {Pollutant.PM25, Pollutant.NO2}  # RH dropped
    assert not stations[2].is_reference  # AirGradient low-cost sensor


def test_sensor_overlap_skips_legacy_feed() -> None:
    legacy = next(s for s in btm_station().sensors if s.sensor_id == 14635)
    assert not legacy.overlaps(datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 4, 1, tzinfo=UTC))
    assert legacy.overlaps(datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 2, 1, tzinfo=UTC))


def test_hours_normalized_to_canonical_raw_shape() -> None:
    rec = hour_record(T0, 15.3, coverage=75.0, flagged=True)
    raw = openaq.normalize_hour(rec, btm_station())
    assert raw.station_id == "openaq:5548"
    assert raw.timestamp == T0  # period END, UTC
    assert (raw.latitude, raw.longitude) == (BTM["lat"], BTM["lon"])
    assert (raw.pollutant, raw.value, raw.unit) == ("pm25", 15.3, "µg/m³")
    assert raw.coverage_pct == 75.0
    assert raw.provider_flagged


def test_malformed_hour_record_does_not_raise() -> None:
    raw = openaq.normalize_hour({"value": "abc"}, btm_station())
    assert raw.timestamp is None and raw.value is None and raw.pollutant is None


def test_hours_paginate_until_short_page(monkeypatch) -> None:
    monkeypatch.setattr(openaq, "PAGE_LIMIT", 2)
    pages = {
        "1": [hour_record(T0, 1), hour_record(T0 + timedelta(hours=1), 2)],
        "2": [hour_record(T0 + timedelta(hours=2), 3)],
    }
    calls = []

    def handler(req):
        calls.append(req.url.params["page"])
        return httpx.Response(200, json=hours_payload(pages[req.url.params["page"]]))

    st = btm_station()
    raw = openaq_client(handler).fetch_sensor_hours(st, st.sensors[1], T0, T0 + timedelta(hours=3))
    assert calls == ["1", "2"]
    assert [r.value for r in raw] == [1, 2, 3]


def test_empty_response_is_not_an_error() -> None:
    st = btm_station()
    raw = openaq_client(lambda r: httpx.Response(200, json=hours_payload([]))).fetch_sensor_hours(
        st, st.sensors[1], T0, T0
    )
    assert raw == []


def test_unexpected_structure_raises_source_error() -> None:
    with pytest.raises(SourceError, match="unexpected response structure"):
        openaq_client(lambda r: httpx.Response(200, json={"oops": True})).list_locations(0, 0, 1)


def test_non_json_body_raises_source_error() -> None:
    with pytest.raises(SourceError, match="not valid JSON"):
        openaq_client(lambda r: httpx.Response(200, text="<html>")).list_locations(0, 0, 1)


def test_bad_api_key_is_clear_error_not_retried() -> None:
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401, json={"detail": "unauthorized"})

    with pytest.raises(SourceError, match="check the API key"):
        openaq_client(handler).list_locations(0, 0, 1)
    assert len(calls) == 1


def test_timeout_retried_then_unavailable() -> None:
    calls = []

    def handler(req):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(SourceUnavailable, match="timed out after 4 attempts"):
        openaq_client(handler).list_locations(0, 0, 1)
    assert len(calls) == 4


def test_rate_limit_respects_reset_header_then_succeeds() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"x-ratelimit-reset": "7"}),
            httpx.Response(200, json=locations_payload()),
        ]
    )
    waits = []
    client = openaq.OpenAQClient(
        "k",
        "https://api.openaq.test/v3",
        transport=httpx.MockTransport(lambda r: next(responses)),
        sleep=waits.append,
    )
    assert len(client.list_locations(0, 0, 1)) == 3
    assert waits == [7.0]


def test_missing_api_key_fails_fast() -> None:
    with pytest.raises(SourceError, match="OPENAQ_API_KEY"):
        openaq.OpenAQClient("", "https://x")


# --- Open-Meteo --------------------------------------------------------------------------


def meteo_client(handler) -> openmeteo.OpenMeteoClient:
    return openmeteo.OpenMeteoClient(
        "https://archive.test",
        "https://forecast.test",
        transport=httpx.MockTransport(handler),
        **NO_SLEEP,
    )


def test_archive_parsed_to_utc_records() -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    seen = {}

    def handler(req):
        seen.update(req.url.params)
        return httpx.Response(200, json=openmeteo_payload(start, 3))

    recs = meteo_client(handler).fetch_archive(
        12.97160001, 77.5946, date(2026, 9, 1), date(2026, 9, 1)
    )
    assert seen["timezone"] == "GMT" and seen["wind_speed_unit"] == "ms"
    assert [r.timestamp for r in recs] == [start + timedelta(hours=i) for i in range(3)]
    assert recs[0].kind == "reanalysis" and recs[0].issued_at is None
    assert (recs[0].latitude, recs[0].temperature, recs[0].pressure) == (12.9716, 22.0, 912.0)


def test_archive_null_tail_is_dropped_not_filled() -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    recs = meteo_client(
        lambda r: httpx.Response(200, json=openmeteo_payload(start, 5, null_from=3))
    ).fetch_archive(12.97, 77.59, date(2026, 9, 1), date(2026, 9, 1))
    assert len(recs) == 3


def test_forecast_records_carry_issue_time() -> None:
    issued = datetime(2026, 9, 24, 17, 42, tzinfo=UTC)
    start = datetime(2026, 9, 24, 18, tzinfo=UTC)
    recs = meteo_client(
        lambda r: httpx.Response(200, json=openmeteo_payload(start, 48))
    ).fetch_forecast(12.97, 77.59, issued)
    assert len(recs) == 48
    assert {r.issued_at for r in recs} == {datetime(2026, 9, 24, 17, tzinfo=UTC)}
    assert all(r.kind == "forecast" for r in recs)


def test_length_mismatch_is_malformed() -> None:
    body = openmeteo_payload(datetime(2026, 9, 1, tzinfo=UTC), 3)
    body["hourly"]["temperature_2m"] = [1.0]
    with pytest.raises(SourceError, match="length"):
        openmeteo.parse_hourly(body, kind="reanalysis", latitude=0, longitude=0, issued_at=None)


def test_missing_hourly_block_is_malformed() -> None:
    with pytest.raises(SourceError, match="hourly.time"):
        openmeteo.parse_hourly(
            {"error": True}, kind="reanalysis", latitude=0, longitude=0, issued_at=None
        )


def test_invalid_weather_coordinates_rejected_before_request() -> None:
    with pytest.raises(SourceError, match="invalid coordinates"):
        meteo_client(lambda r: pytest.fail("should not call")).fetch_archive(
            100, 0, date(2026, 1, 1), date(2026, 1, 1)
        )


# --- data.gov.in -------------------------------------------------------------------------


def test_datagovin_refuses_to_emit_until_units_verified() -> None:
    assert datagovin.UNITS_VERIFIED is False
    with pytest.raises(SourceError, match="units not verified"):
        datagovin.to_raw_observations(
            [{"station": "Hebbal", "pollutant_id": "PM2.5", "avg_value": "30"}]
        )


def test_datagovin_parsing_when_verified(monkeypatch) -> None:
    monkeypatch.setattr(datagovin, "UNITS_VERIFIED", True)
    raw = datagovin.to_raw_observations(
        [
            {
                "station": "Hebbal, Bengaluru - KSPCB",
                "pollutant_id": "PM2.5",
                "avg_value": "31",
                "last_update": "24-09-2026 15:00:00",
                "latitude": "13.029",
                "longitude": "77.585",
            },
            {
                "station": "Hebbal, Bengaluru - KSPCB",
                "pollutant_id": "NO2",
                "avg_value": "NA",
                "last_update": "24-09-2026 15:00:00",
                "latitude": "13.029",
                "longitude": "77.585",
            },
        ]
    )
    assert raw[0].station_id == "cpcb_datagovin:hebbal__bengaluru___kspcb"
    assert raw[0].timestamp == datetime(2026, 9, 24, 9, 30, tzinfo=UTC)
    assert raw[0].value == 31.0
    assert raw[1].value is None  # "NA" stays missing, never 0


def test_datagovin_bad_date_is_none() -> None:
    assert datagovin.parse_last_update("2026-09-24T15:00") is None


def test_datagovin_gateway_error_fails_fast() -> None:
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(502)

    client = datagovin.DataGovInClient(
        "k", "https://api.test", "res", transport=httpx.MockTransport(handler), **NO_SLEEP
    )
    with pytest.raises(SourceUnavailable, match="HTTP 502"):
        client.fetch_current("Bengaluru")
    assert len(calls) == 2


def test_cpcb_co_labelled_ppb_is_read_as_mg_m3() -> None:
    raw = openaq.normalize_hour(hour_record(T0, 0.6, param="co", units="ppb"), btm_station())
    assert raw.unit == "mg/m³"


def test_no2_ppb_label_is_not_overridden() -> None:
    raw = openaq.normalize_hour(hour_record(T0, 20.0, param="no2", units="ppb"), btm_station())
    assert raw.unit == "ppb"
