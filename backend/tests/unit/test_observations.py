import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.domain.observations import (
    QualityFlag,
    RawObservation,
    RejectReason,
    deduplicate_observations,
    normalize_timestamp,
    validate_observation,
)
from app.domain.pollutants import Pollutant, UnitConversionError, parse_pollutant, to_canonical

IST = timezone(timedelta(hours=5, minutes=30))
T = datetime(2026, 9, 20, 0, 30, tzinfo=UTC)

RAW = RawObservation(
    source="openaq",
    station_id="openaq:5548",
    timestamp=T,
    latitude=12.9135,
    longitude=77.5951,
    pollutant="pm25",
    value=15.3,
    unit="µg/m³",
    coverage_pct=100.0,
)


# --- units -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pollutant", "ppb", "expected_ug"),
    [
        (Pollutant.NO2, 1.0, 1.8816),  # 46.0055 / 24.45
        (Pollutant.O3, 1.0, 1.9631),
        (Pollutant.SO2, 1.0, 2.6203),
        (Pollutant.CO, 1000.0, 1145.603),
    ],
)
def test_ppb_to_ug_m3_at_25c(pollutant, ppb, expected_ug) -> None:
    assert to_canonical(pollutant, ppb, "ppb") == pytest.approx(expected_ug, abs=1e-3)


def test_ppm_and_mg_m3_conversions() -> None:
    assert to_canonical(Pollutant.NO2, 1.0, "ppm") == pytest.approx(1881.6, abs=0.1)
    assert to_canonical(Pollutant.CO, 1.2, "mg/m³") == pytest.approx(1200.0)


def test_greek_mu_and_micro_sign_both_accepted() -> None:
    assert (
        to_canonical(Pollutant.PM25, 10, "μg/m³") == to_canonical(Pollutant.PM25, 10, "µg/m³") == 10
    )


def test_particulate_matter_in_ppb_is_rejected() -> None:
    with pytest.raises(UnitConversionError):
        to_canonical(Pollutant.PM25, 10, "ppb")


def test_unknown_unit_is_rejected_not_guessed() -> None:
    with pytest.raises(UnitConversionError):
        to_canonical(Pollutant.NO2, 10, "furlongs")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PM2.5", Pollutant.PM25),
        ("OZONE", Pollutant.O3),
        ("pm25", Pollutant.PM25),
        ("nh3", None),
        (None, None),
    ],
)
def test_parse_pollutant(name, expected) -> None:
    assert parse_pollutant(name) == expected


# --- validation --------------------------------------------------------------------------


def test_valid_record_becomes_canonical() -> None:
    out = validate_observation(RAW)
    assert out.ok
    obs = out.observation
    assert obs.pollutant == Pollutant.PM25
    assert obs.concentration == 15.3
    assert obs.unit == "µg/m³"
    assert obs.quality_flag == QualityFlag.VALID
    assert obs.timestamp.tzinfo == UTC


def test_gas_in_ppb_is_converted_and_raw_value_kept() -> None:
    obs = validate_observation(replace(RAW, pollutant="no2", value=10.0, unit="ppb")).observation
    assert obs.concentration == pytest.approx(18.816, abs=1e-3)
    assert (obs.raw_value, obs.raw_unit) == (10.0, "ppb")


def test_ist_timestamp_normalized_to_utc() -> None:
    local = datetime(2026, 9, 20, 6, 0, tzinfo=IST)
    obs = validate_observation(replace(RAW, timestamp=local)).observation
    assert obs.timestamp == T
    assert obs.timestamp.utcoffset() == timedelta(0)


def test_normalize_timestamp_rejects_naive() -> None:
    with pytest.raises(ValueError):
        normalize_timestamp(datetime(2026, 9, 20, 0, 30))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"timestamp": None}, RejectReason.MISSING_TIMESTAMP),
        ({"timestamp": datetime(2026, 9, 20)}, RejectReason.NAIVE_TIMESTAMP),
        ({"pollutant": None}, RejectReason.MISSING_POLLUTANT),
        ({"pollutant": "nh3"}, RejectReason.UNSUPPORTED_POLLUTANT),
        ({"latitude": 91.0}, RejectReason.INVALID_COORDINATES),
        ({"longitude": -181.0}, RejectReason.INVALID_COORDINATES),
        ({"latitude": math.nan}, RejectReason.INVALID_COORDINATES),
        ({"latitude": None}, RejectReason.INVALID_COORDINATES),
        ({"value": None}, RejectReason.MISSING_VALUE),
        ({"value": math.inf}, RejectReason.NON_FINITE_VALUE),
        ({"value": math.nan}, RejectReason.NON_FINITE_VALUE),
        ({"value": -1.0}, RejectReason.NEGATIVE_VALUE),
        ({"unit": "ppb"}, RejectReason.INVALID_UNIT),  # pm25 in ppb
        ({"unit": None}, RejectReason.INVALID_UNIT),
        ({"station_id": None}, RejectReason.MISSING_STATION),
    ],
)
def test_invalid_records_rejected_with_reason(change, reason) -> None:
    out = validate_observation(replace(RAW, **change))
    assert not out.ok
    assert reason in out.reasons


def test_zero_concentration_is_valid() -> None:
    assert validate_observation(replace(RAW, value=0.0)).ok


def test_multiple_problems_all_reported() -> None:
    out = validate_observation(replace(RAW, timestamp=None, latitude=200.0, value=None))
    assert {
        RejectReason.MISSING_TIMESTAMP,
        RejectReason.INVALID_COORDINATES,
        RejectReason.MISSING_VALUE,
    } <= set(out.reasons)


@pytest.mark.parametrize(
    ("change", "flag"),
    [
        ({"coverage_pct": 74.9}, QualityFlag.LOW_COVERAGE),
        ({"coverage_pct": 75.0}, QualityFlag.VALID),
        ({"coverage_pct": None}, QualityFlag.VALID),
        ({"provider_flagged": True}, QualityFlag.PROVIDER_FLAGGED),
        ({"value": 1500.0}, QualityFlag.IMPLAUSIBLE),
        (
            {"value": 1500.0, "coverage_pct": 10.0, "provider_flagged": True},
            QualityFlag.IMPLAUSIBLE,
        ),
        ({"coverage_pct": 10.0, "provider_flagged": True}, QualityFlag.PROVIDER_FLAGGED),
    ],
)
def test_doubtful_records_kept_but_flagged(change, flag) -> None:
    out = validate_observation(replace(RAW, **change))
    assert out.ok
    assert out.observation.quality_flag == flag


# --- deduplication -----------------------------------------------------------------------


def _obs(**change):
    return validate_observation(replace(RAW, **change)).observation


def test_exact_duplicates_collapse_to_one() -> None:
    result = deduplicate_observations([_obs(), _obs(), _obs()])
    assert len(result.observations) == 1
    assert result.duplicates_dropped == 2
    assert result.conflicts == 0


def test_different_identity_is_not_a_duplicate() -> None:
    result = deduplicate_observations(
        [
            _obs(),
            _obs(timestamp=T + timedelta(hours=1)),
            _obs(pollutant="pm10"),
            _obs(station_id="openaq:1"),
        ]
    )
    assert len(result.observations) == 4


def test_conflict_prefers_valid_then_higher_coverage() -> None:
    flagged = _obs(value=20.0, provider_flagged=True, coverage_pct=100.0)
    partial = _obs(value=18.0, coverage_pct=80.0)
    full = _obs(value=16.0, coverage_pct=100.0)
    result = deduplicate_observations([flagged, partial, full])
    assert len(result.observations) == 1
    assert result.observations[0].concentration == 16.0
    assert result.conflicts == 2


def test_conflict_full_tie_keeps_first() -> None:
    result = deduplicate_observations([_obs(value=10.0), _obs(value=11.0)])
    assert result.observations[0].concentration == 10.0
    assert result.conflicts == 1
