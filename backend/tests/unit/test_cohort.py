"""Synthetic cohort: determinism, the live-rule twin, labels, and the observation process."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.domain.observations import QualityFlag
from app.domain.spatial import GeoPoint, LocalEstimate, StationReading, estimate_local_air_quality
from app.ml.cohort import COHORT_VERSION, idw_field
from app.ml.risk_dataset import (
    WARMUP_HOURS,
    build_cohort_matrix,
    labels_and_mask,
    patient_exposure,
    sample_origins,
)
from tests.fixtures.cohort import small_cohort


@pytest.fixture(scope="module")
def cohort():
    return small_cohort(40, 60)


def test_same_seed_same_cohort(cohort) -> None:
    again = small_cohort(40, 60)
    assert np.array_equal(cohort.onset, again.onset)
    pd.testing.assert_frame_equal(cohort.diary, again.diary)
    assert cohort.version == COHORT_VERSION


def test_negative_control_changes_only_pollution_driven_parts(cohort) -> None:
    null = small_cohort(40, 60, aq_effect_scale=0.0)
    pd.testing.assert_frame_equal(cohort.patients, null.patients)
    for p in ("pm25", "pm10", "o3"):
        np.testing.assert_array_equal(cohort.true_exposure[p], null.true_exposure[p])
        np.testing.assert_array_equal(cohort.amb_est[p], null.amb_est[p])
    assert null.summary()["aq_effect_scale"] == 0.0


def test_event_rates_are_in_the_configured_range() -> None:
    c = small_cohort(200, 120)
    s = c.summary()
    assert 3.5 < s["asthma_events_per_year"] < 9
    assert 2.5 < s["copd_events_per_year"] < 7


def test_episodes_never_overlap(cohort) -> None:
    gap = cohort.config.episode_hours
    for row in cohort.onset:
        t = np.flatnonzero(row)
        assert (np.diff(t) >= gap).all()


def test_estimated_exposure_is_missing_without_a_location(cohort) -> None:
    assert np.isnan(cohort.amb_est["pm25"][~cohort.location_available]).all()
    assert 0.8 < cohort.location_available.mean() < 1.0
    exp = patient_exposure(cohort, 0)
    assert exp["pm25"].isna().sum() >= (~cohort.location_available[0]).sum()


def test_true_and_estimated_exposure_differ_but_correlate(cohort) -> None:
    est = patient_exposure(cohort, 0)["pm25"].to_numpy()
    true = cohort.true_exposure["pm25"][0]
    ok = ~np.isnan(est)
    assert not np.allclose(est[ok], true[ok])
    assert np.corrcoef(est[ok], true[ok])[0, 1] > 0.5


def test_hidden_traits_are_not_in_the_profile(cohort) -> None:
    hidden = {"frailty", "sensitivity", "diary_adherence", "location_log_rate"}
    assert hidden <= set(cohort.truth.columns)
    assert not hidden & set(cohort.patients.columns)


def test_diary_missingness_is_explicit(cohort) -> None:
    d = cohort.diary
    assert d["symptom_score"].isna().any()
    assert d["spo2"].notna().sum() < len(d)  # only wearable users have SpO2
    assert d["symptom_score"].dropna().between(0, 10).all()
    # Diaries are at 21:00 IST (15:30 UTC).
    assert (d["timestamp"].dt.hour == 15).all() and (d["timestamp"].dt.minute == 30).all()


def test_labels_look_forward_24h_and_skip_ongoing_episodes() -> None:
    onset = np.zeros(200, dtype=bool)
    onset[100] = True
    idx = np.array([75, 76, 99, 100, 101, 171, 172])
    y, keep = labels_and_mask(onset, idx, episode_hours=72)
    # t=75: window (75, 99] misses 100; t=76 catches it; t=100: onset is not in (100, 124]
    assert y.tolist() == [False, True, True, False, False, False, False]
    # in episode while an onset lies in (t-72, t]: t=100..171
    assert keep.tolist() == [True, True, True, False, False, False, True]


def test_matrix_rows_are_aligned_and_warm(cohort) -> None:
    origins = sample_origins(cohort)
    X, y, meta = build_cohort_matrix(cohort, origins)
    assert len(X) == len(y) == len(meta)
    assert (meta["t"] >= cohort.hours[WARMUP_HOURS]).all()
    assert (meta["t"] <= cohort.hours[-25]).all()
    assert set(y.unique()) <= {0, 1}
    assert (X["pt_is_copd"] == (meta["disease_type"] == "copd")).all()


def _readings(values, flags, coords, at):
    return [
        StationReading(f"s{j}", lat, lon, float(v), at, f)
        for j, ((lat, lon), v, f) in enumerate(zip(coords, values, flags, strict=True))
        if not np.isnan(v)
    ]


def test_vectorised_idw_matches_the_domain_rule() -> None:
    """The cohort's view of ambient must be the same rule the API applies (no drift)."""
    rng = np.random.default_rng(3)
    coords = [(12.9 + rng.uniform(-0.1, 0.1), 77.6 + rng.uniform(-0.1, 0.1)) for _ in range(8)]
    at = pd.Timestamp("2025-06-01 00:30", tz="UTC").to_pydatetime()
    for _ in range(200):
        point = GeoPoint(12.9 + rng.uniform(-0.1, 0.1), 77.6 + rng.uniform(-0.1, 0.1))
        values = np.where(rng.random(8) < 0.3, np.nan, rng.uniform(5, 100, 8))
        valid = rng.random(8) < 0.7
        flags = [QualityFlag.VALID if v else QualityFlag.LOW_COVERAGE for v in valid]
        qual = np.where(valid, 1.0, 0.5)
        from app.ml.cohort import _haversine_km

        d = _haversine_km(
            point.latitude,
            point.longitude,
            np.array([c[0] for c in coords]),
            np.array([c[1] for c in coords]),
        )
        ours = idw_field(
            values[None, :], qual[None, :], d, radius_km=10, max_stations=4, exact_km=0.5
        )[0]
        ref = estimate_local_air_quality(
            point,
            _readings(values, flags, coords, at),
            at,
            radius_km=10,
            max_age=timedelta(hours=3),
            exact_match_km=0.5,
            max_stations=4,
        )
        if isinstance(ref, LocalEstimate):
            assert ours == pytest.approx(ref.concentration, abs=0.006)
        else:
            assert np.isnan(ours)
