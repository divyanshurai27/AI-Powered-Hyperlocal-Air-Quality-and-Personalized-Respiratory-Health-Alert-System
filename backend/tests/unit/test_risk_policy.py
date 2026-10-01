"""Risk level policy (BR-01, BR-02) and the personal baseline (PRD §16)."""

import math

import pytest

from app.domain.enums import DiseaseType, RiskLevel
from app.domain.risk_policy import (
    MAX_OFFSET,
    RISK_THRESHOLDS,
    InvalidProbabilityError,
    apply_personal_offset,
    classify_risk,
    estimate_personal_baseline,
    is_alert,
    validate_probability,
)


@pytest.mark.parametrize("disease", list(DiseaseType))
def test_thresholds_are_strictly_increasing_probabilities(disease) -> None:
    moderate, high, severe = RISK_THRESHOLDS[disease]
    assert 0 < moderate < high < severe < 1


@pytest.mark.parametrize(
    ("offset_from", "expected"),
    [
        (None, RiskLevel.LOW),
        (0, RiskLevel.MODERATE),
        (1, RiskLevel.HIGH),
        (2, RiskLevel.SEVERE),
    ],
)
@pytest.mark.parametrize("disease", list(DiseaseType))
def test_classification_boundaries_are_inclusive(disease, offset_from, expected) -> None:
    bounds = RISK_THRESHOLDS[disease]
    p = bounds[0] / 2 if offset_from is None else bounds[offset_from]
    assert classify_risk(p, disease) == expected
    if offset_from is not None:
        # just below the bound falls into the level beneath
        assert classify_risk(p - 1e-9, disease) != expected


def test_extremes() -> None:
    assert classify_risk(0.0, "asthma") == RiskLevel.LOW
    assert classify_risk(1.0, "copd") == RiskLevel.SEVERE


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_out_of_range_probability_is_rejected(bad) -> None:
    with pytest.raises(InvalidProbabilityError):
        validate_probability(bad)
    with pytest.raises(InvalidProbabilityError):
        classify_risk(bad, "asthma")


def test_alert_means_high_or_severe() -> None:
    assert not is_alert(RiskLevel.LOW)
    assert not is_alert(RiskLevel.MODERATE)
    assert is_alert(RiskLevel.HIGH)
    assert is_alert("Severe")


def test_no_history_gives_zero_offset() -> None:
    b = estimate_personal_baseline(0, 0, 0.02)
    assert b.offset == 0
    assert apply_personal_offset(0.1, b) == pytest.approx(0.1)


def test_offset_sign_follows_observed_vs_expected() -> None:
    rate = 5 / 365
    frequent = estimate_personal_baseline(15, 365, rate)
    rare = estimate_personal_baseline(0, 365, rate)
    assert frequent.offset > 0 > rare.offset
    assert apply_personal_offset(0.05, frequent) > 0.05 > apply_personal_offset(0.05, rare)


def test_offset_shrinks_with_little_history() -> None:
    rate = 5 / 365
    short = estimate_personal_baseline(2, 30, rate)  # 2 events in a month: striking but thin
    long = estimate_personal_baseline(24, 360, rate)  # the same rate for a year
    assert 0 < short.offset < long.offset


def test_offset_is_bounded() -> None:
    b = estimate_personal_baseline(500, 365, 0.001)
    assert b.offset == pytest.approx(MAX_OFFSET)
    p = apply_personal_offset(0.2, b)
    odds = p / (1 - p)
    assert odds == pytest.approx(2 * 0.2 / 0.8)
    assert estimate_personal_baseline(0, 365, 1.0).offset == pytest.approx(-math.log(2))


def test_negative_history_is_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_personal_baseline(-1, 10, 0.01)
