"""Alert rules (PRD §19, §27 "Alerts", BR-10, BR-12)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.domain.alerts import (
    AlertDecision,
    ExistingAlert,
    PredictionForAlert,
    alert_message,
    effective_state,
    evaluate_alert,
    format_percent,
)
from app.domain.enums import AlertState, RiskLevel

T = datetime(2026, 9, 20, 9, 30, tzinfo=UTC)
MAX_AGE = timedelta(hours=2)


def pred(level=RiskLevel.HIGH, p=0.06, at=T, horizon=0, pid=None) -> PredictionForAlert:
    return PredictionForAlert(
        prediction_id=pid or uuid.uuid4(),
        horizon_hours=horizon,
        risk_probability=p,
        risk_level=level,
        prediction_time=at,
        target_window_end=at + timedelta(hours=24),
    )


def alert(
    level=RiskLevel.HIGH, state=AlertState.CREATED, expires=T + timedelta(hours=20), pid=None
):
    return ExistingAlert(uuid.uuid4(), pid or uuid.uuid4(), state, level, expires)


def test_below_threshold_no_alert() -> None:
    for level in (RiskLevel.LOW, RiskLevel.MODERATE):
        assert evaluate_alert(pred(level, 0.03), [], T, MAX_AGE).decision == (
            AlertDecision.BELOW_ALERT_LEVEL
        )


@pytest.mark.parametrize("level", [RiskLevel.HIGH, RiskLevel.SEVERE])
def test_threshold_met_creates_alert(level) -> None:
    e = evaluate_alert(pred(level), [], T + timedelta(minutes=5), MAX_AGE)
    assert e.create and e.cancel == {}


def test_repeated_prediction_one_alert() -> None:
    p = pred()
    existing = [alert(pid=p.prediction_id)]
    assert evaluate_alert(p, existing, T, MAX_AGE).decision == AlertDecision.ALREADY_ALERTED


def test_stale_prediction_no_new_alert() -> None:
    e = evaluate_alert(pred(), [], T + MAX_AGE + timedelta(minutes=1), MAX_AGE)
    assert e.decision == AlertDecision.STALE_PREDICTION
    future = evaluate_alert(pred(at=T + timedelta(hours=1)), [], T, MAX_AGE)
    assert future.decision == AlertDecision.STALE_PREDICTION


def test_projected_hours_never_alert() -> None:
    e = evaluate_alert(pred(RiskLevel.SEVERE, 0.2, horizon=6), [], T, MAX_AGE)
    assert e.decision == AlertDecision.PROJECTED_HORIZON


@pytest.mark.parametrize("p", [-0.1, 1.2, float("nan")])
def test_invalid_probability_never_alerts(p) -> None:
    assert evaluate_alert(pred(p=p), [], T, MAX_AGE).decision == AlertDecision.INVALID_PREDICTION


@pytest.mark.parametrize("state", [AlertState.CREATED, AlertState.SENT, AlertState.ACKNOWLEDGED])
def test_equivalent_open_or_acknowledged_alert_blocks(state) -> None:
    e = evaluate_alert(pred(), [alert(state=state)], T, MAX_AGE)
    assert e.decision == AlertDecision.EQUIVALENT_ALERT_OPEN


def test_higher_open_alert_blocks_a_lower_one() -> None:
    e = evaluate_alert(pred(RiskLevel.HIGH), [alert(RiskLevel.SEVERE)], T, MAX_AGE)
    assert e.decision == AlertDecision.EQUIVALENT_ALERT_OPEN


def test_ended_or_cancelled_alerts_do_not_block() -> None:
    old = alert(expires=T)  # window ended exactly at the new prediction time
    cancelled = alert(state=AlertState.CANCELLED)
    assert evaluate_alert(pred(), [old, cancelled], T, MAX_AGE).create


def test_escalation_creates_and_supersedes_open_lower_alerts() -> None:
    open_high = alert(RiskLevel.HIGH)
    acked_high = alert(RiskLevel.HIGH, AlertState.ACKNOWLEDGED)
    e = evaluate_alert(pred(RiskLevel.SEVERE, 0.1), [open_high, acked_high], T, MAX_AGE)
    assert e.create
    assert e.cancel == {open_high.id: "superseded_by_higher_level"}


def test_revised_prediction_withdraws_its_own_alert() -> None:
    p = pred(RiskLevel.MODERATE, 0.03)
    own = alert(pid=p.prediction_id)
    other = alert()
    e = evaluate_alert(p, [own, other], T, MAX_AGE)
    assert e.decision == AlertDecision.BELOW_ALERT_LEVEL
    assert e.cancel == {own.id: "prediction_revised"}


def test_effective_state_expires_only_open_alerts() -> None:
    end = T + timedelta(hours=24)
    assert effective_state("created", end, end) == AlertState.EXPIRED
    assert effective_state("sent", end, T) == AlertState.SENT
    assert effective_state("acknowledged", end, end + timedelta(days=1)) == (
        AlertState.ACKNOWLEDGED
    )
    assert effective_state("cancelled", end, end) == AlertState.CANCELLED


def test_message_is_level_and_percent_only() -> None:
    msg = alert_message(RiskLevel.HIGH, 0.0612)
    assert msg == "Your estimated respiratory risk for the next 24 hours is High (6.1%)."
    for word in ("inhaler", "dose", "medication", "take ", "increase"):
        assert word not in msg.lower()
    assert format_percent(0.25) == "25%"
