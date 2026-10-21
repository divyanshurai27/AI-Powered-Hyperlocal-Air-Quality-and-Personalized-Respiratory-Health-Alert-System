"""Alert rules (PRD §19, BR-10, BR-12), ALERT_POLICY_VERSION.

    prediction valid
    AND it is the current estimate (horizon 0), not a projected hour
    AND risk level ≥ ALERT_LEVEL (risk_policy, High)
    AND the prediction is current (made ≤ max age before evaluation)
    AND no alert exists for this prediction
    AND no equivalent alert is open
    → create alert

"Equivalent" = an alert for the same patient, not cancelled, whose window has not ended at
the new prediction time, at the same or a higher level. So a patient whose risk stays High
hour after hour gets one alert per 24 h window, and an acknowledged alert is not repeated.
A HIGHER level than every open alert (High → Severe) is an escalation: a new alert is
created and the open lower ones are cancelled as superseded.

An alert says only the risk level and percentage (locked scope): never what to do with
medication (BR-12).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from app.domain.enums import AlertState, RiskLevel
from app.domain.risk_policy import ALERT_LEVEL, InvalidProbabilityError, level_rank
from app.domain.risk_policy import validate_probability as _validate

ALERT_POLICY_VERSION = "alert_policy_v1.0.0"

OPEN_STATES = frozenset({AlertState.CREATED, AlertState.SENT})
# States that still count as "the patient has been told" for deduplication.
BLOCKING_STATES = OPEN_STATES | {AlertState.ACKNOWLEDGED}


class AlertDecision(StrEnum):
    CREATED = "created"
    INVALID_PREDICTION = "invalid_prediction"
    PROJECTED_HORIZON = "projected_horizon"
    STALE_PREDICTION = "stale_prediction"
    BELOW_ALERT_LEVEL = "below_alert_level"
    ALREADY_ALERTED = "already_alerted"
    EQUIVALENT_ALERT_OPEN = "equivalent_alert_open"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class PredictionForAlert:
    prediction_id: object
    horizon_hours: int
    risk_probability: float
    risk_level: RiskLevel
    prediction_time: datetime
    target_window_end: datetime


@dataclass(frozen=True)
class ExistingAlert:
    id: object
    prediction_id: object
    state: AlertState
    risk_level: RiskLevel
    expires_at: datetime


@dataclass(frozen=True)
class AlertEvaluation:
    decision: AlertDecision
    # open alerts to cancel, with the reason
    cancel: dict[object, str] = field(default_factory=dict)

    @property
    def create(self) -> bool:
        return self.decision == AlertDecision.CREATED


def effective_state(state: AlertState | str, expires_at: datetime, at: datetime) -> AlertState:
    """An open alert whose window has ended reads as expired."""
    state = AlertState(state)
    if state in OPEN_STATES and expires_at <= at:
        return AlertState.EXPIRED
    return state


def evaluate_alert(
    prediction: PredictionForAlert,
    existing: list[ExistingAlert],
    now: datetime,
    max_age: timedelta,
) -> AlertEvaluation:
    """Pure decision for one prediction against the patient's existing alerts."""
    try:
        _validate(prediction.risk_probability)
        rank = level_rank(prediction.risk_level)
    except (InvalidProbabilityError, ValueError):
        return AlertEvaluation(AlertDecision.INVALID_PREDICTION)
    if prediction.horizon_hours != 0:
        return AlertEvaluation(AlertDecision.PROJECTED_HORIZON)
    age = now - prediction.prediction_time
    if age > max_age or age < -timedelta(minutes=1):
        return AlertEvaluation(AlertDecision.STALE_PREDICTION)

    same = [a for a in existing if a.prediction_id == prediction.prediction_id]
    if rank < level_rank(ALERT_LEVEL):
        # A recomputed prediction (e.g. after a late diary entry) that no longer meets the
        # level withdraws its own still-open alert.
        revised = {a.id: "prediction_revised" for a in same if a.state in OPEN_STATES}
        return AlertEvaluation(AlertDecision.BELOW_ALERT_LEVEL, revised)
    if same:
        return AlertEvaluation(AlertDecision.ALREADY_ALERTED)

    live = [
        a
        for a in existing
        if a.state in BLOCKING_STATES and a.expires_at > prediction.prediction_time
    ]
    if any(level_rank(a.risk_level) >= rank for a in live):
        return AlertEvaluation(AlertDecision.EQUIVALENT_ALERT_OPEN)
    superseded = {a.id: "superseded_by_higher_level" for a in live if a.state in OPEN_STATES}
    return AlertEvaluation(AlertDecision.CREATED, superseded)


def alert_message(level: RiskLevel | str, probability: float) -> str:
    """Level and percentage only: no medication or treatment instruction (BR-12)."""
    return (
        f"Your estimated respiratory risk for the next 24 hours is {RiskLevel(level).value} "
        f"({format_percent(probability)})."
    )


def format_percent(p: float) -> str:
    pct = 100 * p
    return f"{pct:.1f}%" if pct < 10 else f"{pct:.0f}%"
