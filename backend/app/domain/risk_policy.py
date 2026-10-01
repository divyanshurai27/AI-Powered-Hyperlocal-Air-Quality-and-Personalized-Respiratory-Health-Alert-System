"""Risk-level policy and patient personalization (PRD §15.3, §16, BR-01, BR-02).

The model outputs a calibrated probability of a respiratory event in the next 24 h. What
counts as Low / Moderate / High / Severe is a *policy* over that probability, stored here
under RISK_POLICY_VERSION, so thresholds can change without retraining (PRD §15.3).

Thresholds are per disease because base rates differ. They were fixed from the VALIDATION
period of the synthetic cohort (ml/risk/train_risk_model.py prints the derivation) and
never tuned on the test period:
    Moderate  p ≥ ~1.5 × the disease's base rate (clearly above that person's usual day)
    High      the F1-optimal alert threshold on validation
    Severe    the top 1% of validation probabilities for that disease (precision 0.5 is
              never reached at a ~1.5% base rate, so a precision target is not usable)
They describe the *synthetic* cohort; a real deployment would have to re-derive them.

Personalization (PERSONALIZATION_VERSION) is a research configuration, not a clinical
instruction (PRD §16 guardrail). The only adaptive rule is a bounded personal offset:

    offset = log((O + k) / (E + k)),  clipped to ±MAX_OFFSET
    p_personal = sigmoid(logit(p) + offset)

O = exacerbations the patient reported in their diary before the prediction time, E = the
number expected over the same diary days at their disease/severity stratum's rate of
reported exacerbations per diary day, k = prior strength. Counting per diary day (not per
calendar day) keeps a patient who skips the diary from looking healthier than they are.
With little history both O and E are small and the offset shrinks toward 0, so new users
get the unpersonalized probability. Only records dated before the prediction time count.
"""

import math
from dataclasses import dataclass

from app.domain.enums import DiseaseType, RiskLevel

RISK_POLICY_VERSION = "risk_policy_v1.0.0"
PERSONALIZATION_VERSION = "personal_offset_v1.0.0"

# Lower bounds (inclusive) of Moderate, High, Severe; below the first is Low.
RISK_THRESHOLDS: dict[DiseaseType, tuple[float, float, float]] = {
    DiseaseType.ASTHMA: (0.025, 0.048, 0.077),
    DiseaseType.COPD: (0.021, 0.052, 0.080),
}
# Level at or above which a prediction is an "alert" in the personalization comparison.
ALERT_LEVEL = RiskLevel.HIGH

PRIOR_STRENGTH_EVENTS = 2.0
MAX_OFFSET = math.log(2.0)  # personalization can at most double or halve the odds
PERSONAL_HISTORY_DAYS = 365

_ORDER = (RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.SEVERE)


class InvalidProbabilityError(ValueError):
    pass


def validate_probability(p: float) -> float:
    """BR-01: anything outside [0, 1] (or NaN) is an invalid model result."""
    if not isinstance(p, int | float) or math.isnan(p) or not 0.0 <= p <= 1.0:
        raise InvalidProbabilityError(f"risk probability {p!r} is outside [0, 1]")
    return float(p)


def classify_risk(probability: float, disease_type: DiseaseType | str) -> RiskLevel:
    p = validate_probability(probability)
    moderate, high, severe = RISK_THRESHOLDS[DiseaseType(disease_type)]
    if p >= severe:
        return RiskLevel.SEVERE
    if p >= high:
        return RiskLevel.HIGH
    if p >= moderate:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def level_rank(level: RiskLevel | str) -> int:
    return _ORDER.index(RiskLevel(level))


def is_alert(level: RiskLevel | str) -> bool:
    return level_rank(level) >= level_rank(ALERT_LEVEL)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass(frozen=True)
class PersonalBaseline:
    observed_events: int
    expected_events: float
    observed_days: float
    offset: float
    version: str = PERSONALIZATION_VERSION


def estimate_personal_baseline(
    observed_events: int, observed_days: float, stratum_daily_rate: float
) -> PersonalBaseline:
    """Shrunk observed/expected ratio of the patient's own event history (PRD §16).

    `observed_days` = diary days (records answering the exacerbation question) dated before
    the prediction time, within PERSONAL_HISTORY_DAYS; `stratum_daily_rate` = reported
    exacerbations per diary day in the stratum's training data.
    """
    if observed_events < 0 or observed_days < 0 or stratum_daily_rate < 0:
        raise ValueError("history counts and rates cannot be negative")
    expected = stratum_daily_rate * observed_days
    k = PRIOR_STRENGTH_EVENTS
    raw = math.log((observed_events + k) / (expected + k))
    offset = max(-MAX_OFFSET, min(MAX_OFFSET, raw))
    return PersonalBaseline(observed_events, round(expected, 4), observed_days, round(offset, 6))


def apply_personal_offset(probability: float, baseline: PersonalBaseline | None) -> float:
    p = validate_probability(probability)
    if baseline is None or baseline.offset == 0:
        return p
    return _sigmoid(_logit(p) + baseline.offset)
