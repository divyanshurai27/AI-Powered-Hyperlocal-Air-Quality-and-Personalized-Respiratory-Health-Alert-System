"""Rule-based daily air-quality guidance (GUIDANCE_POLICY_VERSION).

This is NOT the Phase 4 respiratory-risk model. It turns the 24 h pollutant forecast into
plain-language guidance using India's National Air Quality Index (NAQI) categories and their
published health statements, made stricter for people with asthma or COPD. It never says an
outing is "safe" and never mentions medication (BR-12).

Rules
- Per hour, each forecast pollutant is placed in its NAQI category; the worst one decides
  the hour's category (the "dominant" pollutant). NAQI breakpoints are defined on 24 h
  (8 h for O3) averages, so applying them to hourly forecasts is an approximation.
- Category → guidance level depends on sensitivity:
      general            Good/Satisfactory → good, Moderate → caution, Poor → limit, worse → avoid
      asthma/COPD        Good → good, Satisfactory → caution, Moderate → limit, Poor+ → avoid
      severe/very severe Good → good, Satisfactory → caution, Moderate+ → avoid
  NAQI's own statements motivate the shift: "Satisfactory" may cause minor breathing
  discomfort to sensitive people; "Moderate" causes breathing discomfort to people with lung
  disease such as asthma.
- Best window: the lowest-scoring contiguous run of waking hours (06:00–22:00 IST).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum

from app.domain.pollutants import Pollutant

GUIDANCE_POLICY_VERSION = "guidance_naqi_v1.0.0"
IST = timedelta(hours=5, minutes=30)
WAKING_HOURS = (6, 22)  # IST, [start, end)
WINDOW_HOURS = 2

# NAQI category upper bounds (inclusive), µg/m³; CO is compared in mg/m³.
NAQI_BREAKPOINTS: dict[Pollutant, tuple[float, ...]] = {
    Pollutant.PM25: (30, 60, 90, 120, 250),
    Pollutant.PM10: (50, 100, 250, 350, 430),
    Pollutant.O3: (50, 100, 168, 208, 748),
    Pollutant.NO2: (40, 80, 180, 280, 400),
    Pollutant.SO2: (40, 80, 380, 800, 1600),
    Pollutant.CO: (1, 2, 10, 17, 34),
}
CATEGORY_NAMES = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
# Paraphrased from the NAQI health statements.
CATEGORY_HEALTH = (
    "Minimal impact.",
    "May cause minor breathing discomfort to sensitive people.",
    "May cause breathing discomfort to people with lung disease such as asthma.",
    "May cause breathing discomfort to most people on prolonged exposure.",
    "May cause respiratory illness on prolonged exposure.",
    "May affect healthy people and seriously affect those with existing disease.",
)


class Level(IntEnum):
    GOOD = 0
    CAUTION = 1
    LIMIT = 2
    AVOID = 3


LEVEL_TEXT = {
    Level.GOOD: "Good time to be outdoors",
    Level.CAUTION: "Fine for most activities; ease off if you notice symptoms",
    Level.LIMIT: "Keep outdoor time short and avoid strenuous exercise",
    Level.AVOID: "Avoid prolonged or strenuous outdoor activity",
}

# category index (0..5) → level, per sensitivity tier
_LEVEL_BY_TIER = {
    "general": (Level.GOOD, Level.GOOD, Level.CAUTION, Level.LIMIT, Level.AVOID, Level.AVOID),
    "sensitive": (Level.GOOD, Level.CAUTION, Level.LIMIT, Level.AVOID, Level.AVOID, Level.AVOID),
    "high": (Level.GOOD, Level.CAUTION, Level.AVOID, Level.AVOID, Level.AVOID, Level.AVOID),
}


def sensitivity_tier(disease_type: str | None, severity: str | None) -> str:
    if disease_type is None:
        return "general"
    return "high" if severity in ("severe", "very_severe") else "sensitive"


def naqi_category(pollutant: Pollutant, concentration: float) -> int:
    """0 (Good) … 5 (Severe). `concentration` in µg/m³ (CO converted to mg/m³ here)."""
    value = concentration / 1000 if pollutant == Pollutant.CO else concentration
    for i, upper in enumerate(NAQI_BREAKPOINTS[pollutant]):
        if value <= upper:
            return i
    return 5


@dataclass(frozen=True)
class HourGuidance:
    time: datetime  # end of the forecast hour, UTC
    category: int
    dominant: Pollutant
    level: Level
    values: dict[Pollutant, float]

    @property
    def ist_hour(self) -> int:
        return (self.time + IST).hour

    @property
    def waking(self) -> bool:
        # `time` is the hour's END, so 06:00–07:00 IST has ist_hour 7 and 21:00–22:00 has 22.
        return WAKING_HOURS[0] < self.ist_hour <= WAKING_HOURS[1]


@dataclass
class DayGuidance:
    tier: str
    headline_level: Level
    headline_category: int
    hours: list[HourGuidance]
    best_window: tuple[datetime, datetime] | None
    worst_window: tuple[datetime, datetime] | None = None
    avoid_windows: list[tuple[datetime, datetime]] = field(default_factory=list)
    policy_version: str = GUIDANCE_POLICY_VERSION


def hourly_guidance(
    forecasts: dict[Pollutant, list[tuple[datetime, float]]], tier: str
) -> list[HourGuidance]:
    """Combine per-pollutant hourly forecasts into one guidance entry per hour."""
    by_time: dict[datetime, dict[Pollutant, float]] = {}
    for pollutant, series in forecasts.items():
        for t, v in series:
            by_time.setdefault(t, {})[pollutant] = v
    out = []
    for t in sorted(by_time):
        values = by_time[t]
        cats = {p: naqi_category(p, v) for p, v in values.items()}
        # Worst category wins; ties go to the pollutant listed first (PM2.5 before others).
        dominant = max(cats, key=lambda p: (cats[p], -list(Pollutant).index(p)))
        cat = cats[dominant]
        out.append(HourGuidance(t, cat, dominant, _LEVEL_BY_TIER[tier][cat], values))
    return out


def _score(h: HourGuidance) -> float:
    # Level first, then how far into its NAQI category the dominant pollutant is.
    bp = NAQI_BREAKPOINTS[h.dominant]
    v = h.values[h.dominant] / (1000 if h.dominant == Pollutant.CO else 1)
    lo = 0 if h.category == 0 else bp[h.category - 1]
    hi = bp[h.category] if h.category < len(bp) else lo * 2
    frac = min(1.0, max(0.0, (v - lo) / (hi - lo))) if hi > lo else 0.0
    return h.level * 10 + h.category + frac


def best_window(
    hours: list[HourGuidance], length: int = WINDOW_HOURS, worst: bool = False
) -> tuple[datetime, datetime] | None:
    """Lowest (worst=True: highest) mean score over `length` consecutive waking hours.
    Returns (start, end) UTC."""
    waking = [h for h in hours if h.waking]
    best, best_score = None, float("inf")
    for i in range(len(waking) - length + 1):
        run = waking[i : i + length]
        if any(run[j + 1].time - run[j].time != timedelta(hours=1) for j in range(length - 1)):
            continue
        score = sum(_score(h) for h in run) / length
        score = -score if worst else score
        if score < best_score:
            best, best_score = run, score
    if best is None:
        return None
    return best[0].time - timedelta(hours=1), best[-1].time


def avoid_windows(
    hours: list[HourGuidance], min_level: Level = Level.LIMIT
) -> list[tuple[datetime, datetime]]:
    """Contiguous waking runs at or above `min_level` AND worse than the best waking hour.

    When the whole day sits at one level there is no better time to switch to, so nothing
    is flagged here; the worst stretch is reported separately as `worst_window`.
    """
    floor = min((h.level for h in hours if h.waking), default=Level.GOOD)
    runs: list[list[HourGuidance]] = []
    for h in hours:
        if not (h.waking and h.level >= min_level and h.level > floor):
            continue
        if runs and h.time - runs[-1][-1].time == timedelta(hours=1):
            runs[-1].append(h)
        else:
            runs.append([h])
    return [(r[0].time - timedelta(hours=1), r[-1].time) for r in runs]


def day_guidance(
    forecasts: dict[Pollutant, list[tuple[datetime, float]]],
    disease_type: str | None,
    severity: str | None,
) -> DayGuidance:
    tier = sensitivity_tier(disease_type, severity)
    hours = hourly_guidance(forecasts, tier)
    waking = [h for h in hours if h.waking] or hours
    # Headline is the worst waking hour: conservative on purpose.
    worst = max(waking, key=_score)
    return DayGuidance(
        tier=tier,
        headline_level=worst.level,
        headline_category=worst.category,
        hours=hours,
        best_window=best_window(hours),
        worst_window=best_window(hours, worst=True),
        avoid_windows=avoid_windows(hours),
    )
