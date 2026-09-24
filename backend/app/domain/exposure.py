"""Personal exposure ESTIMATE (PRD §13). Not a measurement: no personal sensor exists in the MVP.

Model (versioned as EXPOSURE_MODEL_VERSION):

    exposure_t[p] = ambient_t[p at the user's location] × F[microenvironment_t, p]

`ambient` comes from hyperlocal matching (domain/spatial). F is a microenvironment factor:
the typical ratio of concentration in that setting to the outdoor ambient level. Values below
are central estimates chosen from ranges reported in indoor/outdoor (I/O) and in-transit
studies of naturally ventilated South-Asian buildings and urban traffic. They are ASSUMPTIONS,
not calibrated to AeroGard users; `sensitivity_factors` exists so every result can be
re-run with them perturbed, and the paper must report that sensitivity.

Rules:
- unknown microenvironment → F = 1.0 (ambient), and the hour is marked `assumed_outdoor`
- no valid location for an hour → that hour's exposure is missing, never guessed (BR-07)
- window statistics report their coverage; < MIN_WINDOW_COVERAGE marks them incomplete
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from app.domain.pollutants import Pollutant

EXPOSURE_MODEL_VERSION = "exposure_io_v1.0.0"
MIN_WINDOW_COVERAGE = 0.75


class MicroEnvironment(StrEnum):
    OUTDOOR = "outdoor"
    INDOOR_HOME = "indoor_home"
    INDOOR_OTHER = "indoor_other"  # office, school, mall: often mechanically ventilated
    IN_TRANSIT = "in_transit"  # road travel, close to traffic emissions
    UNKNOWN = "unknown"


# F[microenvironment][pollutant]. PM10 infiltrates less than PM2.5 (coarse particles settle);
# O3 is reactive and depletes indoors; NO2/CO are elevated near traffic.
MICROENVIRONMENT_FACTORS: dict[MicroEnvironment, dict[Pollutant, float]] = {
    MicroEnvironment.OUTDOOR: {p: 1.0 for p in Pollutant},
    MicroEnvironment.INDOOR_HOME: {
        Pollutant.PM25: 0.8,
        Pollutant.PM10: 0.6,
        Pollutant.NO2: 0.6,
        Pollutant.O3: 0.3,
        Pollutant.SO2: 0.5,
        Pollutant.CO: 0.9,
    },
    MicroEnvironment.INDOOR_OTHER: {
        Pollutant.PM25: 0.6,
        Pollutant.PM10: 0.5,
        Pollutant.NO2: 0.5,
        Pollutant.O3: 0.2,
        Pollutant.SO2: 0.4,
        Pollutant.CO: 0.8,
    },
    MicroEnvironment.IN_TRANSIT: {
        Pollutant.PM25: 1.3,
        Pollutant.PM10: 1.3,
        Pollutant.NO2: 1.6,
        Pollutant.O3: 0.8,
        Pollutant.SO2: 1.1,
        Pollutant.CO: 1.8,
    },
    MicroEnvironment.UNKNOWN: {p: 1.0 for p in Pollutant},
}


@dataclass(frozen=True)
class HourlyExposure:
    hour: datetime
    pollutant: Pollutant
    ambient: float | None
    factor: float | None
    exposure: float | None
    microenvironment: MicroEnvironment | None
    assumed_outdoor: bool = False
    missing_reason: str | None = None  # "location_unavailable" | "ambient_unavailable"


@dataclass(frozen=True)
class WindowStat:
    window_hours: int
    mean: float | None
    maximum: float | None
    cumulative: float | None  # µg/m³·h over the hours that have data
    coverage: float  # fraction of hours in the window with an estimate
    complete: bool


@dataclass
class ExposureProfile:
    pollutant: Pollutant
    model_version: str
    hours: list[HourlyExposure]
    windows: dict[int, WindowStat] = field(default_factory=dict)
    lagged: dict[str, float | None] = field(default_factory=dict)


def microenvironment_factor(
    env: MicroEnvironment, pollutant: Pollutant, factors: dict | None = None
) -> float:
    table = factors or MICROENVIRONMENT_FACTORS
    return table[env][pollutant]


def estimate_hourly_exposure(
    hour: datetime,
    pollutant: Pollutant,
    ambient: float | None,
    microenvironment: MicroEnvironment | None,
    *,
    location_available: bool = True,
    factors: dict | None = None,
) -> HourlyExposure:
    if not location_available:
        return HourlyExposure(
            hour, pollutant, None, None, None, None, missing_reason="location_unavailable"
        )
    if ambient is None or not math.isfinite(ambient):
        return HourlyExposure(
            hour,
            pollutant,
            None,
            None,
            None,
            microenvironment,
            missing_reason="ambient_unavailable",
        )
    env = microenvironment or MicroEnvironment.UNKNOWN
    f = microenvironment_factor(env, pollutant, factors)
    return HourlyExposure(
        hour=hour,
        pollutant=pollutant,
        ambient=ambient,
        factor=f,
        exposure=round(ambient * f, 3),
        microenvironment=env,
        assumed_outdoor=env == MicroEnvironment.UNKNOWN,
    )


def calculate_cumulative_exposure(
    hours: list[HourlyExposure], window_hours: int, end: datetime
) -> WindowStat:
    """Stats over (end - window, end]. Only hours that exist in `hours` count as covered."""
    start = end - timedelta(hours=window_hours)
    values = [h.exposure for h in hours if start < h.hour <= end and h.exposure is not None]
    coverage = len(values) / window_hours
    if not values:
        return WindowStat(window_hours, None, None, None, 0.0, False)
    return WindowStat(
        window_hours=window_hours,
        mean=round(sum(values) / len(values), 3),
        maximum=round(max(values), 3),
        cumulative=round(sum(values), 3),
        coverage=round(coverage, 4),
        complete=coverage >= MIN_WINDOW_COVERAGE,
    )


def calculate_lagged_exposure(
    hours: list[HourlyExposure], end: datetime
) -> dict[str, float | None]:
    """Point lags and the recent change, all from hours ≤ end (BR-04)."""
    by_hour = {h.hour: h.exposure for h in hours if h.hour <= end}

    def at(lag: int) -> float | None:
        return by_hour.get(end - timedelta(hours=lag))

    lag0, lag24 = at(0), at(24)
    return {
        "lag_0h": lag0,
        "lag_1h": at(1),
        "lag_3h": at(3),
        "lag_24h": lag24,
        "change_vs_24h": round(lag0 - lag24, 3) if lag0 is not None and lag24 is not None else None,
    }


def build_exposure_profile(
    pollutant: Pollutant,
    hours: list[HourlyExposure],
    end: datetime,
    windows: tuple[int, ...] = (6, 24, 72),
) -> ExposureProfile:
    ordered = sorted(hours, key=lambda h: h.hour)
    return ExposureProfile(
        pollutant=pollutant,
        model_version=EXPOSURE_MODEL_VERSION,
        hours=ordered,
        windows={w: calculate_cumulative_exposure(ordered, w, end) for w in windows},
        lagged=calculate_lagged_exposure(ordered, end),
    )


def sensitivity_factors(scale: float, only: MicroEnvironment | None = None) -> dict:
    """Factor table with non-outdoor factors multiplied by `scale` (e.g. 0.8 / 1.2), for the
    sensitivity analysis required by PRD §13. Outdoor stays 1.0 by definition."""
    out = {}
    for env, row in MICROENVIRONMENT_FACTORS.items():
        perturb = env not in (MicroEnvironment.OUTDOOR, MicroEnvironment.UNKNOWN) and (
            only is None or env == only
        )
        out[env] = {p: (f * scale if perturb else f) for p, f in row.items()}
    return out
