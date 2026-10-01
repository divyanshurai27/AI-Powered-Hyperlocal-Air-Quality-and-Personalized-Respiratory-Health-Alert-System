"""SYNTHETIC respiratory cohort for developing and evaluating the risk model (PRD §3.2).

No real respiratory outcomes are available for Bengaluru, so the risk layer is developed on
a simulated panel of asthma/COPD patients. Everything the paper reports about the risk model
is therefore a *synthetic-data evaluation*, not clinical validation, and must say so.

What is real and what is simulated
- REAL: hourly PM2.5/PM10/O3 at the 14 CPCB/KSPCB monitors and ERA5 weather. Simulated
  patients live and move inside the real pollution field of 2025-02 → 2026-09.
- SIMULATED: the patients (profiles, homes, workplaces, daily routines), their personal
  exposure, respiratory infections, symptom diaries, wearable readings and events.

Data-generating process (COHORT_VERSION; every constant lives in CohortConfig)
1. Patients: disease, severity, age, sex, home/work in the city, work schedule, walking
   habit. Hidden traits: frailty u ~ N(0, σ_u), pollution sensitivity s ~ LogNormal with
   mean 1, and personal microenvironment factors F_true = F_table × LogNormal(0, σ_F)
   (housing and ventilation differ between people).
2. Hourly routine → microenvironment (home / work indoors / transit / outdoors) and place.
3. TRUE personal exposure = true ambient at the place × F_true[microenvironment]. True
   ambient is an IDW field over every station with a value, times a smooth AR(1) noise
   (σ_amb) for street-scale variation the monitors cannot see.
4. What the SYSTEM sees mirrors production: location fixes are logged with gaps, the
   activity context is sometimes "unknown", and ambient comes from the live spatial rule
   (radius 10 km, ≤ 4 stations, readings ≤ 3 h old). So estimated exposure ≠ true exposure.
5. Hourly event hazard (event = respiratory exacerbation episode: a symptom flare needing
   extra reliever or unscheduled care):
       log λ = c_stratum + u + s·β25·(Ē25 − 25)/10 + β10·(Ē10 − 50)/10 + βO3·(ĒO3 − 60)/10
               + βcold·max(0, 18 − T)/5 + βhum·[RH > 85] + βinf·infection + βctl·control
               + βrecent·[event in last 14 d] + asthma night term
   Ē are exponential moving averages of TRUE personal exposure (half-life 24 h for PM, 8 h
   for O3). All pollution terms are multiplied by `aq_effect_scale` (0 = negative control:
   pollution has no effect at all). c_stratum is solved so each disease/severity stratum
   has its configured annual rate. `control` is a latent daily AR(1) state (unit variance)
   for day-to-day loss of disease control: exacerbations are typically preceded by days of
   worsening symptoms and reliever use, and the diary is where that shows. After an onset
   the episode lasts `episode_hours`, with no new onset in that time.
6. Daily diary at 21:00 IST covering the preceding 24 h: symptom score 0–10 and rescue
   puffs (noisy readouts of control, infection, pollution response and any ongoing episode), and an
   exacerbation flag. Filled with the patient's adherence probability. A subset wears a
   device giving SpO2, resting heart rate, steps and sleep.

Effect sizes are ASSUMPTIONS chosen for a plausible, learnable signal (panel studies report
symptom-flare odds ratios of roughly 1.05–1.15 per 10 µg/m³ PM2.5; viral infection is the
dominant trigger). They are not calibrated to Indian epidemiology.
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from app.domain.exposure import MICROENVIRONMENT_FACTORS, MicroEnvironment
from app.domain.pollutants import Pollutant
from app.domain.spatial import DEGRADED_QUALITY_WEIGHT, EARTH_RADIUS_KM, IDW_POWER, MIN_DISTANCE_KM

COHORT_VERSION = "cohort_v1.0.0"

COHORT_POLLUTANTS = (Pollutant.PM25, Pollutant.PM10, Pollutant.O3)
IST = pd.Timedelta(hours=5, minutes=30)
DIARY_HOUR_UTC = pd.Timedelta(hours=15, minutes=30)  # 21:00 IST

# Integer codes for the hourly microenvironment arrays.
ENV_CODES = (
    MicroEnvironment.OUTDOOR,
    MicroEnvironment.INDOOR_HOME,
    MicroEnvironment.INDOOR_OTHER,
    MicroEnvironment.IN_TRANSIT,
    MicroEnvironment.UNKNOWN,
)
OUTDOOR, HOME, OTHER, TRANSIT, UNKNOWN = range(5)
PLACE_HOME, PLACE_WORK, PLACE_MID = range(3)

SEVERITIES = ("mild", "moderate", "severe", "very_severe")


@dataclass
class CohortConfig:
    n_patients: int = 400
    seed: int = 2026
    aq_effect_scale: float = 1.0
    copd_fraction: float = 0.4
    city_center: tuple[float, float] = (12.9716, 77.5946)
    home_radius_km: float = 12.0
    work_radius_km: float = 14.0
    # Annual event rates per (disease, severity).
    event_rate_per_year: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {"asthma": (4.0, 6.0, 9.0, 12.0), "copd": (3.0, 4.0, 6.0, 8.0)}
    )
    severity_probs: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "asthma": (0.40, 0.35, 0.18, 0.07),
            "copd": (0.25, 0.40, 0.25, 0.10),
        }
    )
    frailty_sd: float = 0.5
    sensitivity_log_sd: float = 0.7
    factor_log_sd: float = 0.25
    ambient_noise_sd: float = 0.15
    ambient_noise_rho: float = 0.97  # hourly AR(1) coefficient
    # Hazard log-coefficients.
    beta_pm25_per10: float = math.log(1.10)
    beta_pm10_per10: float = math.log(1.03)
    beta_o3_per10: float = math.log(1.04)
    beta_cold_per5c: float = math.log(1.15)
    beta_humid: float = math.log(1.20)
    beta_infection: float = math.log(8.0)
    beta_control_per_sd: float = math.log(2.2)
    control_rho_daily: float = 0.85
    beta_recent_event: float = math.log(1.8)
    beta_asthma_night: float = math.log(1.5)
    pm_halflife_hours: float = 24.0
    o3_halflife_hours: float = 8.0
    episode_hours: int = 72
    recent_event_days: int = 14
    infections_per_year: float = 4.0
    # Monsoon (Jun–Sep) and winter (Dec–Feb) respiratory-virus seasons.
    infection_season_multiplier: dict[int, float] = field(
        default_factory=lambda: {
            **dict.fromkeys((6, 7, 8, 9), 1.6),
            **dict.fromkeys((12, 1, 2), 1.3),
            **dict.fromkeys((3, 4, 5, 10, 11), 0.7),
        }
    )
    # Observation process (what the app would actually receive).
    location_log_beta: tuple[float, float] = (8.0, 1.5)
    location_day_dropout: float = 0.05
    context_known_prob: float = 0.9
    diary_adherence_beta: tuple[float, float] = (6.0, 2.0)
    wearable_fraction: float = 0.4
    wearable_wear_prob: float = 0.9
    # Live spatial rule used for the system's view (mirrors app/core/config.py defaults).
    est_radius_km: float = 10.0
    est_max_stations: int = 4
    est_exact_km: float = 0.5
    est_max_age_hours: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Cohort:
    config: CohortConfig
    hours: pd.DatetimeIndex  # UTC AQ hour ends (:30)
    patients: pd.DataFrame  # what a profile would hold (no hidden traits)
    truth: pd.DataFrame  # hidden traits: never features
    # (P, H) arrays
    amb_est: dict[str, np.ndarray]  # ambient at the logged location, live rule; NaN = none
    amb_est_home: dict[str, np.ndarray]  # ambient at home, live rule (sensitivity analysis)
    true_exposure: dict[str, np.ndarray]
    env_reported: np.ndarray  # int8 ENV_CODES index; UNKNOWN when context not known
    location_available: np.ndarray  # bool: a fix ≤ 2 h old exists (BR-07)
    infection: np.ndarray
    onset: np.ndarray  # bool: event starts in this hour
    diary: pd.DataFrame  # long: patient_id, timestamp, symptom_score, ...
    weather: pd.DataFrame  # hourly on `hours`: temperature, relative_humidity, precipitation
    version: str = COHORT_VERSION

    def summary(self) -> dict:
        years = len(self.hours) / 8760
        events = self.onset.sum(axis=1)
        out = {
            "version": self.version,
            "patients": len(self.patients),
            "start": str(self.hours[0]),
            "end": str(self.hours[-1]),
            "events": int(events.sum()),
            "events_per_patient_year": round(float(events.mean() / years), 3),
            "aq_effect_scale": self.config.aq_effect_scale,
        }
        for d, grp in self.patients.groupby("disease_type"):
            out[f"{d}_patients"] = len(grp)
            out[f"{d}_events_per_year"] = round(float(events[grp.index].mean() / years), 3)
        out["diary_rows"] = len(self.diary)
        out["diary_symptom_completeness"] = round(
            float(self.diary["symptom_score"].notna().sum() / (len(self.patients) * years * 365)),
            3,
        )
        out["location_availability"] = round(float(self.location_available.mean()), 3)
        return out


# --- geometry and spatial fields ---------------------------------------------------------


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def _offset_point(center: tuple[float, float], km: np.ndarray, bearing: np.ndarray):
    """Points `km` from `center` along `bearing` (radians); flat-earth is fine at city scale."""
    lat0, lon0 = center
    dlat = km * np.cos(bearing) / 111.32
    dlon = km * np.sin(bearing) / (111.32 * math.cos(math.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def idw_field(
    values: np.ndarray,
    quality_weight: np.ndarray,
    distances_km: np.ndarray,
    *,
    radius_km: float,
    max_stations: int,
    exact_km: float,
) -> np.ndarray:
    """Vectorised twin of domain.spatial.estimate_local_air_quality for one point.

    values (H, S) with NaN = no usable reading at that hour; quality_weight (H, S) 1.0 for
    VALID, DEGRADED_QUALITY_WEIGHT otherwise; distances_km (S,). Returns (H,), NaN where no
    station is usable. A unit test checks it against the domain function.
    """
    order = np.argsort(distances_km, kind="stable")
    order = order[distances_km[order] <= radius_km]
    if order.size == 0:
        return np.full(values.shape[0], np.nan)
    v = values[:, order]
    q = quality_weight[:, order]
    d = distances_km[order]
    usable = ~np.isnan(v)
    chosen = usable & (np.cumsum(usable, axis=1) <= max_stations)
    first = np.argmax(usable, axis=1)
    any_usable = usable.any(axis=1)
    exact = any_usable & (d[first] <= exact_km)
    w = np.where(chosen, q / np.maximum(d, MIN_DISTANCE_KM) ** IDW_POWER, 0.0)
    w[exact] = 0.0
    w[exact, first[exact]] = 1.0
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, (w * np.nan_to_num(v)).sum(axis=1) / total, np.nan)
    return out


def station_matrices(
    frames: dict[str, pd.DataFrame], hours: pd.DatetimeIndex, pollutant: Pollutant
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(station_ids, cleaned (H,S), observed-live (H,S), quality weight live (H,S)).

    `cleaned` is the gap-limited cleaned series (for the true field). The live view is the
    last real reading carried forward at most `est_max_age_hours`, as the API would see it.
    """
    p = pollutant.value
    ids = sorted(s for s, f in frames.items() if p in f)
    cleaned = np.full((len(hours), len(ids)), np.nan)
    live = np.full_like(cleaned, np.nan)
    qual = np.ones_like(cleaned)
    for j, sid in enumerate(ids):
        f = frames[sid].reindex(hours)
        cleaned[:, j] = f[p].to_numpy(float)
        observed = f[f"{p}__observed"].fillna(False).astype(bool).to_numpy()
        valid = f[f"{p}__valid"].fillna(False).astype(bool).to_numpy()
        obs_vals = pd.Series(np.where(observed, cleaned[:, j], np.nan))
        q = pd.Series(np.where(observed, np.where(valid, 1.0, DEGRADED_QUALITY_WEIGHT), np.nan))
        live[:, j] = obs_vals.ffill(limit=3).to_numpy()
        qual[:, j] = q.ffill(limit=3).fillna(1.0).to_numpy()
    return ids, cleaned, live, qual


def _fill_field(x: np.ndarray, hours: pd.DatetimeIndex) -> np.ndarray:
    """Close remaining holes in the TRUE field (the DGP needs a value every hour): carry
    forward/back up to 12 h, then the hour-of-day × month median of the field itself."""
    s = pd.Series(x, index=hours).ffill(limit=12).bfill(limit=12)
    if s.isna().any():
        local = hours + IST
        clim = s.groupby([local.month, local.hour]).transform("median")
        s = s.fillna(clim).fillna(s.median())
    return s.to_numpy()


# --- patients and routines ---------------------------------------------------------------


def _sample_patients(
    cfg: CohortConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = cfg.n_patients
    copd = rng.random(n) < cfg.copd_fraction
    disease = np.where(copd, "copd", "asthma")
    sev_idx = np.array([rng.choice(4, p=cfg.severity_probs[d]) for d in disease], dtype=int)
    age = np.where(
        copd, rng.normal(62, 9, n).clip(40, 85), rng.normal(35, 14, n).clip(12, 75)
    ).round()
    female_p = np.where(copd, 0.35, 0.55)
    sex = np.where(rng.random(n) < female_p, "female", "male")
    home_lat, home_lon = _offset_point(
        cfg.city_center, cfg.home_radius_km * np.sqrt(rng.random(n)), rng.random(n) * 2 * np.pi
    )
    employed = rng.random(n) < np.where(age < 60, np.where(copd, 0.5, 0.7), 0.1)
    work_lat, work_lon = _offset_point(
        cfg.city_center, cfg.work_radius_km * np.sqrt(rng.random(n)), rng.random(n) * 2 * np.pi
    )
    patients = pd.DataFrame(
        {
            "patient_id": [f"syn-{i:04d}" for i in range(n)],
            "age": age.astype(int),
            "sex": sex,
            "disease_type": disease,
            "severity": [SEVERITIES[i] for i in sev_idx],
            "severity_idx": sev_idx,
            "home_latitude": home_lat,
            "home_longitude": home_lon,
            "employed": employed,
            "work_latitude": np.where(employed, work_lat, np.nan),
            "work_longitude": np.where(employed, work_lon, np.nan),
            "depart_hour_ist": rng.integers(7, 11, n),
            "return_hour_ist": rng.integers(17, 21, n),
            "walker": rng.random(n) < 0.4,
            "walk_hour_ist": rng.integers(6, 8, n),
            "has_wearable": rng.random(n) < cfg.wearable_fraction,
        }
    )
    s_sd = cfg.sensitivity_log_sd
    truth = pd.DataFrame(
        {
            "frailty": rng.normal(0, cfg.frailty_sd, n),
            "sensitivity": np.exp(rng.normal(-(s_sd**2) / 2, s_sd, n)),
            "diary_adherence": rng.beta(*cfg.diary_adherence_beta, n),
            "location_log_rate": rng.beta(*cfg.location_log_beta, n),
            **{
                f"factor_mult_{env.value}": np.exp(rng.normal(0, cfg.factor_log_sd, n))
                for env in (
                    MicroEnvironment.INDOOR_HOME,
                    MicroEnvironment.INDOOR_OTHER,
                    MicroEnvironment.IN_TRANSIT,
                )
            },
        }
    )
    return patients, truth


def _routine(
    pt: pd.Series, hours: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Hourly (env code, place code) for one patient. Hour label = the hour's START in IST."""
    local = hours + IST - pd.Timedelta(hours=1)
    hr = local.hour.to_numpy()
    day = (local.normalize() - local.normalize()[0]).days.to_numpy()
    weekend = local.dayofweek.to_numpy() >= 5
    n_days = int(day.max()) + 1
    env = np.full(len(hours), HOME, dtype=np.int8)
    place = np.full(len(hours), PLACE_HOME, dtype=np.int8)

    if pt.employed:
        wfh = rng.random(n_days) < 0.15
        workday = ~weekend & ~wfh[day]
        dep, ret = int(pt.depart_hour_ist), int(pt.return_hour_ist)
        commute = workday & ((hr == dep) | (hr == ret))
        at_work = workday & (hr > dep) & (hr < ret)
        env[commute], place[commute] = TRANSIT, PLACE_MID
        env[at_work], place[at_work] = OTHER, PLACE_WORK
        errand_day = weekend & (rng.random(n_days) < 0.3)[day]
    else:
        errand_day = (rng.random(n_days) < 0.3)[day]
    errand = errand_day & ((hr == 11) | (hr == 14))
    shop = errand_day & ((hr == 12) | (hr == 13))
    env[errand], env[shop] = TRANSIT, OTHER
    evening_out = weekend & (rng.random(n_days) < 0.4)[day] & ((hr == 17) | (hr == 18))
    env[evening_out] = OUTDOOR
    if pt.walker:
        walk = (rng.random(n_days) < 0.8)[day] & (hr == int(pt.walk_hour_ist))
        env[walk] = OUTDOOR
    return env, place


def _ar1_lognormal(n: int, sd: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0, sd * math.sqrt(1 - rho**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return np.exp(x - sd**2 / 2)


def _ema(x: np.ndarray, halflife_hours: float) -> np.ndarray:
    """Exponential moving average along the last axis (hours)."""
    a = 1 - 0.5 ** (1 / halflife_hours)
    out = np.empty_like(x)
    out[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        out[..., t] = out[..., t - 1] + a * (x[..., t] - out[..., t - 1])
    return out


# --- generation ---------------------------------------------------------------------------


def generate_cohort(
    frames: dict[str, pd.DataFrame],
    weather: pd.DataFrame,
    station_coords: dict[str, tuple[float, float]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: CohortConfig | None = None,
) -> Cohort:
    cfg = config or CohortConfig()
    streams = np.random.SeedSequence(cfg.seed).spawn(8)
    rng_pat, rng_route, rng_obs, rng_inf, rng_event, rng_diary, rng_noise, _ = (
        np.random.default_rng(s) for s in streams
    )
    first = pd.Timestamp(start).tz_convert("UTC").floor("h") + pd.Timedelta(minutes=30)
    hours = pd.date_range(first, pd.Timestamp(end).tz_convert("UTC"), freq="h")
    H = len(hours)
    patients, truth = _sample_patients(cfg, rng_pat)
    P = len(patients)

    # --- routines, true exposure and what the app logs (one patient at a time) ---
    home = patients[["home_latitude", "home_longitude"]].to_numpy()
    work = patients[["work_latitude", "work_longitude"]].to_numpy()
    work = np.where(np.isnan(work), home, work)
    places = np.stack([home, work, (home + work) / 2], axis=1)  # (P, 3, 2)
    stations = {}
    for pol in COHORT_POLLUTANTS:
        ids, cleaned, live, qual = station_matrices(frames, hours, pol)
        coords = np.array([station_coords[s] for s in ids]).reshape(-1, 2)
        stations[pol.value] = (cleaned, live, qual, coords)

    local_day = ((hours + IST).normalize() - (hours + IST).normalize()[0]).days.to_numpy()
    n_days = int(local_day.max()) + 1
    f_mult = np.ones((P, len(ENV_CODES)))
    for e_idx, e in enumerate(ENV_CODES):
        col = f"factor_mult_{e.value}"
        if col in truth:
            f_mult[:, e_idx] = truth[col].to_numpy()

    shape = (P, H)
    true_exposure = {p.value: np.empty(shape, np.float32) for p in COHORT_POLLUTANTS}
    amb_est = {p.value: np.empty(shape, np.float32) for p in COHORT_POLLUTANTS}
    amb_est_home = {p.value: np.empty(shape, np.float32) for p in COHORT_POLLUTANTS}
    env_reported = np.empty(shape, np.int8)
    location_available = np.empty(shape, bool)
    arange = np.arange(H)
    for i in range(P):
        env, place = _routine(patients.iloc[i], hours, rng_route)
        logged = (rng_obs.random(H) < truth["location_log_rate"].iat[i]) & ~(
            rng_obs.random(n_days) < cfg.location_day_dropout
        )[local_day]
        # BR-07: an hour is located if a fix <= 2 h old exists; use that fix's place/context.
        last = np.maximum.accumulate(np.where(logged, arange, -1))
        located = (last >= 0) & (arange - last <= 2)
        src = np.clip(last, 0, None)
        known = rng_obs.random(H) < cfg.context_known_prob
        env_reported[i] = np.where(known, env[src], UNKNOWN)
        location_available[i] = located
        noise = _ar1_lognormal(H, cfg.ambient_noise_sd, cfg.ambient_noise_rho, rng_noise)
        for pol in COHORT_POLLUTANTS:
            cleaned, live, qual, coords = stations[pol.value]
            true_at = np.empty((3, H))
            live_at = np.empty((3, H))
            for k in range(3):
                d = _haversine_km(places[i, k, 0], places[i, k, 1], coords[:, 0], coords[:, 1])
                t = idw_field(
                    cleaned,
                    np.ones_like(cleaned),
                    d,
                    radius_km=np.inf,
                    max_stations=len(d),
                    exact_km=cfg.est_exact_km,
                )
                true_at[k] = _fill_field(t, hours) * noise
                live_at[k] = idw_field(
                    live,
                    qual,
                    d,
                    radius_km=cfg.est_radius_km,
                    max_stations=cfg.est_max_stations,
                    exact_km=cfg.est_exact_km,
                )
            factor = np.array([MICROENVIRONMENT_FACTORS[e][pol] for e in ENV_CODES]) * f_mult[i]
            true_exposure[pol.value][i] = true_at[place, arange] * factor[env]
            amb_est[pol.value][i] = np.where(located, live_at[place[src], arange], np.nan)
            amb_est_home[pol.value][i] = live_at[PLACE_HOME]

    # --- weather on the grid ---
    w = weather.sort_index()
    w = w[~w.index.duplicated(keep="last")]
    wx = w.reindex(hours, method="ffill", tolerance=pd.Timedelta(hours=3))
    temp = wx["temperature"].interpolate(limit_direction="both").to_numpy()
    rh = wx["relative_humidity"].interpolate(limit_direction="both").to_numpy()
    precip = wx["precipitation"].fillna(0).to_numpy()

    # --- infections (daily onset, 5–10 day episodes with a rise and decay) ---
    local = hours + IST
    day_month = pd.Series(local.month, index=local_day).groupby(level=0).first().to_numpy()
    season = np.array([cfg.infection_season_multiplier[m] for m in day_month])
    p_inf = cfg.infections_per_year / 365 * season
    inf_day = np.zeros((P, n_days))
    starts = rng_inf.random((P, n_days)) < p_inf[None, :]
    durations = rng_inf.integers(5, 11, (P, n_days))
    for i, d0 in zip(*np.nonzero(starts), strict=True):
        dur = durations[i, d0]
        prof = np.minimum(1.0, np.minimum(np.arange(1, dur + 1) / 2, (dur - np.arange(dur)) / 3))
        seg = slice(d0, min(n_days, d0 + dur))
        inf_day[i, seg] = np.maximum(inf_day[i, seg], prof[: seg.stop - seg.start])
    infection = inf_day[:, local_day]

    # --- latent daily disease control (AR(1), stationary unit variance) ---
    rho = cfg.control_rho_daily
    ctl_day = np.empty((P, n_days))
    ctl_day[:, 0] = rng_inf.normal(0, 1, P)
    shocks = rng_inf.normal(0, math.sqrt(1 - rho**2), (P, n_days))
    for d in range(1, n_days):
        ctl_day[:, d] = rho * ctl_day[:, d - 1] + shocks[:, d]
    control = ctl_day[:, local_day]

    # --- hazard ---
    scale = cfg.aq_effect_scale
    sens = truth["sensitivity"].to_numpy()[:, None]
    e25 = _ema(true_exposure["pm25"], cfg.pm_halflife_hours)
    e10 = _ema(true_exposure["pm10"], cfg.pm_halflife_hours)
    eo3 = _ema(true_exposure["o3"], cfg.o3_halflife_hours)
    ist_hour = local.hour.to_numpy()
    is_asthma = (patients["disease_type"] == "asthma").to_numpy()[:, None]
    eta = (
        scale
        * (
            sens * cfg.beta_pm25_per10 * (e25 - 25) / 10
            + cfg.beta_pm10_per10 * (e10 - 50) / 10
            + cfg.beta_o3_per10 * (eo3 - 60) / 10
        )
        + cfg.beta_cold_per5c * np.maximum(0, 18 - temp)[None, :] / 5
        + cfg.beta_humid * (rh > 85)[None, :]
        + cfg.beta_infection * infection
        + cfg.beta_control_per_sd * control
        + cfg.beta_asthma_night * (is_asthma & (ist_hour < 6)[None, :])
        + truth["frailty"].to_numpy()[:, None]
    )
    del e25, e10, eo3
    # Solve the stratum intercept so the stratum's mean hazard equals its configured rate
    # (ignoring the recent-event term, which only redistributes events in time).
    intercept = np.empty(P)
    for (d, s), grp in patients.groupby(["disease_type", "severity_idx"]):
        target = cfg.event_rate_per_year[d][s] / 8760
        intercept[grp.index] = math.log(target) - math.log(np.exp(eta[grp.index]).mean())
    eta += intercept[:, None]

    u = rng_event.random((P, H))
    onset = np.zeros((P, H), dtype=bool)
    busy_until = np.full(P, -1)
    last_onset = np.full(P, -(10**9))
    recent_h = cfg.recent_event_days * 24
    for t in range(H):
        recent = (t - last_onset) <= recent_h
        lam = np.exp(eta[:, t] + cfg.beta_recent_event * recent)
        hit = (u[:, t] < 1 - np.exp(-lam)) & (t > busy_until)
        onset[hit, t] = True
        busy_until[hit] = t + cfg.episode_hours - 1
        last_onset[hit] = t
    del eta, u

    diary = _diary(cfg, patients, truth, hours, true_exposure, infection, control, onset, rng_diary)
    return Cohort(
        config=cfg,
        hours=hours,
        patients=patients,
        truth=truth,
        amb_est=amb_est,
        amb_est_home=amb_est_home,
        true_exposure={k: v.astype(np.float32) for k, v in true_exposure.items()},
        env_reported=env_reported,
        location_available=location_available,
        infection=infection.astype(np.float32),
        onset=onset,
        diary=diary,
        weather=pd.DataFrame(
            {"temperature": temp, "relative_humidity": rh, "precipitation": precip}, index=hours
        ),
    )


def _diary(
    cfg: CohortConfig,
    patients: pd.DataFrame,
    truth: pd.DataFrame,
    hours: pd.DatetimeIndex,
    true_exposure: dict[str, np.ndarray],
    infection: np.ndarray,
    control: np.ndarray,
    onset: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One diary per patient-day at 21:00 IST, summarising the 24 h ending then."""
    P, H = onset.shape
    # in_episode[t]: an onset within the last episode_hours.
    csum = np.cumsum(onset, axis=1)
    lagged = (
        np.concatenate(
            [np.zeros((P, cfg.episode_hours), dtype=csum.dtype), csum[:, : H - cfg.episode_hours]],
            1,
        )
        if H > cfg.episode_hours
        else np.zeros_like(csum)
    )
    in_episode = (csum - lagged) > 0

    first_diary = hours[0].normalize() + DIARY_HOUR_UTC
    if first_diary < hours[0] + pd.Timedelta(hours=24):
        first_diary += pd.Timedelta(days=1)
    stamps = pd.date_range(first_diary, hours[-1], freq="D")
    ends = hours.searchsorted(stamps, side="right")  # hours ≤ stamp are [.., end)
    sev = patients["severity_idx"].to_numpy()
    sens = truth["sensitivity"].to_numpy()
    adherence = truth["diary_adherence"].to_numpy()
    is_copd = (patients["disease_type"] == "copd").to_numpy()
    wearable = patients["has_wearable"].to_numpy()
    employed = patients["employed"].to_numpy()
    scale = cfg.aq_effect_scale

    out = []
    for stamp, e in zip(stamps, ends, strict=True):
        s = max(0, e - 24)
        pm = true_exposure["pm25"][:, s:e].mean(axis=1)
        inf = infection[:, s:e].max(axis=1)
        ctl = control[:, s:e].mean(axis=1)
        episode = in_episode[:, s:e].any(axis=1)
        exac = onset[:, s:e].any(axis=1)
        mean_sym = (
            0.8
            + 0.7 * sev
            + 3.5 * inf
            + 1.0 * ctl
            + 0.35 * scale * sens * np.maximum(0, pm - 25) / 10
            + 3.0 * episode
        )
        sym = np.clip(np.round(rng.normal(mean_sym, 1.0)), 0, 10)
        puffs = np.minimum(30, rng.poisson(0.25 + 0.5 * np.maximum(sym - 1, 0)))
        filled = rng.random(P) < adherence
        worn = wearable & (rng.random(P) < cfg.wearable_wear_prob)
        spo2 = np.clip(
            np.where(is_copd, 94.5 - 0.7 * sev, 97.5)
            - 2.0 * episode
            - 1.0 * inf
            + rng.normal(0, 0.8, P),
            80,
            100,
        ).round(1)
        hr = (68 + 4 * sev + 10 * episode + 6 * inf + rng.normal(0, 4, P)).round()
        steps = (
            np.where(employed, 6000, 3500) * (1 - 0.5 * episode) * np.exp(rng.normal(0, 0.35, P))
        ).round()
        sleep = np.clip(7.2 - 0.25 * sym + rng.normal(0, 0.7, P), 2, 12).round(1)
        keep = filled | worn
        if not keep.any():
            continue
        out.append(
            pd.DataFrame(
                {
                    "patient_idx": np.flatnonzero(keep),
                    "timestamp": stamp,
                    "symptom_score": np.where(filled, sym, np.nan)[keep],
                    "rescue_inhaler_puffs": np.where(filled, puffs, np.nan)[keep],
                    "exacerbation": np.where(filled, exac, np.nan)[keep],
                    "spo2": np.where(worn, spo2, np.nan)[keep],
                    "heart_rate": np.where(worn, hr, np.nan)[keep],
                    "steps": np.where(worn, steps, np.nan)[keep],
                    "sleep_hours": np.where(worn, sleep, np.nan)[keep],
                }
            )
        )
    diary = pd.concat(out, ignore_index=True)
    diary.insert(0, "patient_id", patients["patient_id"].to_numpy()[diary["patient_idx"]])
    return diary.sort_values(["patient_idx", "timestamp"], ignore_index=True)
