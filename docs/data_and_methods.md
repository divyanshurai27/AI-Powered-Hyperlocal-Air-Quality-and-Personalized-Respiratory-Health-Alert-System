# AeroGard — Data and Methods

Living record of every data and modelling decision, for the report's Methods section.
Each rule points to the code that implements it. Last updated: 2026-09-25.

## 1. Study area and sources

| Item | Decision |
|---|---|
| Study area | Bengaluru, India. Centre 12.9716 N, 77.5946 E; station search radius 25 km |
| AQ ground truth | CPCB/KSPCB regulatory reference monitors, accessed via the OpenAQ v3 API (`/sensors/{id}/hours`) |
| Excluded from ground truth | Low-cost sensor networks listed on OpenAQ (e.g. AirGradient). Stored as stations, never ingested as observations |
| Live fallback | CPCB real-time feed on data.gov.in. Adapter built, but disabled until its units are verified against a live response (the portal returned HTTP 502 during development) |
| Weather | Open-Meteo: ERA5 reanalysis (archive) for history; numerical forecast for live use. ERA5 is modelled, not station-measured |

**Coverage finding.** Every Bengaluru station has a gap from about Oct 2022 to Feb 2025, where the old OpenAQ sensor feed ends and the new one begins. Modelling therefore uses **18 Feb 2025 onward** (about 19 months). Availability also varies within that period: for example, BTM Layout PM2.5 reported 934 of 1,464 hours (64%) in Jun–Jul 2025.

**Unit-label finding (checked 2026-09-25).** The 2025+ CPCB feed on OpenAQ labels NO₂, SO₂ and CO as "ppb". Each gas was compared with the legacy feed (labelled µg/m³) at BTM Layout, Jun–Sep 2021 vs Jun–Sep 2025, medians:

| Gas | Legacy (µg/m³) | New ("ppb") | Treatment |
|---|---|---|---|
| CO | 1260 | 0.49 | **Corrected.** Real ambient CO is ~400–1500 ppb, so the label is wrong. Values are read as mg/m³, CPCB's native CO unit (`UNIT_OVERRIDES` in `app/ingestion/openaq.py`). The 129,348 CO rows stored before the fix were converted in place on 2026-09-25 |
| NO₂ | 13.4 | 23.4 | **Unverified.** Stored as ppb → µg/m³ (×1.8816). Whether the label is right can't be decided from magnitudes alone |
| SO₂ | 8.5 | 15.6 | **Unverified.** Stored as ppb → µg/m³ (×2.6203). Same issue as NO₂ |
| PM2.5 (control) | 24.1 | 17.4 | Same unit on both feeds |

Consequences:
- NO₂ and SO₂ absolute levels may be overstated by 1.88× and 2.62×. They are not used in risk thresholds or user-facing levels until checked against CPCB's own µg/m³ values in the data.gov.in feed, comparing the same station and hour.
- As *forecast inputs* they are unaffected, because tree models are invariant to a constant rescaling of a feature.

**Latency finding.** OpenAQ lags real time by about 3 days, so a "current" value from OpenAQ alone is normally stale and is reported as unavailable (BR-05).

## 2. Canonical observation (`app/domain/observations.py`)

- **Timestamp** = the UTC *end* of the averaging hour. IST hours end at :30 UTC, so a value stamped 00:30 UTC is the 05:00–06:00 IST mean and is fully known at its timestamp.
- **Units.** Everything is stored in µg/m³. Gases reported in ppb are converted at 25 °C and 1 atm: µg/m³ = ppb × MW / 24.45 (NO₂ 1.8816, O₃ 1.9631, SO₂ 2.6203, CO 1.1456 per ppb). Any other unit pair is rejected. Raw value and unit are kept alongside.
- **Rejected** (never stored), with the reason counted per ingestion run: missing or naive timestamp, missing or unsupported pollutant, invalid coordinates, missing, non-finite or negative value, unknown unit.
- **Flagged but kept**, highest severity wins:
  - `implausible`: above the physical ceiling, e.g. PM2.5 > 1000 µg/m³;
  - `provider_flagged`;
  - `low_coverage`: under 75% of sub-hourly readings (a common data-completeness convention, not a CPCB-specific rule).
- **Deduplication identity** = (source, station, pollutant, timestamp). When records conflict, a VALID flag wins, then higher coverage, then the first seen. Database upserts apply the same rule, so re-ingesting is idempotent.

## 3. Cleaning (`app/domain/cleaning.py`)

- The hourly grid never invents values; missing hours stay NaN with `is_missing`.
- **Model inputs** use a past-only forward fill for gaps of 3 hours or less (`was_imputed=True`). Longer gaps stay missing in full.
- Linear interpolation exists for charts and descriptive statistics only. It uses the value after the gap, so it would leak future information into features.
- **Outliers** are flagged with a trailing 24 h robust z-score (median/MAD, |z| > 6, at least 6 prior hours) and never deleted. Real extreme events are expected to be flagged too.
- `implausible` readings are removed before feature building and are never targets.

## 4. Hyperlocal matching (`app/domain/spatial.py`, `spatial_idw_v1.0.0`)

1. Candidate stations lie within 10 km, measured by haversine distance with R = 6371.0088 km.
2. Readings are dropped if they are older than 3 h, come from after the query time, or are implausible.
3. If the nearest usable station is 0.5 km away or less, that station is used alone.
4. Otherwise, inverse-distance weighting (power 2) over the 4 nearest usable stations. Low-coverage or provider-flagged readings get half weight.
5. If no station is usable, the result is explicitly unavailable, with a reason: `no_stations_in_radius`, `no_fresh_data`, `no_usable_data` or `invalid_location`.

## 5. Forecast features (`app/ml/features.py`, `aq_features_v1.0.0`)

One row per (station, origin hour t). Every feature uses only data at or before t:

- **Target pollutant:** current value, imputed flag, lags of 1/6/12/24/48 h, 1 h difference, rolling 6 h and 24 h mean/max/std, and the count of observed hours in the last 24 h.
- **Other pollutants:** current PM2.5, PM10, NO₂, O₃ and CO, excluding the target.
- **Weather:** latest row at or before t (at most 3 h old): temperature, RH, wind speed, sine and cosine of wind direction, pressure, precipitation, and 24 h precipitation total.
- **Calendar (IST):** sine and cosine of hour, day of week, weekend flag, sine and cosine of month.
- **Station:** latitude and longitude.

Leakage is guarded by two tests. First, every feature declares how far back it reads, and every offset must be ≤ 0. Second, changing all data after t leaves the features at t bit-identical.

## 6. 24 h forecasting protocol (`ml/forecasting/train_aq_forecast.py`)

- **Direct multi-horizon:** one LightGBM regressor per horizon h = 1…24, trained on log1p(y). Forecasts are never fed back in as inputs.
- **Baselines:**
  - persistence: ŷ(t+h) = y(t);
  - seasonal naive: ŷ(t+h) = y(t+h−24).
- **Targets** are observed values only. Imputed hours are never scored.
- **Chronological split**, never shuffled, with a 24 h embargo before each boundary:

  | Split | Period |
  |---|---|
  | Train | 18 Feb 2025 – 31 Jan 2026 |
  | Validation | 1 Feb – 30 Apr 2026 (early stopping only) |
  | Test | 1 May 2026 onward (touched once) |

- **Fair comparison:** all models are scored on the same (origin, horizon) pairs, i.e. where every model has a forecast and an observation exists.
- **Metrics:** MAE, RMSE, R² and bias, pooled and per horizon, plus skill vs persistence = 1 − MAE/MAE_persistence.
- **Artifact:** `model.joblib` plus `metadata.json` (PRD §39) with a SHA-256 checksum. The API refuses an artifact whose checksum, feature version, feature list or horizons don't match the code.

**Train/serve difference.** Training uses ERA5 weather at t. Live serving uses the latest weather forecast issued at or before t, because ERA5 is published about 5 days late. Historical `at=` requests use ERA5, as training does.

## 7. Exposure estimate (`app/domain/exposure.py`, `exposure_io_v1.0.0`)

exposure = local ambient × F(microenvironment, pollutant). **This is a proxy, not a measurement.**

| Microenvironment | PM2.5 | PM10 | NO₂ | O₃ | SO₂ | CO |
|---|---|---|---|---|---|---|
| outdoor / unknown | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| indoor_home | 0.8 | 0.6 | 0.6 | 0.3 | 0.5 | 0.9 |
| indoor_other | 0.6 | 0.5 | 0.5 | 0.2 | 0.4 | 0.8 |
| in_transit | 1.3 | 1.3 | 1.6 | 0.8 | 1.1 | 1.8 |

- The factors are central assumptions drawn from indoor/outdoor and in-transit literature ranges, not calibrated to AeroGard users. `sensitivity_factors(scale)` re-runs results with the non-outdoor factors scaled. The sensitivity analysis is in §14.
- If the microenvironment is unknown, F = 1.0 and the hour is flagged `assumed_outdoor`.
- If there's no location fix within 2 h at or before an hour, that hour's exposure is missing (`location_unavailable`), never guessed (BR-07).
- Window statistics (6/24/72 h mean, max and cumulative µg/m³·h) report their coverage and are marked incomplete below 75%.

## 8. Privacy and access

- Location, exposure, health-record and risk endpoints need `consent_status = granted`. They are patient-scoped with ownership checks (BR-11): every route is `/me`, and every query filters by the caller's patient ID.
- Audit logs record counts and field names, never location, health values or risk probabilities.

## 9. Results: 24 h AQ forecast v1.1.0 (dataset `aq_1a51008ca3ce9fa6`, run 2026-09-25)

Test period 2026-05-01 → 2026-09-21 (monsoon season; no winter in test). All models are scored on the same pairs. Targets are VALID hours.

| Pollutant | Model | MAE | RMSE | R² | MAE h1 | MAE h6 | MAE h24 | Skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| PM2.5 | **LightGBM** | **5.78** | 15.67 | 0.31 | 3.18 | 5.71 | 6.39 | +23.1% |
| PM2.5 | seasonal naive | 6.92 | 22.96 | −0.49 | 7.24 | 6.87 | 7.02 | +7.9% |
| PM2.5 | persistence | 7.51 | 22.82 | −0.47 | 3.32 | 7.66 | 7.02 | 0 |
| PM10 | **LightGBM** | **14.01** | 26.12 | 0.56 | 6.68 | 14.19 | 15.15 | +26.4% |
| PM10 | seasonal naive | 15.61 | 34.34 | 0.24 | 15.72 | 15.62 | 15.66 | +18.0% |
| NO₂* | **LightGBM** | **5.74** | 37.69 | 0.24 | 3.76 | 5.83 | 6.08 | +27.6% |
| NO₂* | persistence | 7.93 | 45.39 | −0.11 | **2.36** | 7.51 | 6.95 | 0 |
| O₃ | LightGBM | 5.02 | **10.51** | **0.59** | 2.79 | 5.27 | 5.29 | +24.8% |
| O₃ | **seasonal naive** | **4.83** | 11.32 | 0.52 | 4.90 | 4.88 | 4.85 | +27.7% |

*NO₂ absolute units are unverified (see §1).

Findings to report:
- LightGBM beats both baselines on MAE for PM2.5, PM10 and NO₂.
- **For O₃, seasonal naive has the lower MAE**, while LightGBM has the lower RMSE and higher R². Ozone's strong diurnal cycle makes "same hour yesterday" a hard baseline.
- For NO₂ at 1 h ahead, persistence beats LightGBM.
- For PM2.5, RMSE ≫ MAE and R² is modest: isolated spikes remain even among VALID hours, and the model does not anticipate them.
- PM2.5 on all observed hours, including low-coverage and flagged ones: LightGBM MAE 5.95 vs persistence 7.67 (+22.4%). The conclusion does not depend on the target filter.
- **v1.0.0 → v1.1.0.** v1.0.0 trained on and was scored against all observed hours. Isolated 600–900 µg/m³ PM2.5 spikes during the monsoon, mostly at 25–50% coverage, looked like sensor glitches, so the target definition was tightened to VALID hours. Both artifacts are kept.

Reports: `docs/experiments/aq_forecast_<pollutant>_1.1.0.json`

## 10. "Today for you" guidance (`app/domain/guidance.py`, `guidance_naqi_v1.0.0`)

This is rule-based air-quality guidance, **not** the Phase 4 risk model. It never says "safe" and never mentions medication.

**Inputs**
- The 24 h forecasts for PM2.5, PM10 and O₃ at the user's saved home or work, or at a chosen spot.
- NO₂ and SO₂ are excluded until their units are verified.
- If a pollutant has no model, it is listed as skipped, never filled in.

**Hourly category.** Each hour is assigned India's NAQI category for its worst pollutant. NAQI is defined on 24 h averages (8 h for O₃), so applying it to hourly forecasts is an approximation.

**Guidance level by sensitivity**

| Tier | Good | Satisfactory | Moderate | Poor | Very Poor / Severe |
|---|---|---|---|---|---|
| General (no condition) | good | good | caution | limit | avoid |
| Asthma/COPD (mild or moderate) | good | caution | limit | avoid | avoid |
| Asthma/COPD (severe or very severe) | good | caution | avoid | avoid | avoid |

The shift follows NAQI's health statements: "Satisfactory" means minor breathing discomfort for sensitive people, and "Moderate" means breathing discomfort for people with lung disease such as asthma.

**Headline and windows**
- **Headline** = the worst waking hour (06:00–22:00 IST). This is deliberately conservative.
- **Best window** = the cleanest two consecutive waking hours.
- **Avoid windows** = waking runs at "limit" or worse **and** worse than the best hour. On a uniformly bad day none are flagged, and the worst stretch is reported instead.

**City map.** A ~2.2 km grid. Each station is forecast once, then blended per cell with the same inverse-distance weights as point queries. Cells with no usable station within 10 km are left empty.

## 11. Synthetic respiratory cohort (`app/ml/cohort.py`, `cohort_v1.0.0`)

No respiratory outcome data exist for Bengaluru, so the risk layer is developed and evaluated on a **simulated** panel of patients (PRD §3.2). **Every risk result in this document is a synthetic-data evaluation, not clinical validation.**

**Real vs simulated**
- Real: hourly PM2.5, PM10 and O₃ at the 14 CPCB/KSPCB monitors, and ERA5 weather, 2025-02-18 → last observation.
- Simulated: the patients, their homes, workplaces and routines, their personal exposure, infections, symptom diaries, wearable readings and exacerbations.

**Data-generating process** (all constants in `CohortConfig`)
1. **Patients** (default 400, seed 2026): 40% COPD. Severity mix: asthma 40/35/18/7%, COPD 25/40/25/10% (mild → very severe). Ages: asthma ~N(35, 14), COPD ~N(62, 9). Homes are uniform within 12 km of the city centre; workers (about 70% of asthma patients under 60, 50% of COPD patients) have a workplace within 14 km.
2. **Hidden traits** (never features): frailty u ~ N(0, 0.5); pollution sensitivity s ~ LogNormal with mean 1 (σ 0.7); personal microenvironment factors = the §7 table × LogNormal(0, 0.25), standing for housing and ventilation differences.
3. **Routine:** home, a 1 h commute (in transit, halfway point), work indoors until the return hour, a commute back; 15% work-from-home days; weekend errands and evenings out; a morning walk for 40% of patients.
4. **True exposure** = true ambient at the place × the personal factor. True ambient is an IDW field over every station with a value, times a smooth AR(1) noise (σ 0.15, ρ 0.97 per hour) for street-scale variation the monitors can't see.
5. **What the system sees** mirrors production:
   - Location is logged per hour with a patient-specific rate (Beta(8, 1.5), mean 0.84) plus 5% whole-day dropouts.
   - A fix up to 2 h old covers an hour (BR-07).
   - The activity context is unknown 10% of the time.
   - Ambient is computed with the live spatial rule: 10 km radius, at most 4 stations, readings at most 3 h old. `idw_field` is tested against `domain.spatial.estimate_local_air_quality`.
6. **Hourly event hazard.** Event = a respiratory exacerbation episode: a symptom flare needing extra reliever or unscheduled care.

   log λ = c_stratum + u + s·ln(1.10)·(Ē₂.₅ − 25)/10 + ln(1.03)·(Ē₁₀ − 50)/10 + ln(1.04)·(Ē_O₃ − 60)/10 + ln(1.15)·max(0, 18 − T)/5 + ln(1.2)·[RH > 85] + ln(8)·infection + ln(2.2)·control + ln(1.8)·[event in last 14 d] + ln(1.5)·[asthma, 00–06 IST]

   - Ē are exponential moving averages of **true** personal exposure, with a half-life of 24 h for PM and 8 h for O₃.
   - Infections: 4 per year, seasonal (×1.6 in Jun–Sep, ×1.3 in Dec–Feb), lasting 5–10 days.
   - *control* is a latent daily AR(1) loss-of-control state (ρ 0.85, unit variance). Exacerbations are typically preceded by days of worsening symptoms, and this is where that shows up.
   - c_stratum is solved so each disease × severity stratum has its configured annual rate: asthma 4/6/9/12, COPD 3/4/6/8 events per year.
   - After an onset, the episode lasts 72 h, with no new onset during it.
   - All pollution terms are multiplied by `aq_effect_scale`. Setting it to 0 is a negative control in which pollution has no effect.
7. **Diary** at 21:00 IST, covering the preceding 24 h:
   - Symptom score 0–10 ≈ severity + 3.5·infection + control + the pollution response + 3·[in an episode] + N(0, 1). Rescue puffs are Poisson in the symptom score. The exacerbation flag is set if an onset occurred that day.
   - Each diary is filled with probability equal to the patient's adherence, Beta(6, 2), mean 0.75.
   - 40% of patients wear a device giving SpO₂, resting heart rate, steps and sleep.

**Assumptions to state in the paper.** The effect sizes are chosen to give a plausible, learnable signal. Panel studies report symptom-flare odds ratios of roughly 1.05–1.15 per 10 µg/m³ PM2.5, and viral infection is the dominant trigger. They are **not** calibrated to Indian epidemiology. The latent control state was added after a first draft gave AUROC ≈ 0.62 on a synthetic field: almost all events in that draft were unpredictable noise, which contradicts the prodrome literature. No further tuning toward a target score was done.

## 12. Health records and risk features

**Health records** (`health_records`, PRD §6.4). Diary entries and device summaries: symptom score, rescue puffs, exacerbation flag, heart rate, SpO₂, respiratory rate, steps, sleep, peak flow.
- Every field is optional, but a record needs at least one (DB check `num_nonnulls > 0`). An absent field stays NULL, never 0 (BR-06).
- Ranges are enforced in both the API and the database. Repeats of (patient, timestamp, source) are ignored.
- Clients can send `manual` or `health_connect`. `synthetic` is reserved for demo seeding.

**Risk target.** y = 1 if an event **starts** in (T, T+24 h]. Origins are two random whole hours per patient-day, so the model can be applied at any hour. Excluded origins:
- the first 10 days (warm-up for the longest lookback)
- the last 24 h of data
- origins inside an ongoing episode

**Features** (`app/ml/risk_features.py`, `risk_features_v1.0.0`, 38 features). The same functions build training rows and the live API row.

| Group | Features |
|---|---|
| Exposure (estimated, µg/m³) | PM2.5 now, 6/24/72 h mean, 24 h max, 24 h change, 24 h coverage; PM10 24 h mean, max and change; O₃ 8 h mean, 24 h max |
| Weather | temperature, RH, 24 h temperature change, 24 h precipitation |
| Calendar | hour and month (sin/cos), weekend |
| Profile | age, sex, COPD flag, severity (0–3) |
| Health | last symptom score (≤ 48 h old), 3/7-day mean, trend (3 d vs the 7 d before), last and 3-day rescue puffs, diary days in 7 d, events in 90 d, days since last event (capped at 365), last SpO₂ and heart rate (≤ 72 h old), 3-day steps, last sleep |

- Every feature declares a lookback of 0 or less. A test scrambles all inputs after T and requires identical features (BR-04).
- Missing values stay NaN. The training data contains the same kinds of gaps (unlogged hours, skipped diaries, no wearable), so LightGBM learns a branch for "missing". This is the BR-06 fallback.
- NO₂ and SO₂ are excluded until their units are verified (§1).

## 13. Risk model, calibration, policy and personalization

**Model** (`app/ml/risk.py`). A shared LightGBM trunk is trained on all patients, with disease as a feature. One LightGBM head per disease is then boosted from the trunk's output: small trees, a slow learning rate, and early stopping on validation. Probability = sigmoid(trunk + head). Baselines on the same protocol:
- stratum prior (base rate per disease × severity)
- logistic regression with missing-value indicators
- pooled LightGBM (the trunk alone)
- separate per-disease LightGBM
- ablations: without exposure features, and without health features

**Protocol** (`ml/risk/train_risk_model.py`, fixed before any test result was seen)
- **Patients:** 80% seen, 20% unseen (test only).
- **Time:** train ≤ 2026-01-31 < validation ≤ 2026-04-30 < test. Origins whose label window crosses a boundary are dropped.
- **Validation is used for:** early stopping, calibration, and deriving the policy thresholds. Test data is used once.
- **95% CIs:** patient-level cluster bootstrap with 200 resamples.

**Calibration** (`app/ml/calibration.py`, `calibration_v1.1.0`)
- One Platt (logistic) calibrator per disease, fitted on validation.
- **v1.0.0 → v1.1.0.** v1.0.0 used isotonic regression whenever validation had at least 200 events. Isotonic's lowest block can contain no events, so v1.0.0 reported a risk of about 0 for 6% of asthma and 4% of COPD test predictions. 25 events then occurred among those "zero-risk" moments, and isotonic gave no Brier gain. An exact-zero risk is not credible at a ~1.5% daily base rate, so Platt (monotone, never 0 or 1) became the rule. The problem was found during the live API check of a demo patient. As a side effect, AUROC also improved from 0.700 to 0.708, because isotonic's flat steps had created ties in the ranking.
- Reported: Brier score, Brier skill vs the base rate, and ECE/MCE over 10 equal-frequency bins. Equal-width bins would be empty above about 0.2 at these base rates.

**Policy** (`app/domain/risk_policy.py`, `risk_policy_v1.0.0`). Thresholds are set per disease on calibrated validation probabilities:
- Moderate ≥ 1.5 × base rate
- High = the F1-optimal alert threshold
- Severe = the top 1% of validation probabilities. A precision-0.5 target is never reached at a ~1.5% base rate, so it can't be used.

The thresholds are a config, so changing them needs no retraining (PRD §15.3).

**Personalization** (`personal_offset_v1.0.0`). p' = sigmoid(logit p + offset), where offset = ln((O + 2) / (E + 2)), clipped to ±ln 2.
- O = exacerbations the patient reported in the past year, before T.
- E = the stratum's rate of reported exacerbations per diary day × the patient's diary days.
- Counting per diary day stops a non-diarist from looking healthier than they are. With little history the offset tends to 0. `?personalize=false` turns it off.

**Serving** (`app/services/risk.py`)
- Location: the latest fresh fix, otherwise the saved home (flagged), otherwise none. With no location the confidence is marked low and the forecast is refused.
- The rolling forecast applies the model at T+1 … T+24. Exposure is projected from the 24 h AQ forecast at the current place and microenvironment, while health and weather are held at T. Horizons the AQ forecast doesn't cover are returned as unavailable.
- Each prediction is stored once per (patient, time, horizon, versions), with its feature vector (BR-08/09/10).

## 14. Results: respiratory risk `risk_v1.0.0` + `calibration_v1.1.0` (SYNTHETIC labels)

Cohort `cohort_v1.0.0`: 400 patients (250 asthma, 150 COPD), AQ dataset `aq_12ffff43711bdabc`, 3,449 events (5.4 per patient-year).
- Matrix `risk_86916fe8fac8bc65`: 425,114 origins.
- Train 201,422 rows (2,974 events); validation 52,623 (832); test (seen patients, 2026-05-01 → 2026-09-20) 84,908 (1,241); test (unseen patients) 21,293 (311).
- Test prevalence ≈ 1.5% per 24 h.

**Model comparison, test, seen patients (95% CI, patient-level bootstrap)**

| Model | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|
| stratum prior (no features) | 0.597 [0.564, 0.632] | 0.020 | 0.01438 | 0.0018 |
| logistic regression | 0.699 [0.674, 0.724] | 0.043 | 0.01424 | 0.0025 |
| LightGBM pooled | 0.709 [0.685, 0.732] | 0.043 | 0.01423 | 0.0025 |
| LightGBM separate per disease | 0.701 [0.679, 0.724] | 0.043 | 0.01423 | 0.0022 |
| **LightGBM shared trunk + heads (served)** | **0.708 [0.684, 0.732]** | 0.043 | 0.01423 | 0.0025 |
| ablation: no exposure features | 0.709 [0.685, 0.732] | 0.045 | 0.01421 | 0.0025 |
| ablation: no health features | 0.583 [0.555, 0.608] | 0.020 | 0.01439 | 0.0022 |

- Unseen patients: shared heads 0.721 [0.667, 0.758], logistic 0.725, no-exposure 0.730. The model generalises to people it never saw.
- By disease (seen): asthma 0.713, COPD 0.689. Early stopping kept the asthma head at 1 tree and the COPD head at 27.

**Findings to report**
- **Symptom-diary features carry the predictive signal.** Without them AUROC falls to 0.58, close to the stratum prior.
- **The shared-head architecture is not measurably better** than one pooled LightGBM, or than logistic regression. All CIs overlap. It is kept because it costs nothing and lets the COPD head learn a correction (27 trees), but the paper must not claim a gain.
- **Exposure features add no measurable value** at the configured effect sizes: the no-exposure ablation is equal or better at every effect scale (table below). Two reasons were diagnosed (`data/processed/diag_exposure.py`, not committed):
  1. The test period is the monsoon. Test PM2.5 exposure averages 17 µg/m³ (90th percentile 29, against 23.8 in training), mostly below the hazard's 25 µg/m³ reference. At scale 1 the PM2.5 term varies by only 0.11 log-hazard units, against 2.08 for a peak infection.
  2. The diary already carries the pollution response, because the simulated symptom score rises with exposure. At scale 4, exposure alone reaches a univariate AUROC of 0.68 in training but only 0.56 in test, and the symptom features absorb most of it.
  
  The test period was fixed before any result was seen and was not moved. A winter test window would be a separate, clearly labelled secondary analysis.
- **Well calibrated in absolute terms** (ECE 0.0025). Test events run about 20% above the predicted rate in most bins, because the monsoon test period has more infections than the Feb–Apr validation period the calibrators were fitted on. This is a seasonal base-rate shift, and it should be named in the paper.

**Effect-size sweep** (`--aq-effect-scale`, test on seen patients; the scale-1 row is the main run)

| Pollution effect | Events | Shared heads AUROC | No-exposure ablation AUROC |
|---|---|---|---|
| 0 (negative control) | 3,469 | 0.722 [0.701, 0.744] | 0.723 |
| 1 (default) | 3,449 | 0.708 [0.684, 0.732] | 0.709 |
| 2 | 3,431 | 0.713 [0.690, 0.736] | 0.716 |
| 4 | 2,370 | 0.717 [0.687, 0.747] | 0.724 |

The negative control behaves as it should: with no pollution effect, exposure features add nothing. They still add nothing at 2× and 4×, for the two reasons above. At scale 4 there are fewer events, because the stratum rates are solved over the whole period, which shifts events into the polluted winter.

**Risk levels on test (`risk_policy_v1.0.0`)**

| | Low | Moderate | High | Severe |
|---|---|---|---|---|
| Asthma: share of time | 88.9% | 8.2% | 2.1% | 0.8% |
| Asthma: observed event rate | 1.2% | 4.2% | 7.0% | 11.5% |
| COPD: share of time | 89.8% | 8.5% | 1.3% | 0.5% |
| COPD: observed event rate | 0.9% | 3.3% | 4.4% | 5.0% |

Event rates rise with each level, so the levels separate risk. The thresholds (asthma 0.025 / 0.048 / 0.077, COPD 0.021 / 0.052 / 0.080) came from validation only.

**Personalization (PRD §16), test, seen patients.** Alert = High or Severe, except for the AQ rules.

| Strategy | Precision | Recall | F1 | Share of time alerting |
|---|---|---|---|---|
| generic AQ threshold (NAQI Poor: PM2.5 > 90 or PM10 > 250) | 0.043 | 0.001 | 0.002 | 0.03% |
| disease-specific AQ threshold (NAQI Moderate: PM2.5 > 60 or PM10 > 100) | 0.025 | 0.011 | 0.015 | 0.6% |
| model, one global threshold | 0.069 | 0.123 | 0.089 | 2.6% |
| model, disease-specific thresholds | 0.072 | 0.120 | 0.090 | 2.5% |
| model + adaptive personal offset | 0.067 | 0.178 | 0.097 | 3.9% |

- **Bengaluru's air almost never crosses the NAQI thresholds**, so AQ-only alerting is close to silent. This is a real finding about the study area, not a weakness of the rules.
- The personal offset raises recall (0.12 → 0.18) at the cost of alerting more often (2.5% → 3.9% of the time). Probability quality is unchanged: AUROC 0.708 → 0.707, AUPRC 0.043 → 0.045, Brier 0.01423 both. For unseen patients its F1 is slightly lower (0.089 vs 0.093). **The gain is a different operating point, not better discrimination.** The model's own history features (events in 90 d, days since the last event) already capture most of what the offset adds.

**Exposure sensitivity analysis (deferred from Phase 3).** The model is retrained for each scenario. Exposure error is the 24 h mean PM2.5 against the simulated truth, test period.

| Exposure used | MAE (µg/m³) | Bias | Spearman ρ | AUROC |
|---|---|---|---|---|
| estimated (production rule) | 3.99 | −0.21 | 0.800 | 0.708 |
| microenvironment factors × 0.8 | 4.50 | −3.18 | 0.799 | 0.708 |
| microenvironment factors × 1.2 | 4.97 | +2.76 | 0.800 | 0.708 |
| ambient at home only (no mobility) | 5.76 | +3.66 | 0.777 | 0.706 |
| true exposure (oracle) | 0 | 0 | 1 | 0.708 |

- The production estimate is the least biased option. The ±20% factor perturbations only shift the bias, because rescaling barely changes the ranking (ρ stays 0.80).
- Ignoring mobility (home ambient only) increases error by 44% and overstates exposure, because people spend time in lower-exposure indoor settings.
- Risk discrimination does not depend on exposure quality here, which follows from the exposure finding above.

**Calibration version note.** `calibration_v1.0.0` (isotonic) gave AUROC 0.700 and reported exact-zero risks. See §13.

Reports: `docs/experiments/risk_1.0.0.json` (main), `risk_1.0.0_aq0.json`, `_aq2.json`, `_aq4.json`.

## 15. Explanations and alerts (Phase 5)

**Explanation method** (`app/ml/explain.py`, `treeshap_grouped_v1.0.0`)
- Exact TreeSHAP from LightGBM itself (`pred_contrib=True`), so the API needs no extra dependency. SHAP values are additive, so the served model's attribution is trunk + the patient's disease head: Σφ + base = the model's log-odds exactly (tests check to 1e-9).
- Platt calibration is linear in log-odds (logit p_cal = a·margin + b) and the personal offset is added in log-odds. Each feature's contribution is therefore reported as a·φ, the base as a·base + b, and the offset as its own "Your own flare-up history" factor. **The parts add up to the probability the patient is shown** (`reconstructed_probability`, checked in tests and on every stored prediction). An isotonic calibrator would not be linear; the explanation would then stay in raw log-odds and say so (`space`).
- The 38 features are grouped into 14 patient-facing factors (PM2.5 exposure, change in PM2.5, PM10, ozone, weather, time of day/season, condition and severity, age and sex, recent symptoms, reliever use, flare-ups, diary regularity, wearable readings, location-record completeness). Each feature belongs to exactly one factor (tested). A factor's contribution is the sum over its features, so effects inside a factor can cancel.
- The top 5 factors by |contribution| (≥ 0.02 log-odds) are worded as "… raised / lowered this estimate", never as causes (PRD §17), and carry a note that they are associations learned from a synthetic cohort.
- **Consistency (BR-09).** The explanation is computed in the same call, from the same model object and the same feature rows that produced the probability. It is stored on the prediction row with the model, feature and calibration versions and a SHA-256 fingerprint of the stored feature vector. `GET /risk/predictions/{id}/explanation` re-checks all four against the row and refuses (409 `EXPLANATION_UNAVAILABLE`) if anything differs.

**Alert engine** (`app/domain/alerts.py`, `alert_policy_v1.0.0`; `app/services/alerts.py`)

Create an alert when all of these hold:
1. The probability is valid (BR-01).
2. It is the current estimate (horizon 0). Projected hours never alert: they assume the patient stays put.
3. The level is High or above (`risk_policy.ALERT_LEVEL`).
4. The prediction is current: made at most 2 h before the evaluation time (`ALERT_MAX_PREDICTION_AGE_HOURS`).
5. No alert exists for this prediction (unique `prediction_id`, BR-10).
6. No *equivalent* alert is live: one for the same patient, not cancelled, whose 24 h window has not ended at the new prediction time, at the same or a higher level. An acknowledged alert still counts, so it is not repeated.

- **Escalation.** A level above every live alert (High → Severe) creates a new alert and cancels the open lower ones (`superseded_by_higher_level`).
- **Revision.** If the same prediction is recomputed (e.g. after a late diary entry) and falls below High, its own open alert is cancelled (`prediction_revised`).
- **States:** created → (sent) → acknowledged; cancelled; expired. "Expired" is derived when the alert is read (`expires_at` = end of the prediction's 24 h window), so replaying a past date shows the states as they were then. "Sent" is reserved for the push channel that arrives with the Android app.
- **Scope.** Only the product configuration alerts: `?personalize=false` is a research view and never raises one.
- **Contents.** The alert holds the risk level and percentage only, e.g. "Your estimated respiratory risk for the next 24 hours is High (6.1%)." It never includes medication guidance (BR-12). Every alert records its patient, prediction id, model, calibration, risk-policy and alert-policy versions, and time (PRD §19).
- **Atomicity (PRD §23).** The predictions and the alert decision are written in one transaction, with a row lock on the patient so two concurrent requests cannot both decide that no equivalent alert exists.

**Other Phase 5 endpoints**
- `GET /air/history`: hourly location-matched AQ for up to 168 h, with the same matching and freshness rules as `/air/current`. Gaps are reported per hour, not filled.
- `GET /risk/debug`: the PRD §33 prediction trace (input freshness, features, raw → calibrated → personal probability, policy, alert decision, explanation). It is served only when `APP_ENV=development`.
- Structured logs `risk_generated` and `alert_evaluated` carry the request's correlation id (PRD §34). Probabilities stay out of the logs and the audit trail.

**Global importance** (`ml/explainability/global_shap.py`). Mean |SHAP| per feature and per factor in calibrated log-odds, for validation, test (seen) and test (unseen), by disease. It explains the saved artifact on the exact training matrix and refuses to run if the matrix fingerprint differs. Output: `docs/experiments/risk_1.0.0_shap.json`.

## 16. Results: global SHAP importance of `risk_v1.0.0` (SYNTHETIC labels, run 2026-10-21)

Matrix `risk_86916fe8fac8bc65` (the training matrix, verified by fingerprint). Mean |SHAP| per factor in calibrated log-odds, test on seen patients (n = 84,908). The additivity check (base + Σ contributions vs the calibrated probability) holds to ≤ 1.1e-15 in every part.

| Factor | Mean \|SHAP\| | Share |
|---|---|---|
| Recent symptom scores | 0.488 | 54.2% |
| Recent flare-ups | 0.123 | 13.7% |
| Wearable readings (SpO₂, heart rate, steps, sleep) | 0.049 | 5.5% |
| Estimated PM2.5 exposure | 0.035 | 3.9% |
| Age and sex | 0.034 | 3.8% |
| Condition and severity | 0.033 | 3.7% |
| Estimated PM10 exposure | 0.030 | 3.3% |
| Estimated ozone exposure | 0.029 | 3.2% |
| Time of day and season | 0.027 | 3.0% |
| Recent reliever-inhaler use | 0.019 | 2.1% |
| Weather | 0.017 | 1.9% |
| Recent change in PM2.5 exposure | 0.010 | 1.1% |
| Diary regularity | 0.005 | 0.5% |
| Location-record completeness | 0.002 | 0.2% |

- **The symptom diary dominates**: symptoms + flare-ups ≈ 68% of attribution in every part (validation, test seen, test unseen) and for both diseases. The top individual features are the 3-day and last symptom scores, the 7-day mean, and days since the last event. This matches the no-health ablation (AUROC 0.708 → 0.583, §14).
- **All exposure factors together get 11.7% of attribution on test (15.5% on validation), yet removing them does not lower AUROC (§14).** The two findings are consistent. SHAP shows what the trained model *uses*. The ablation shows what it *needs*: exposure is correlated with season and with the simulated symptom response, so a model trained without it reaches the same discrimination through those features. The paper should report SHAP as model behaviour, not as evidence that pollution adds predictive value.
- Exposure attribution is lower on the monsoon test period (11.7%) than on Feb–Apr validation (15.5%), consistent with the low, narrow test-period exposure range described in §14.
- Importances are near-identical for seen and unseen patients and for asthma and COPD (COPD's exposure share is 0.2–1.2 points higher).

These describe what the simulated cohort encoded, not clinical knowledge.
