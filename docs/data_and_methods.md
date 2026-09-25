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

- The factors are central assumptions drawn from indoor/outdoor and in-transit literature ranges, not calibrated to AeroGard users. `sensitivity_factors(scale)` re-runs results with the non-outdoor factors scaled, and the paper must report that sensitivity.
- If the microenvironment is unknown, F = 1.0 and the hour is flagged `assumed_outdoor`.
- If there's no location fix within 2 h at or before an hour, that hour's exposure is missing (`location_unavailable`), never guessed (BR-07).
- Window statistics (6/24/72 h mean, max and cumulative µg/m³·h) report their coverage and are marked incomplete below 75%.

## 8. Privacy and access

- Location and exposure endpoints need `consent_status = granted`. They are patient-scoped with ownership checks (BR-11).
- Audit logs record counts and field names, never location or health values.

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
