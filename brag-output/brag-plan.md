# AeroGard — brag plan

**What it is:** Hyperlocal air-quality forecasts for Bengaluru plus a personal respiratory-risk estimate and alerts for people with asthma or COPD.
**Who it's for:** people whose lungs react before the city's air index does.
**What sets it apart:** the same air reads differently per person — a "Moderate" NAQI hour can be a High-risk hour for a very severe asthmatic, and AeroGard shows *why* (TreeSHAP factors).
**Share caption:** The air index said Moderate. For a very severe asthmatic, AeroGard said High.

## Angle
Same air, different lungs. Open on the city's verdict ("Moderate"), overturn it with the personal one ("High"), then show the three pieces that produce it.

## Tone
`default` with a calm, clinical edge: punchy cuts, soft transitions, nothing alarmist. Honest caveat in the outro (synthetic cohort, not medical advice).

## Visual identity
The app's own `styles.css` (copied into `work/`): accent `#0b6e4f`, NAQI colours, Segoe UI, real card / pill / factor-bar markup from `app.js`. Scenes sit on a deep green-black (`#0c1a15`) so the light cards pop.

## Data
Real API responses captured from the high-risk demo account (synthetic patient syn-0099) at 29 Jun 2026, 11:30 IST: PM10 105 µg/m³ (Moderate), risk High 5.6 %, top factors, one open alert (High 5.1 %). No emails, tokens or hostnames appear on screen; the location is labelled only "Bengaluru".

## Music
120 BPM (beat 0.5 s, bar 2 s), A minor → C major lift at the outro. Soft kick/hat, round bass, plucked arps; whoosh swells and a pitched tick on cuts; a soft chime on the alert and a click on "Acknowledge".

## Storyboard (21 s, every cut on a beat)
| # | Time | Scene |
|---|---|---|
| 1 Hook | 0.0–3.0 | "29 June, 11:30 IST · Bengaluru". **105** counts up, "µg/m³ PM10", NAQI pill **Moderate** pops on beat. "The air index says: Moderate." |
| 2 Turn | 3.0–6.0 | Moderate pill dims and strikes; **High** risk pill slams in. "For a very severe asthmatic, it's a High-risk day." |
| 3 Reveal | 6.0–9.0 | Logo + **AeroGard**. "Hyperlocal air + personal risk alerts for asthma & COPD." |
| 4 Hyperlocal | 9.0–12.0 | Real "Air quality now" card; pollutant chips land one by one. "Air estimated for your exact spot." |
| 5 Risk | 12.0–16.0 | Real risk card: High, 5.6 % counts up, factor bars grow. "Your 24-hour risk. And why." |
| 6 Alert | 16.0–18.5 | Alert slides in, cursor clicks Acknowledge. "One alert, when it counts." |
| 7 Outro | 18.5–21.0 | Logo, "Forecast the air. Warn the person." Fine print: research prototype · synthetic cohort · not medical advice. |
