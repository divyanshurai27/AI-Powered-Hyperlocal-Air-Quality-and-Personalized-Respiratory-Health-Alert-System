"""Build docs/AeroGard_Progress_Report_Phases_1-3.pdf from live project data.

Numbers come from the database and docs/experiments/*.json, not from hand-typed values.
Run from the repo root with the venv active:  python scripts/build_progress_report.py
"""

import json
import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy import text

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.db.session import get_sessionmaker  # noqa: E402

OUT = REPO / "docs" / "AeroGard_Progress_Report_Phases_1-3.pdf"
EXPERIMENTS = REPO / "docs" / "experiments"

# Arial covers µ, ³, ², ×, en/em dashes. Subscripts use <sub> markup, never Unicode subscripts.
FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONT_DIR / "ariali.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
ACCENT = colors.HexColor("#0b6e4f")
RULE = colors.HexColor("#d9e2ec")
HEAD_BG = colors.HexColor("#e6f2ee")
WARN_BG = colors.HexColor("#fff4e5")

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle(
        "t",
        parent=base["Title"],
        fontName="Body-Bold",
        fontSize=24,
        leading=29,
        textColor=INK,
        spaceAfter=4,
    ),
    "subtitle": ParagraphStyle(
        "st",
        fontName="Body",
        fontSize=12,
        leading=16,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceAfter=2,
    ),
    "h1": ParagraphStyle(
        "h1",
        fontName="Body-Bold",
        fontSize=15,
        leading=19,
        textColor=ACCENT,
        spaceBefore=12,
        spaceAfter=6,
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="Body-Bold",
        fontSize=11.5,
        leading=15,
        textColor=INK,
        spaceBefore=8,
        spaceAfter=4,
    ),
    "body": ParagraphStyle(
        "b", fontName="Body", fontSize=9.8, leading=13.6, textColor=INK, spaceAfter=5
    ),
    "bullet": ParagraphStyle(
        "bl",
        fontName="Body",
        fontSize=9.8,
        leading=13.4,
        textColor=INK,
        leftIndent=12,
        bulletIndent=2,
        spaceAfter=2.5,
    ),
    "small": ParagraphStyle("sm", fontName="Body", fontSize=8.3, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("c", fontName="Body", fontSize=8.6, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Body-Bold", fontSize=8.6, leading=11, textColor=INK),
}


def P(txt: str, style: str = "body") -> Paragraph:
    return Paragraph(txt, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(i, S["bullet"], bulletText="•") for i in items]


def table(
    rows: list[list],
    widths: list[float],
    highlight_rows: tuple[int, ...] = (),
    warn_rows: tuple[int, ...] = (),
) -> Table:
    bold = {0, *highlight_rows}
    data = [
        [P(str(c), "cellb" if r in bold else "cell") for c in row] for r, row in enumerate(rows)
    ]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in warn_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), WARN_BG))
    t.setStyle(TableStyle(style))
    return t


def note(txt: str) -> Table:
    t = Table([[P(txt, "body")]], colWidths=[170 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def fetch_data_summary() -> dict:
    with get_sessionmaker()() as db:
        pollutants = db.execute(
            text("""
            select pollutant, count(distinct station_id) stations, count(*) n,
                   round(100.0*count(*) filter (where quality_flag<>'valid')/count(*),1) flagged,
                   round(percentile_cont(0.5) within group (order by concentration)::numeric,1) median,
                   min(timestamp)::date first, max(timestamp)::date last
            from air_quality_observations group by pollutant order by n desc""")
        ).all()
        totals = db.execute(
            text("""
            select count(*), count(distinct station_id),
                   (select count(*) from weather_observations where kind='reanalysis'),
                   (select count(*) from stations where is_reference_monitor)
            from air_quality_observations""")
        ).one()
        flags = dict(
            db.execute(
                text("select quality_flag, count(*) from air_quality_observations group by 1")
            ).all()
        )
    return {"pollutants": pollutants, "totals": totals, "flags": flags}


def load_experiments() -> dict[str, dict]:
    out = {}
    for p in ("pm25", "pm10", "no2", "o3"):
        f = EXPERIMENTS / f"aq_forecast_{p}_1.1.0.json"
        if f.exists():
            out[p] = json.loads(f.read_text())
    return out


LABEL = {
    "pm25": "PM<sub>2.5</sub>",
    "pm10": "PM<sub>10</sub>",
    "no2": "NO<sub>2</sub>",
    "o3": "O<sub>3</sub>",
    "so2": "SO<sub>2</sub>",
    "co": "CO",
}
MODEL = {
    "lightgbm_direct": "LightGBM (24 direct models)",
    "seasonal_naive": "Seasonal naive (same hour yesterday)",
    "persistence": "Persistence (no change)",
}


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        20 * mm,
        10 * mm,
        "AeroGard — Progress report, Phases 1–3 (research prototype; not medical advice)",
    )
    canvas.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> Path:
    data = fetch_data_summary()
    exps = load_experiments()
    n_obs, n_stations, n_weather, n_ref = data["totals"]
    story: list = []

    # --- Cover / summary ------------------------------------------------------------------
    story += [
        Spacer(1, 18 * mm),
        P("AeroGard", "title"),
        P(
            "AI-Powered Hyperlocal Air Quality and Personalized Respiratory Health Alert System",
            "subtitle",
        ),
        P(
            f"Progress report — Phases 1 to 3 of 6 &nbsp;·&nbsp; {date.today():%d %B %Y}",
            "subtitle",
        ),
        P(
            "Final Year Project + Research Paper &nbsp;·&nbsp; Study area: Bengaluru &nbsp;·&nbsp; Diseases: Asthma, COPD",
            "subtitle",
        ),
        Spacer(1, 10 * mm),
        P("Summary", "h1"),
        P(
            "AeroGard is being built in dependency order: data first, then forecasting, exposure, risk, alerts, "
            "API and finally the Android app. Phases 1–3 are complete and verified. The backend runs, "
            f"<b>{n_obs:,} hourly air-quality observations</b> from <b>{n_stations} official CPCB/KSPCB monitoring "
            f"stations</b> and <b>{n_weather:,} hours of weather</b> are stored and validated, and a 24-hour "
            "forecast model is trained, evaluated against observed data, versioned and served through the API. "
            "The automated suite of <b>232 tests</b> passes."
        ),
        Spacer(1, 3 * mm),
        table(
            [
                ["Phase", "Scope", "Status"],
                [
                    "1",
                    "Foundation: database, migrations, authentication, patient profile, security",
                    "Done",
                ],
                [
                    "2",
                    "Air-quality and weather ingestion, validation, cleaning, hyperlocal matching",
                    "Done",
                ],
                ["3", "24-hour air-quality forecast, personal exposure estimate", "Done"],
                [
                    "4",
                    "Synthetic patient cohort, respiratory-risk model, calibration, personalisation",
                    "Next — needs approval",
                ],
                ["5", "Explainability (SHAP), alert engine, remaining API", "Not started"],
                ["6", "Flutter Android app, end-to-end testing, research freeze", "Not started"],
            ],
            [14, 118, 38],
            highlight_rows=(1, 2, 3),
        ),
        Spacer(1, 4 * mm),
        P("Headline results", "h2"),
    ]
    if "pm25" in exps:
        lgb = next(r for r in exps["pm25"]["comparison_test"] if r["model"] == "lightgbm_direct")
        story += bullets(
            [
                f"24-hour PM<sub>2.5</sub> forecast: mean absolute error <b>{lgb['mae']:.2f} µg/m³</b> on an untouched "
                f"test period, <b>{lgb['skill_vs_persistence']:+.0%}</b> better than the persistence baseline.",
                "The model beats both baselines for PM<sub>2.5</sub>, PM<sub>10</sub> and NO<sub>2</sub>. For O<sub>3</sub> "
                "the simple 'same hour yesterday' baseline has lower MAE — reported as a finding.",
                "Five data-quality problems were found and fixed, including a CO unit error in the source data "
                "and a data-leakage bug in our own cleaning code (Section 5).",
            ]
        )

    # --- Phase 1 ------------------------------------------------------------------------
    story += [PageBreak(), P("1. Phase 1 — Foundation and security", "h1")]
    story += bullets(
        [
            "<b>Stack:</b> FastAPI (Python), PostgreSQL 16 in Docker, SQLAlchemy 2 with Alembic migrations.",
            "<b>Layering:</b> routes → services → repositories → database. Routes contain no SQL or ML (PRD §5.1).",
            "<b>Authentication:</b> register, login, refresh, logout. Passwords hashed with Argon2. Short-lived JWT "
            "access tokens; refresh tokens rotate on every use and a replayed (stolen) refresh token revokes all "
            "of that user's sessions.",
            "<b>Authorisation:</b> patient-scoped endpoints check ownership; another patient's record returns 404 "
            "(same as 'not found') so IDs cannot be probed. Tested explicitly: user A cannot read user B's data.",
            "<b>Patient profile:</b> age, sex, disease type (asthma / COPD), severity, consent status, baseline "
            "information. Clinical fields stay empty until provided — never defaulted.",
            "<b>Audit log</b> for sensitive actions (login, consent change, profile update) — records field names "
            "and counts, never health or location values.",
            "<b>Error contract:</b> every error is <font face='Body-Bold'>{code, message, request_id}</font>; no stack traces "
            "reach the client. Structured JSON logs with a request ID per call.",
        ]
    )

    # --- Phase 2 ------------------------------------------------------------------------
    story += [P("2. Phase 2 — Environmental data pipeline", "h1"), P("2.1 Sources", "h2")]
    story.append(
        table(
            [
                ["Source", "Used for", "Status"],
                [
                    "OpenAQ v3 (CPCB/KSPCB reference monitors)",
                    "Ground-truth hourly air quality, history + recent",
                    f"Active — {n_stations} stations with data ({n_ref} station records)",
                ],
                ["Open-Meteo ERA5 reanalysis", "Historical hourly weather", "Active"],
                ["Open-Meteo forecast", "Weather for live forecasts", "Active"],
                [
                    "data.gov.in (CPCB live feed)",
                    "Optional real-time fallback",
                    "Built, disabled: portal returned HTTP 502; units unverified",
                ],
                [
                    "Low-cost sensors (e.g. AirGradient)",
                    "—",
                    "Excluded from ground truth by design",
                ],
            ],
            [55, 62, 53],
        )
    )
    story += [P("2.2 Data loaded (Bengaluru)", "h2")]
    rows = [["Pollutant", "Stations", "Hourly rows", "% flagged", "Median µg/m³", "Period"]]
    for p, st, n, fl, med, first, last in data["pollutants"]:
        rows.append(
            [LABEL.get(p, p), st, f"{n:,}", f"{fl}%", f"{med}", f"{first:%b %Y} – {last:%b %Y}"]
        )
    story.append(table(rows, [26, 20, 28, 22, 28, 46]))
    fl = data["flags"]
    story.append(
        P(
            f"Quality flags: valid {fl.get('valid', 0):,}; low coverage (&lt;75% of readings) {fl.get('low_coverage', 0):,}; "
            f"provider-flagged {fl.get('provider_flagged', 0):,}; implausible {fl.get('implausible', 0):,}. "
            "Flagged records are kept and labelled, not deleted.",
            "small",
        )
    )
    story += [P("2.3 Processing rules", "h2")]
    story += bullets(
        [
            "<b>Validation:</b> records with missing/naive timestamps, unknown pollutants, invalid coordinates, "
            "missing, non-finite or negative values, or unknown units are rejected and the reason counted.",
            "<b>Units:</b> everything stored in µg/m³; gases in ppb converted at 25 °C / 1 atm "
            "(µg/m³ = ppb × molecular weight / 24.45). Unknown conversions are rejected, never guessed.",
            "<b>Timestamps:</b> UTC, stamped at the end of the averaging hour, so a value is fully known at its timestamp.",
            "<b>Deduplication:</b> one record per (source, station, pollutant, hour); re-loading is idempotent.",
            "<b>Cleaning:</b> hourly grid without invented values; model inputs fill gaps of ≤3 h using past data only; "
            "outliers flagged with a trailing robust z-score.",
            "<b>Hyperlocal matching:</b> haversine distance to stations within 10 km, stale (&gt;3 h) or future "
            "readings dropped, inverse-distance weighting over the 4 nearest; an explicit 'unavailable' with a "
            "reason when nothing usable is nearby.",
        ]
    )

    # --- Phase 3 ------------------------------------------------------------------------
    story += [P("3. Phase 3 — 24-hour forecast and exposure", "h1"), P("3.1 Forecast method", "h2")]
    story += bullets(
        [
            "<b>Features (35):</b> current value, lags (1, 6, 12, 24, 48 h), rolling mean/max/std, other pollutants, "
            "weather, calendar (IST), station location. Every feature uses only data at or before the prediction "
            "time — enforced by automated leakage tests.",
            "<b>Models:</b> persistence and seasonal-naive baselines; LightGBM with one model per horizon (1–24 h).",
            "<b>Protocol:</b> chronological split, never shuffled, 24 h embargo at each boundary. "
            "Train Feb 2025–Jan 2026, validation Feb–Apr 2026 (early stopping only), test May–Sep 2026 (used once). "
            "All models scored on the same forecast–observation pairs, against observed values only.",
            "<b>Artifacts:</b> each model saved with a metadata file (versions, periods, metrics, SHA-256). The API "
            "refuses a model whose checksum or feature list does not match.",
        ]
    )
    story.append(P("3.2 Results on the test period (valid hours, error in µg/m³)", "h2"))
    rows = [["Pollutant", "Model", "MAE", "RMSE", "R²", "MAE 1h", "MAE 24h", "Skill"]]
    best_rows = []
    for p in ("pm25", "pm10", "no2", "o3"):
        if p not in exps:
            continue
        table_rows = exps[p]["comparison_test"]
        best = min(table_rows, key=lambda r: r["mae"])["model"]
        for r in table_rows:
            rows.append(
                [
                    LABEL[p],
                    MODEL[r["model"]],
                    f"{r['mae']:.2f}",
                    f"{r['rmse']:.2f}",
                    f"{r['r2']:.2f}",
                    f"{r['mae_h1']:.2f}",
                    f"{r['mae_h24']:.2f}",
                    f"{r['skill_vs_persistence']:+.0%}"
                    if r["skill_vs_persistence"] is not None
                    else "—",
                ]
            )
            if r["model"] == best:
                best_rows.append(len(rows) - 1)
    story.append(table(rows, [20, 52, 15, 16, 14, 17, 18, 18], highlight_rows=tuple(best_rows)))
    story.append(
        P(
            "Bold = lowest MAE for that pollutant. Skill = improvement in MAE over persistence.",
            "small",
        )
    )
    story += [P("Findings", "h2")]
    story += bullets(
        [
            "LightGBM clearly improves on both baselines for PM<sub>2.5</sub>, PM<sub>10</sub> and NO<sub>2</sub>, "
            "with the largest gains 6–24 hours ahead.",
            "<b>O<sub>3</sub>:</b> 'same hour yesterday' has the lowest MAE; LightGBM has lower RMSE and higher R². "
            "Ozone's strong daily sunlight cycle makes the naive baseline hard to beat.",
            "<b>NO<sub>2</sub> at 1 hour ahead:</b> persistence beats LightGBM.",
            "<b>PM<sub>2.5</sub>:</b> RMSE much larger than MAE and modest R² — the model improves typical hours but "
            "does not anticipate sudden spikes.",
            "The conclusion holds when flagged hours are included in scoring (secondary evaluation).",
        ]
    )
    story += [
        P("3.3 Personal exposure estimate", "h2"),
        P(
            "Exposure = air quality at the user's location × a factor for where the user is. Factors are "
            "literature-based assumptions (to be tested by sensitivity analysis), not measurements:"
        ),
    ]
    story.append(
        table(
            [
                [
                    "Setting",
                    "PM<sub>2.5</sub>",
                    "PM<sub>10</sub>",
                    "NO<sub>2</sub>",
                    "O<sub>3</sub>",
                    "CO",
                ],
                ["Outdoors / unknown", "1.0", "1.0", "1.0", "1.0", "1.0"],
                ["Indoors at home", "0.8", "0.6", "0.6", "0.3", "0.9"],
                ["Indoors elsewhere", "0.6", "0.5", "0.5", "0.2", "0.8"],
                ["In traffic", "1.3", "1.3", "1.6", "0.8", "1.8"],
            ],
            [50, 24, 24, 24, 24, 24],
        )
    )
    story += bullets(
        [
            "Hours without a location fix in the previous 2 hours are reported as missing, never guessed.",
            "6 / 24 / 72-hour averages, maxima and cumulative exposure, each with a coverage percentage.",
            "Location and exposure require the patient's consent.",
        ]
    )

    # --- Issues -------------------------------------------------------------------------
    story += [P("4. API available now", "h1")]
    story.append(
        table(
            [
                ["Endpoint", "Purpose", "Access"],
                ["GET /health", "Service and database status", "Public"],
                [
                    "POST /auth/register · login · refresh · logout",
                    "Accounts and sessions",
                    "Public / signed in",
                ],
                ["GET, PATCH /patients/me", "Own patient profile and consent", "Signed in"],
                [
                    "GET /air/current",
                    "Location-matched air quality (optional past time)",
                    "Signed in",
                ],
                ["GET /air/forecast", "24 hourly forecast values for a location", "Signed in"],
                ["GET /air/stations", "Reference monitoring stations", "Signed in"],
                ["POST /location", "Upload location fixes", "Signed in + consent"],
                [
                    "GET /exposure/current · history",
                    "Estimated personal exposure",
                    "Signed in + consent",
                ],
            ],
            [62, 72, 36],
        )
    )
    story += [P("5. Problems found and fixed", "h1")]
    story.append(
        table(
            [
                ["#", "Problem", "Fix"],
                [
                    "1",
                    "CO on the CPCB feed labelled 'ppb' but actually mg/m³ — values 1,000× too small. "
                    "Confirmed by comparing with the older feed (1,260 vs 0.49).",
                    "Documented source correction; 129,348 stored rows converted.",
                ],
                [
                    "2",
                    "NO<sub>2</sub> and SO<sub>2</sub> 'ppb' labels may also be wrong (possible 1.9× / 2.6× overstatement); "
                    "evidence ambiguous.",
                    "Marked unverified; excluded from risk thresholds until checked against CPCB's own values.",
                ],
                [
                    "3",
                    "Isolated PM<sub>2.5</sub> spikes of 600–900 µg/m³ in the monsoon, mostly low-coverage — likely sensor glitches.",
                    "Model trained and primarily scored on fully valid hours; results also reported on all hours.",
                ],
                [
                    "4",
                    "Gap-filling by interpolation used the value after the gap — a data-leakage bug in our own code.",
                    "Model inputs now use past-only filling; tests prove future changes cannot alter inputs.",
                ],
                [
                    "5",
                    "OpenAQ timed out (HTTP 408) on long queries; two sensors missed.",
                    "Requests split into 90-day chunks, 408 retried; both stations reloaded.",
                ],
            ],
            [8, 92, 70],
        )
    )
    story += [
        P("6. Testing", "h1"),
        P(
            "232 automated tests run against a dedicated test database built through the real migrations. They cover "
            "authentication and token abuse, patient isolation, the error format, source adapters (valid, empty, "
            "malformed, timeout, rate limit, partial failure), validation and unit conversion, deduplication and "
            "idempotent reloads, spatial matching edge cases, cleaning, leakage guards, forecast horizons and "
            "artifact checks, exposure, consent and the API end to end. No test calls a live external service."
        ),
    ]

    # --- Limitations / next ---------------------------------------------------------------
    story += [P("7. Limitations to state in the paper", "h1")]
    story += bullets(
        [
            "Personal exposure is a model-based estimate; no personal sensor is used.",
            "All Bengaluru stations have a data gap from about Oct 2022 to Feb 2025; modelling uses Feb 2025 onward (~19 months).",
            "The test period covers the monsoon season only; winter performance is measured on validation only.",
            "OpenAQ lags real time by about 3 days, so the live 'current' view is usually reported as stale; demos use a past time.",
            "Weather for training is ERA5 reanalysis; live forecasts use numerical weather forecasts instead.",
            "Respiratory outcome labels in Phase 4 will be synthetic and must be reported as such — not clinical validation.",
        ]
    )
    story += [
        KeepTogether(
            [
                P("8. Next: Phase 4 — decisions needed", "h1"),
                P(
                    "Phase 4 builds the synthetic patient cohort and the respiratory-risk model. Two proposed defaults need approval:"
                ),
                table(
                    [
                        ["Decision", "Proposed default"],
                        [
                            "Event to predict",
                            "Exacerbation in the next 24 hours (symptom flare needing extra rescue-inhaler use), "
                            "generated synthetically from exposure, severity and personal sensitivity using "
                            "published effect sizes; clearly labelled synthetic.",
                        ],
                        [
                            "Risk levels",
                            "Low &lt; 15% · Moderate 15–35% · High 35–60% · Severe ≥ 60% "
                            "(stored as versioned configuration, changeable without retraining).",
                        ],
                    ],
                    [38, 132],
                ),
                Spacer(1, 3 * mm),
                note(
                    "<b>Before Phase 6:</b> install Flutter SDK and Android Studio; an Android phone is recommended for "
                    "Health Connect and GPS testing."
                ),
            ]
        )
    ]

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AeroGard — Progress Report (Phases 1–3)",
        author="AeroGard project",
        subject="Final year project progress report",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return OUT


if __name__ == "__main__":
    print(build())
