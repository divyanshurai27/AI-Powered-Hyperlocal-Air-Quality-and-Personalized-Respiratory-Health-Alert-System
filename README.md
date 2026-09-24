# AeroGard

Hyperlocal air-quality forecasting and personalised respiratory-risk alerts for **asthma and COPD**.

> **Research prototype.** AeroGard does not diagnose, prescribe, or change medication. Personal
> exposure is a model-derived *estimate*, not a direct measurement. Risk outputs from synthetic or
> public data are not clinically validated. See the PRD's §2.3 and §51.

Pipeline: `24h AQ forecast → exposure estimate → respiratory-risk probability → calibrated risk level → alert`

Stack: FastAPI · PostgreSQL 16 · SQLAlchemy 2 + Alembic · Python ML · Flutter (Android)

## Repository layout

```
backend/        FastAPI app (api → services → repositories → db), Alembic migrations, tests
ml/             ingestion, cleaning, features, forecasting, exposure, risk, calibration, explainability
flutter_app/    Android client (Phase 6)
data_contracts/ canonical schemas shared by ingestion and API
scripts/        operational scripts (DB init, backfills)
docs/           architecture, experiment records
```

## Quick start (Windows PowerShell)

```powershell
# 1. Python environment (3.12+; developed on 3.14)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e "backend[dev,ml]"

# 2. Configuration
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(64))"   # paste into JWT_SECRET_KEY

# 3. Database (creates both `aerogard` and `aerogard_test`)
docker compose up -d db

# 4. Migrate and run
cd backend
alembic upgrade head
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API and http://localhost:8000/health for the health check.

## Loading data and training

```powershell
cd backend
python -m app.cli backfill-aq --start 2025-02-18 --end 2026-09-25     # ~1 h, resumable
python -m app.cli backfill-weather --start 2025-02-18 --end 2026-09-25
python -m app.cli summary
cd ..
python ml/forecasting/train_aq_forecast.py --pollutant pm25 --version 1.1.0
```

Methods, data caveats and results: [`docs/data_and_methods.md`](docs/data_and_methods.md).

## Tests

```powershell
cd backend
pytest              # unit + integration (integration needs the db container running)
ruff check .
```

Integration tests run against `aerogard_test`. The schema is built through Alembic, so every
test run also checks the migrations.

## API

| Method | Path | Auth |
|---|---|---|
| GET | `/health` | none |
| POST | `/api/v1/auth/register` | none |
| POST | `/api/v1/auth/login` | none |
| POST | `/api/v1/auth/refresh` | refresh token |
| POST | `/api/v1/auth/logout` | bearer |
| GET / PATCH | `/api/v1/patients/me` | bearer |
| GET | `/api/v1/air/current?lat&lon[&at]` | bearer |
| GET | `/api/v1/air/forecast?lat&lon&pollutant[&at]` | bearer |
| GET | `/api/v1/air/stations` | bearer |
| POST | `/api/v1/location` | bearer + consent |
| GET | `/api/v1/exposure/current`, `/exposure/history` | bearer + consent |

Errors always look like this:
`{"detail": {"code": "INVALID_TOKEN", "message": "...", "request_id": "..."}}`.

## Build status

- [x] Phase 1: foundation, auth, patient profile
- [x] Phase 2: AQ and weather ingestion, cleaning, spatial matching
- [x] Phase 3: 24h AQ forecasting and exposure engine
- [ ] Phase 4: health data, risk model, calibration, personalisation
- [ ] Phase 5: explainability, alerts, full API
- [ ] Phase 6: Flutter app, end-to-end testing, research freeze
