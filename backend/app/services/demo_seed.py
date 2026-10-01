"""Seed a demo account from one SYNTHETIC cohort patient (profile, places, locations, diary).

For demos and the viva only. Rows are written straight through the ORM, bypassing the API's
upload limits (7-day location backdate, 1-year health backfill), and every health record is
marked source="synthetic" so it can never be mistaken for real patient data.

The cohort keeps each hour's reported microenvironment but not the exact place, so fixes are
placed by context: indoor_other → work, in_transit → halfway between home and work, anything
else → home.
"""

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.domain.enums import ConsentStatus, HealthSource
from app.ml.cohort import ENV_CODES, OTHER, TRANSIT, Cohort
from app.models import LocationRecord
from app.repositories.health import HealthRecordRepository
from app.services.auth import AuthService

INT_COLUMNS = ("symptom_score", "rescue_inhaler_puffs", "steps")
FLOAT_COLUMNS = ("heart_rate", "spo2", "sleep_hours")


def seed_demo_patient(
    db: Session,
    cohort: Cohort,
    index: int,
    email: str,
    password: str,
    end: datetime,
    days: int = 30,
) -> dict:
    p = cohort.patients.iloc[index]
    user, profile = AuthService(db).register(email, password)
    profile.age = int(p.age)
    profile.sex = p.sex
    profile.disease_type = p.disease_type
    profile.severity = p.severity
    profile.consent_status = ConsentStatus.GRANTED
    profile.consent_updated_at = datetime.now(UTC)
    profile.baseline_information = {
        "demo": True,
        "synthetic_cohort": cohort.version,
        "cohort_patient": p.patient_id,
    }
    profile.home_label, profile.home_latitude, profile.home_longitude = (
        "Home",
        float(p.home_latitude),
        float(p.home_longitude),
    )
    if p.employed:
        profile.work_label, profile.work_latitude, profile.work_longitude = (
            "Work",
            float(p.work_latitude),
            float(p.work_longitude),
        )

    end_ts = pd.Timestamp(end).tz_convert("UTC")
    start_ts = end_ts - pd.Timedelta(days=days)
    in_window = (cohort.hours > start_ts) & (cohort.hours <= end_ts)
    if not in_window.any():
        raise ValueError(f"the cohort covers {cohort.hours[0]} → {cohort.hours[-1]}, not {end}")

    home = np.array([p.home_latitude, p.home_longitude])
    work = np.array([p.work_latitude, p.work_longitude]) if p.employed else home
    places = {OTHER: work, TRANSIT: (home + work) / 2}
    fixes = []
    for h in np.flatnonzero(in_window & cohort.location_available[index]):
        env = int(cohort.env_reported[index, h])
        lat, lon = places.get(env, home)
        fixes.append(
            {
                "id": uuid.uuid4(),
                "patient_id": profile.id,
                "timestamp": cohort.hours[h].to_pydatetime() - timedelta(minutes=15),
                "latitude": float(lat),
                "longitude": float(lon),
                "activity_context": ENV_CODES[env].value,
                "source": "device",
            }
        )
    for i in range(0, len(fixes), 500):
        db.execute(insert(LocationRecord).values(fixes[i : i + 500]).on_conflict_do_nothing())

    d = cohort.diary
    d = d[(d["patient_idx"] == index) & (d["timestamp"] > start_ts) & (d["timestamp"] <= end_ts)]
    records = []
    for row in d.itertuples(index=False):
        rec = {
            "id": uuid.uuid4(),
            "patient_id": profile.id,
            "timestamp": row.timestamp.to_pydatetime(),
            "source": HealthSource.SYNTHETIC.value,
            "exacerbation": None if pd.isna(row.exacerbation) else bool(row.exacerbation),
        }
        rec |= {c: None if pd.isna(getattr(row, c)) else int(getattr(row, c)) for c in INT_COLUMNS}
        rec |= {
            c: None if pd.isna(getattr(row, c)) else float(getattr(row, c)) for c in FLOAT_COLUMNS
        }
        records.append(rec)
    stored = HealthRecordRepository(db).insert_many(records)
    db.commit()
    return {
        "user_id": str(user.id),
        "patient_id": str(profile.id),
        "cohort_patient": p.patient_id,
        "disease_type": p.disease_type,
        "severity": p.severity,
        "locations": len(fixes),
        "health_records": stored,
        "window": f"{start_ts.isoformat()} → {end_ts.isoformat()}",
        "events_in_window": int(cohort.onset[index, in_window].sum()),
    }
