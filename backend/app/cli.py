"""AeroGard data jobs. Run from backend/:

python -m app.cli sync-stations
python -m app.cli backfill-aq --start 2025-02-18 --end 2026-09-24
python -m app.cli backfill-weather --start 2025-02-18 --end 2026-09-24
python -m app.cli fetch-forecast
python -m app.cli summary
python -m app.cli probe-datagovin
"""

import argparse
import sys
from datetime import UTC, date, datetime, time

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import get_sessionmaker
from app.domain.pollutants import Pollutant
from app.ingestion import datagovin, openaq, openmeteo
from app.ingestion.base import SourceError
from app.models import IngestionRun
from app.repositories.environment import ObservationRepository
from app.services.ingestion import IngestionService


def _openaq() -> openaq.OpenAQClient:
    s = get_settings()
    return openaq.OpenAQClient(s.openaq_api_key, s.openaq_base_url, timeout=s.http_timeout_seconds)


def _openmeteo() -> openmeteo.OpenMeteoClient:
    s = get_settings()
    return openmeteo.OpenMeteoClient(
        s.open_meteo_archive_url, s.open_meteo_forecast_url, timeout=s.http_timeout_seconds
    )


def _print_run(run: IngestionRun) -> None:
    print(
        f"\nrun {run.id} [{run.source}/{run.job}] status={run.status}\n"
        f"  fetched={run.records_fetched} valid={run.records_valid} "
        f"rejected={run.records_rejected} duplicates={run.duplicates_dropped} "
        f"written={run.records_written}"
    )
    if run.rejection_reasons:
        print(f"  rejection reasons: {run.rejection_reasons}")
    for err in run.errors[:10]:
        print(f"  error: {err}")
    if len(run.errors) > 10:
        print(f"  ... and {len(run.errors) - 10} more errors")


def cmd_sync_stations(_: argparse.Namespace) -> int:
    s = get_settings()
    client = _openaq()
    with get_sessionmaker()() as db:
        found = IngestionService(db).sync_openaq_stations(
            client, s.study_area_latitude, s.study_area_longitude, s.study_area_radius_km
        )
    ref = [x for x in found if x.is_reference]
    print(
        f"{len(found)} locations within {s.study_area_radius_km} km of {s.study_area_name}; "
        f"{len(ref)} reference monitors:"
    )
    for x in sorted(ref, key=lambda x: x.name):
        print(f"  {x.station_id:18} {x.name[:45]:45} last={x.last_utc}")
    return 0


def cmd_backfill_aq(args: argparse.Namespace) -> int:
    s = get_settings()
    start = datetime.combine(args.start, time.min, UTC)
    end = datetime.combine(args.end, time.max, UTC).replace(microsecond=0)
    pollutants = (
        tuple(Pollutant(p) for p in args.pollutants.split(","))
        if args.pollutants
        else tuple(Pollutant)
    )
    client = _openaq()
    with get_sessionmaker()() as db:
        svc = IngestionService(db)
        stations = svc.sync_openaq_stations(
            client, s.study_area_latitude, s.study_area_longitude, s.study_area_radius_km
        )
        if args.station:
            stations = [x for x in stations if x.station_id in set(args.station)]
            if not stations:
                print(f"no matching station for {args.station}")
                return 1
        run = svc.ingest_openaq_hours(client, stations, start, end, pollutants)
        _print_run(run)
    return 0 if run.status != "failed" else 1


def cmd_backfill_weather(args: argparse.Namespace) -> int:
    s = get_settings()
    with get_sessionmaker()() as db:
        run = IngestionService(db).ingest_weather_archive(
            _openmeteo(), s.study_area_latitude, s.study_area_longitude, args.start, args.end
        )
        _print_run(run)
    return 0 if run.status != "failed" else 1


def cmd_fetch_forecast(_: argparse.Namespace) -> int:
    s = get_settings()
    with get_sessionmaker()() as db:
        run = IngestionService(db).ingest_weather_forecast(
            _openmeteo(), s.study_area_latitude, s.study_area_longitude
        )
        _print_run(run)
    return 0 if run.status != "failed" else 1


def cmd_summary(_: argparse.Namespace) -> int:
    with get_sessionmaker()() as db:
        rows = ObservationRepository(db).coverage_summary()
    if not rows:
        print("No observations stored yet. Run backfill-aq first.")
        return 0
    print(f"{'station':18} {'pollutant':9} {'rows':>7} {'flagged':>8}  first -> last")
    for r in rows:
        print(
            f"{r['station_id']:18} {r['pollutant']:9} {r['rows']:>7} {r['flagged']:>8}  "
            f"{r['first']:%Y-%m-%d %H:%M} -> {r['last']:%Y-%m-%d %H:%M}"
        )
    total = sum(r["rows"] for r in rows)
    print(f"\n{total} observations across {len({r['station_id'] for r in rows})} stations")
    return 0


def cmd_probe_datagovin(_: argparse.Namespace) -> int:
    """Fetch the live CPCB feed and print it, WITHOUT storing, to verify units by hand."""
    s = get_settings()
    client = datagovin.DataGovInClient(
        s.data_gov_in_api_key, s.data_gov_in_base_url, s.data_gov_in_aqi_resource_id, timeout=60
    )
    try:
        records = client.fetch_current(s.study_area_name)
    except SourceError as exc:
        print(f"data.gov.in unavailable: {exc}")
        return 1
    print(f"{len(records)} records. First 10:")
    for r in records[:10]:
        print(
            f"  {r.get('station', '')[:40]:40} {r.get('pollutant_id', ''):6} "
            f"min={r.get('min_value')} avg={r.get('avg_value')} max={r.get('max_value')} "
            f"at {r.get('last_update')}"
        )
    return 0


def _date(v: str) -> date:
    return date.fromisoformat(v)


def main(argv: list[str] | None = None) -> int:
    configure_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="AeroGard data jobs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sync-stations", help="discover monitoring stations near the study area")
    p = sub.add_parser("backfill-aq", help="ingest hourly OpenAQ observations for a date range")
    p.add_argument("--start", type=_date, required=True)
    p.add_argument("--end", type=_date, required=True)
    p.add_argument("--pollutants", help="comma-separated, e.g. pm25,pm10 (default: all)")
    p.add_argument(
        "--station", action="append", help="limit to a station id, e.g. openaq:5644 (repeatable)"
    )
    p = sub.add_parser("backfill-weather", help="ingest Open-Meteo ERA5 hourly weather")
    p.add_argument("--start", type=_date, required=True)
    p.add_argument("--end", type=_date, required=True)
    sub.add_parser("fetch-forecast", help="store the current 48 h Open-Meteo forecast")
    sub.add_parser("summary", help="rows per station and pollutant")
    sub.add_parser("probe-datagovin", help="print the live CPCB feed without storing it")

    args = parser.parse_args(argv)
    handlers = {
        "sync-stations": cmd_sync_stations,
        "backfill-aq": cmd_backfill_aq,
        "backfill-weather": cmd_backfill_weather,
        "fetch-forecast": cmd_fetch_forecast,
        "summary": cmd_summary,
        "probe-datagovin": cmd_probe_datagovin,
    }
    try:
        return handlers[args.command](args)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
