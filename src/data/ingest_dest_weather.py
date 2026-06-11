"""Ingest IEM ASOS weather for every destination airport in the flight data.

Reads the processed flight parquet, extracts the set of distinct
destinations (and optionally the origin too), and runs the multi-station
batched IEM ingest for the full 2020-present range. ORD is included by
default because flights are ORD departures and we want both endpoints'
weather; if its raw files are already on disk the cache makes that
essentially free.

After ingestion this can optionally invoke the cleaner to produce one
processed parquet per station plus a combined long table suitable for
joining onto flights.

Usage
-----
    # ingest weather for every destination + origin (default)
    python -m src.ingest_dest_weather

    # only the top-N busiest destinations
    python -m src.ingest_dest_weather --top 50

    # ingest and immediately clean + build the combined long table
    python -m src.ingest_dest_weather --also-clean
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.ingest_bts_ord import PROJECT_ROOT, month_iter
from src.data.ingest_iem_asos import (
    DEFAULT_RAW_ROOT,
    ingest_month_for_stations,
)

logger = logging.getLogger("ingest_dest_weather")

DEFAULT_FLIGHTS = (
    PROJECT_ROOT / "data" / "processed"
    / "ord_departures_bts_2020_present.parquet"
)
DEFAULT_COMBINED_OUT = (
    PROJECT_ROOT / "data" / "processed"
    / "dest_weather_iem_2020_present.parquet"
)


def discover_destinations(flights_path: Path, *, top: int | None,
                          include_origin: bool) -> list[str]:
    df = pd.read_parquet(flights_path, columns=["origin", "dest"])
    counts = df["dest"].value_counts()
    dests = list(counts.index[:top] if top else counts.index)
    if include_origin:
        for o in df["origin"].dropna().unique():
            if o not in dests:
                dests.insert(0, str(o))
    return [str(s).upper() for s in dests]


def parse_args(argv=None) -> argparse.Namespace:
    today = date.today()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--flights", type=Path, default=DEFAULT_FLIGHTS,
                   help=f"Default: {DEFAULT_FLIGHTS}")
    p.add_argument("--top", type=int, default=None,
                   help="Only ingest the top-N busiest destinations.")
    p.add_argument("--no-origin", action="store_true",
                   help="Skip including the origin (ORD) in the station list.")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--start-month", type=int, default=1)
    p.add_argument("--end-year", type=int, default=today.year)
    p.add_argument("--end-month", type=int, default=today.month)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--also-clean", action="store_true",
                   help="After ingest, run the cleaner over every station and "
                        "produce a combined long parquet for joining.")
    p.add_argument("--combined-out", type=Path, default=DEFAULT_COMBINED_OUT,
                   help=f"Default: {DEFAULT_COMBINED_OUT}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    stations = discover_destinations(
        args.flights, top=args.top, include_origin=not args.no_origin,
    )
    logger.info("found %d station(s) to ingest (%s, %s, ...)",
                len(stations), stations[0],
                stations[1] if len(stations) > 1 else "")

    today = date.today()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    months = list(month_iter(args.start_year, args.start_month,
                             args.end_year, args.end_month))

    totals = {"downloaded": 0, "cached": 0, "missing": 0, "future": 0}
    failed_months: list[str] = []
    for year, month in tqdm(months, desc="months", unit="mo"):
        try:
            counts = ingest_month_for_stations(
                stations, year, month, args.raw_root, args.force,
                args.timeout, today, args.batch_size,
            )
        except Exception:
            logger.exception("%04d-%02d: FAILED (continuing)", year, month)
            failed_months.append(f"{year:04d}-{month:02d}")
            continue
        for k, v in counts.items():
            totals[k] += v

    logger.info(
        "Ingest done. downloaded=%d cached=%d missing=%d future=%d "
        "failed_months=%d",
        totals["downloaded"], totals["cached"], totals["missing"],
        totals["future"], len(failed_months),
    )
    if failed_months:
        logger.error("Failed months: %s", ", ".join(failed_months))

    if args.also_clean:
        logger.info("Running cleaner over every ingested station...")
        rc = subprocess.call([
            sys.executable, "-m", "src.clean_iem_asos",
            "--all",
            "--raw-root", str(args.raw_root),
            "--combined-out", str(args.combined_out),
        ])
        if rc != 0:
            logger.error("Cleaner exited %d", rc)
            return rc

    return 2 if failed_months else 0


if __name__ == "__main__":
    sys.exit(main())
