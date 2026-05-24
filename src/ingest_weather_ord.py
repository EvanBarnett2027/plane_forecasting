"""Ingest historical ASOS/METAR weather observations for ORD.

Data source
-----------
Official Iowa Environmental Mesonet (IEM) ASOS download service:

    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

IEM rehosts the authoritative NWS/FAA ASOS observations for O'Hare and is
the standard programmatic source for long station histories. Observations
are sub-hourly (routine METAR plus SPECI specials); regularisation to a
strict hourly series happens in ``src.clean_weather_ord``.

This mirrors the flight pipeline (:mod:`src.ingest_bts_ord`): one raw file
per month under ``data/raw/iem_asos_ord/YYYY_MM.csv``, cached so re-runs
skip months already on disk unless ``--force`` is given. The default range
is 2020-01 through the current month (IEM is near-real-time, so unlike BTS
there is no multi-month publication lag).

Usage
-----
    python -m src.ingest_weather_ord --start-year 2020 --start-month 1 --station ORD

Run ``python -m src.ingest_weather_ord --help`` for all options.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

# Reuse the month iterator + project root so both pipelines stay in sync.
from src.ingest_bts_ord import PROJECT_ROOT, month_iter

logger = logging.getLogger("ingest_weather_ord")

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "iem_asos_ord"


def _month_bounds(year: int, month: int, not_after: date) -> tuple[date, date]:
    """Return (first_day, end_exclusive) to request for a month.

    IEM's ``asos.py`` treats the end date as **exclusive**, so to include
    the last day of the month the end bound must be the day *after* it.
    The final (current) month is capped at ``not_after`` so we don't ask
    IEM for observations that don't exist yet.
    """
    first = date(year, month, 1)
    last_dom = calendar.monthrange(year, month)[1]
    last_wanted = date(year, month, last_dom)
    if last_wanted > not_after:
        last_wanted = not_after
    end_exclusive = last_wanted + timedelta(days=1)
    return first, end_exclusive


def _fetch_csv_text(params: dict, timeout: int, retries: int) -> str:
    """GET the IEM CSV with retry/backoff on transient network errors."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                IEM_URL, params=params, headers=REQUEST_HEADERS,
                timeout=(15, timeout),
            )
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as err:
            last_err = err
            if attempt < retries:
                backoff = min(60, 5 * 2 ** (attempt - 1))
                logger.warning(
                    "  attempt %d/%d failed (%s); retrying in %ds",
                    attempt, retries, type(err).__name__, backoff,
                )
                time.sleep(backoff)
    raise RuntimeError(f"giving up after {retries} attempts") from last_err


def ingest_month(
    year: int,
    month: int,
    station: str,
    raw_dir: Path,
    force: bool,
    timeout: int,
    not_after: date,
    retries: int = 4,
) -> str:
    """Ingest one month of raw ASOS observations. Returns a status string."""
    out = raw_dir / f"{year:04d}_{month:02d}.csv"
    if out.exists() and not force:
        logger.info("%04d-%02d: cached -> %s (skip; use --force to refresh)",
                    year, month, out.name)
        return "cached"

    first, end_excl = _month_bounds(year, month, not_after)
    if first > not_after:
        logger.info("%04d-%02d: in the future, skipping", year, month)
        return "unavailable"

    params = {
        "station": station.upper(),
        "data": "all",
        "year1": first.year, "month1": first.month, "day1": first.day,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "tz": "Etc/UTC",        # store UTC; local conversion happens in cleaning
        "format": "onlycomma",
        "latlon": "no",
        "missing": "empty",     # blank (not "M") for missing values
        "trace": "0.0001",      # trace precip 'T' -> small float
        "direct": "yes",
    }
    logger.info("%04d-%02d: downloading %s..%s", year, month,
                first, end_excl - timedelta(days=1))
    text = _fetch_csv_text(params, timeout, retries)

    # IEM prefixes a short comment block to empty results; a healthy file
    # starts with the "station,valid,..." header.
    if "station,valid" not in text.split("\n", 1)[0]:
        logger.warning("%04d-%02d: unexpected response (no header); skipping",
                        year, month)
        return "unavailable"

    out.write_text(text)
    n_rows = max(text.count("\n") - 1, 0)
    logger.info("%04d-%02d: wrote %s (%d obs)", year, month, out.name, n_rows)
    return "downloaded"


def parse_args(argv=None) -> argparse.Namespace:
    today = date.today()
    p = argparse.ArgumentParser(
        description="Ingest historical IEM ASOS/METAR weather for a station "
                    "(default ORD), one month at a time.",
    )
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--start-month", type=int, default=1,
                   choices=range(1, 13), metavar="1-12")
    p.add_argument("--end-year", type=int, default=today.year,
                   help="Default: current year.")
    p.add_argument("--end-month", type=int, default=today.month,
                   choices=range(1, 13), metavar="1-12",
                   help="Default: current month.")
    p.add_argument("--station", default="ORD",
                   help="ASOS station id (default: ORD).")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help=f"Default: {DEFAULT_RAW_DIR}")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the monthly file already exists.")
    p.add_argument("--timeout", type=int, default=180,
                   help="Per-request read timeout in seconds (default: 180).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()

    if (args.start_year, args.start_month) > (args.end_year, args.end_month):
        logger.error("Start month is after end month; nothing to do.")
        return 1

    logger.info(
        "Ingesting %s weather %04d-%02d -> %04d-%02d into %s",
        args.station.upper(), args.start_year, args.start_month,
        args.end_year, args.end_month, args.raw_dir,
    )

    counts = {"downloaded": 0, "cached": 0, "unavailable": 0}
    failed: list[str] = []
    months = list(month_iter(args.start_year, args.start_month,
                             args.end_year, args.end_month))
    for year, month in tqdm(months, desc="months", unit="mo"):
        try:
            status = ingest_month(
                year, month, args.station, args.raw_dir,
                args.force, args.timeout, today,
            )
        except Exception:
            logger.exception("%04d-%02d: FAILED (continuing)", year, month)
            failed.append(f"{year:04d}-{month:02d}")
            continue
        counts[status] += 1

    logger.info(
        "Done. downloaded=%d cached=%d unavailable=%d failed=%d",
        counts["downloaded"], counts["cached"], counts["unavailable"],
        len(failed),
    )
    if failed:
        logger.error("Failed months (re-run the same command to retry): %s",
                      ", ".join(failed))
        return 2
    logger.info("Next: python -m src.clean_weather_ord  (combine + clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
