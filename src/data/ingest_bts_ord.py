"""Ingest historical BTS Airline On-Time Performance data for ORD departures.

Data source
-----------
Official BTS / TranStats "Airline On-Time Performance Data" (table
``T_ONTIME_REPORTING``). We use the official pre-zipped monthly files BTS
publishes for programmatic download:

    https://transtats.bts.gov/PREZIP/
        On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip

Each zip contains a single CSV with every reportable field for *all* US
airports for that month. The TranStats web download form (DL_SelectFields)
only builds these same pre-zipped files behind the scenes, so the PREZIP
path is the official endpoint and needs no form parameters. Because the
files are nationwide, the ORD-origin filter is applied client-side.

No Kaggle / third-party mirror is used.

Usage
-----
    python -m src.ingest_bts_ord --start-year 2020 --start-month 1 --origin ORD

Run ``python -m src.ingest_bts_ord --help`` for all options.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger("ingest_bts_ord")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BTS_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Browser-like UA: the BTS host rejects some default client UAs.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# Fields requested / kept (in this order) when present in the source file.
FIELDS = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "DOT_ID_Reporting_Airline",
    "IATA_CODE_Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginCityName",
    "OriginState",
    "Dest",
    "DestCityName",
    "DestState",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "ArrivalDelayGroups",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

# src/data/ingest_bts_ord.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bts_ontime_ord"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def month_iter(start_year: int, start_month: int, end_year: int, end_month: int):
    """Yield (year, month) tuples inclusive from start to end."""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def raw_path(raw_dir: Path, year: int, month: int, fmt: str) -> Path:
    return raw_dir / f"{year:04d}_{month:02d}.{fmt}"


def _fetch_zip_bytes(url: str, timeout: int, retries: int) -> bytes | None:
    """GET the BTS zip with retry/backoff on transient network errors.

    Returns the raw bytes, or ``None`` if BTS signals the month is not yet
    published (HTTP 404). The (connect, read) timeout tuple keeps a slow
    BTS response from hanging forever; large monthly files are streamed.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=(15, timeout),
                stream=True,
            ) as resp:
                if resp.status_code == 404:
                    logger.info("  not published yet (HTTP 404)")
                    return None
                resp.raise_for_status()
                return resp.content
        except (requests.exceptions.RequestException,) as err:
            last_err = err
            if attempt < retries:
                backoff = min(60, 5 * 2 ** (attempt - 1))
                logger.warning(
                    "  attempt %d/%d failed (%s); retrying in %ds",
                    attempt, retries, type(err).__name__, backoff,
                )
                time.sleep(backoff)
    raise RuntimeError(
        f"giving up after {retries} attempts: {url}"
    ) from last_err


def download_month(year: int, month: int, timeout: int,
                   retries: int = 4) -> pd.DataFrame | None:
    """Download one month of nationwide BTS data.

    Returns the full month DataFrame, or ``None`` if BTS has not yet
    published this month (HTTP 404 or a non-zip response).
    """
    url = BTS_URL_TEMPLATE.format(year=year, month=month)
    logger.info("  GET %s", url)
    content = _fetch_zip_bytes(url, timeout, retries)
    if content is None:
        return None

    if not content[:2] == b"PK":
        # BTS returns an HTML error page (HTTP 200) for months not yet released.
        logger.info("  not published yet (non-zip response, %d bytes)", len(content))
        return None

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV inside zip for {year}-{month:02d}")
        with zf.open(csv_names[0]) as fh:
            # The BTS CSV has a trailing comma producing one unnamed column;
            # restricting usecols to known fields drops it cleanly.
            header = pd.read_csv(fh, nrows=0, encoding="latin-1")
        keep = [c for c in FIELDS if c in header.columns]
        missing = [c for c in FIELDS if c not in header.columns]
        if missing:
            logger.warning("  source missing %d field(s): %s", len(missing), missing)
        with zf.open(csv_names[0]) as fh:
            df = pd.read_csv(
                fh,
                usecols=keep,
                encoding="latin-1",
                low_memory=False,
            )
    return df[keep]


def ingest_month(
    year: int,
    month: int,
    origin: str,
    raw_dir: Path,
    fmt: str,
    force: bool,
    timeout: int,
) -> str:
    """Ingest a single month. Returns a status string for summary logging."""
    out = raw_path(raw_dir, year, month, fmt)
    if out.exists() and not force:
        logger.info("%04d-%02d: cached -> %s (skip; use --force to refresh)",
                    year, month, out.name)
        return "cached"

    logger.info("%04d-%02d: downloading...", year, month)
    df = download_month(year, month, timeout)
    if df is None:
        return "unavailable"

    total = len(df)
    df = df[df["Origin"].astype("string").str.upper() == origin.upper()]
    logger.info("%04d-%02d: %d total rows, %d %s-origin rows",
                year, month, total, len(df), origin.upper())

    if fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    logger.info("%04d-%02d: wrote %s", year, month, out)
    return "downloaded"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    today = date.today()
    p = argparse.ArgumentParser(
        description="Ingest historical BTS On-Time Performance data for a "
                    "given origin airport (default ORD), one month at a time.",
    )
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--start-month", type=int, default=1, choices=range(1, 13),
                   metavar="1-12")
    p.add_argument("--end-year", type=int, default=today.year,
                   help="Default: current year. Months BTS has not yet "
                        "published are detected and skipped automatically.")
    p.add_argument("--end-month", type=int, default=today.month,
                   choices=range(1, 13), metavar="1-12",
                   help="Default: current month.")
    p.add_argument("--origin", default="ORD",
                   help="Origin airport IATA code to keep (default: ORD).")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help=f"Default: {DEFAULT_RAW_DIR}")
    p.add_argument("--format", dest="fmt", choices=["parquet", "csv"],
                   default="parquet", help="Raw monthly file format.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the monthly file already exists.")
    p.add_argument("--timeout", type=int, default=180,
                   help="Per-request timeout in seconds (default: 180).")
    p.add_argument("--stop-on-unavailable", action="store_true",
                   help="Stop at the first month BTS has not published "
                        "(months are sequential, so later months are too).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.raw_dir.mkdir(parents=True, exist_ok=True)

    if (args.start_year, args.start_month) > (args.end_year, args.end_month):
        logger.error("Start month is after end month; nothing to do.")
        return 1

    logger.info(
        "Ingesting %s departures %04d-%02d -> %04d-%02d into %s",
        args.origin.upper(), args.start_year, args.start_month,
        args.end_year, args.end_month, args.raw_dir,
    )

    counts = {"downloaded": 0, "cached": 0, "unavailable": 0}
    failed: list[str] = []
    months = list(month_iter(args.start_year, args.start_month,
                             args.end_year, args.end_month))
    for year, month in tqdm(months, desc="months", unit="mo"):
        try:
            status = ingest_month(
                year, month, args.origin, args.raw_dir,
                args.fmt, args.force, args.timeout,
            )
        except Exception:
            # One bad month must not abort a multi-year backfill; record it,
            # keep going, and re-running later resumes from the gap (cached
            # months are skipped).
            logger.exception("%04d-%02d: FAILED (continuing)", year, month)
            failed.append(f"{year:04d}-{month:02d}")
            continue
        counts[status] += 1
        if status == "unavailable" and args.stop_on_unavailable:
            logger.info("Stopping: %04d-%02d not yet published by BTS.",
                        year, month)
            break

    logger.info(
        "Done. downloaded=%d cached=%d unavailable=%d failed=%d",
        counts["downloaded"], counts["cached"], counts["unavailable"],
        len(failed),
    )
    if failed:
        logger.error("Failed months (re-run the same command to retry): %s",
                      ", ".join(failed))
        return 2
    logger.info("Next: python -m src.clean_bts_ord  (combine + clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
