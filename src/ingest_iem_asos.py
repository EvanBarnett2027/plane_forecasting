"""Ingest historical ASOS/METAR observations from IEM for one or many stations.

Data source
-----------
Official Iowa Environmental Mesonet (IEM) ASOS download service, which
rehosts the authoritative NWS/FAA ASOS observations:

    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

Layout
------
One raw CSV per (station, month) under::

    data/raw/iem_asos/{STATION}/{YYYY_MM}.csv

Behaviour
---------
* Default range: 2020-01 through the current month (IEM is near-real-time,
  so the partial current month is capped at "today" automatically).
* IEM's ``day2`` is **exclusive**, so we pass first-of-next-month to get
  the full month.
* Per-station monthly files are cached; existing files are skipped unless
  ``--force`` is given. Retries with exponential backoff absorb transient
  network errors. A single failed (station, month) does not abort the run
  - re-running resumes from the gaps.
* When more than one station is requested, the script issues one IEM
  request per *batch* of stations per month using the API's repeated
  ``station=`` parameter, then splits the response per station. This drops
  the request count from N_stations*N_months to ``ceil(N/batch)*N_months``.

Usage
-----
    # single station
    python -m src.ingest_iem_asos --station ORD

    # several stations explicitly
    python -m src.ingest_iem_asos --stations ORD,LAX,DEN,JFK

    # stations from a text file (one id per line; '#' comments allowed)
    python -m src.ingest_iem_asos --stations-file my_stations.txt --batch-size 25

Run ``python -m src.ingest_iem_asos --help`` for all options.
"""

from __future__ import annotations

import argparse
import calendar
import io
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.ingest_bts_ord import PROJECT_ROOT, month_iter

logger = logging.getLogger("ingest_iem_asos")

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "iem_asos"

# IATA -> IEM station id, for the small set of airports where IEM keys the
# station by ICAO (Hawaii / Alaska / US Caribbean) or by the FAA ASOS site
# id rather than the airline-facing IATA code. When you query IEM with the
# IATA code these silently drop out of the multi-station response.
#
# Files are still written under the IATA name (matching the flight data's
# ``dest`` column) so the join works without further mapping.
STATION_ALIASES: dict[str, str] = {
    # Hawaii
    "HNL": "PHNL", "OGG": "PHOG", "KOA": "PHKO",
    "LIH": "PHLI", "ITO": "PHTO",
    # Alaska
    "ANC": "PANC", "FAI": "PAFA",
    # US territories
    "SJU": "TJSJ", "STT": "TIST", "STX": "TISX",
    # Lower-48 IATA != ASOS site id
    "FCA": "GPI",   # Glacier Park Intl
    "SCE": "UNV",   # University Park / State College
    "HHH": "HXD",   # Hilton Head
    "MQT": "SAW",   # Sawyer Intl / Marquette
    "HDN": "KHDN",  # Yampa Valley / Hayden (full ICAO works for this one)
}


def to_iem_id(iata: str) -> str:
    return STATION_ALIASES.get(iata.upper(), iata.upper())


def to_iata_id(iem: str) -> str:
    """Inverse of STATION_ALIASES (one-to-one)."""
    rev = {v: k for k, v in STATION_ALIASES.items()}
    return rev.get(iem.upper(), iem.upper())


def station_dir(raw_root: Path, station: str) -> Path:
    return raw_root / station.upper()


def station_month_path(raw_root: Path, station: str,
                       year: int, month: int) -> Path:
    return station_dir(raw_root, station) / f"{year:04d}_{month:02d}.csv"


def _month_bounds(year: int, month: int, not_after: date) -> tuple[date, date]:
    """Return (first_day, end_exclusive). See module docstring re: IEM day2."""
    first = date(year, month, 1)
    last_dom = calendar.monthrange(year, month)[1]
    last_wanted = min(date(year, month, last_dom), not_after)
    return first, last_wanted + timedelta(days=1)


def _fetch_csv_text(params: list[tuple[str, str]], timeout: int,
                    retries: int) -> str:
    """GET the IEM CSV with retry/backoff on transient errors."""
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


def _build_params(stations: list[str], first: date,
                  end_excl: date) -> list[tuple[str, str]]:
    """Build the IEM query as a list of tuples so ``station`` can repeat."""
    params: list[tuple[str, str]] = [("station", s.upper()) for s in stations]
    params.extend([
        ("data", "all"),
        ("year1", str(first.year)),
        ("month1", str(first.month)),
        ("day1", str(first.day)),
        ("year2", str(end_excl.year)),
        ("month2", str(end_excl.month)),
        ("day2", str(end_excl.day)),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "empty"),
        ("trace", "0.0001"),
        ("direct", "yes"),
    ])
    return params


def download_batch(stations: list[str], year: int, month: int,
                   not_after: date, timeout: int,
                   retries: int = 4) -> dict[str, str]:
    """Download one month for a *batch* of stations in one IEM request.

    Stations are translated through ``STATION_ALIASES`` for the IEM query
    (e.g. HNL -> PHNL) and translated back for the returned dict, so
    callers always see their original (IATA) ids. The ``station`` column
    inside each returned CSV is rewritten to the IATA id too, so files on
    disk stay keyed by the same id the flight data uses.

    Returns ``{iata_station: csv_text}``. Stations absent from the
    response (truly unknown to IEM) are omitted.
    """
    first, end_excl = _month_bounds(year, month, not_after)
    if first > not_after:
        return {}

    iem_ids = [to_iem_id(s) for s in stations]
    iem_to_iata = {to_iem_id(s): s.upper() for s in stations}
    params = _build_params(iem_ids, first, end_excl)
    text = _fetch_csv_text(params, timeout, retries)

    header_line = text.split("\n", 1)[0]
    if "station,valid" not in header_line:
        logger.warning("  unexpected response (no header) for batch "
                        "starting with %s..", stations[0])
        return {}

    df = pd.read_csv(io.StringIO(text), low_memory=False,
                     na_values=["M", "null", ""])
    df.columns = df.columns.str.strip()
    # Rewrite IEM ids back to the IATA ids the rest of the pipeline uses.
    df["station"] = df["station"].astype("string").str.upper().map(
        lambda s: iem_to_iata.get(s, s))

    out: dict[str, str] = {}
    for station, group in df.groupby("station", sort=False):
        buf = io.StringIO()
        group.to_csv(buf, index=False)
        out[str(station).upper()] = buf.getvalue()
    return out


def ingest_month_for_stations(
    stations: list[str],
    year: int,
    month: int,
    raw_root: Path,
    force: bool,
    timeout: int,
    not_after: date,
    batch_size: int,
    retries: int = 4,
) -> dict[str, int]:
    """Ingest one month across many stations using batched IEM requests.

    Returns counts summed across stations for this month:
    ``{downloaded, cached, missing, future}``.
    """
    counts = {"downloaded": 0, "cached": 0, "missing": 0, "future": 0}
    if date(year, month, 1) > not_after:
        counts["future"] = len(stations)
        return counts

    # Determine which stations still need this month on disk.
    todo: list[str] = []
    for s in stations:
        path = station_month_path(raw_root, s, year, month)
        if path.exists() and not force:
            counts["cached"] += 1
        else:
            todo.append(s)

    if not todo:
        return counts

    n_batches = -(-len(todo) // batch_size)
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        logger.info("%04d-%02d: batch %d/%d -> %d stations (%s..)",
                    year, month, (i // batch_size) + 1, n_batches,
                    len(batch), batch[0])
        result = download_batch(batch, year, month, not_after, timeout, retries)
        returned = set(result.keys())
        for s in batch:
            if s not in returned:
                logger.warning("%04d-%02d: station %s not in IEM response",
                                year, month, s)
                counts["missing"] += 1
                continue
            path = station_month_path(raw_root, s, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result[s])
            counts["downloaded"] += 1
    return counts


def _parse_stations(args: argparse.Namespace) -> list[str]:
    if args.stations_file:
        text = Path(args.stations_file).read_text()
        stations = [ln.strip().upper() for ln in text.splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]
    elif args.stations:
        stations = [s.strip().upper() for s in args.stations.split(",")
                    if s.strip()]
    else:
        stations = [args.station.upper()]
    # dedupe preserving order
    seen, out = set(), []
    for s in stations:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    today = date.today()
    p = argparse.ArgumentParser(
        description="Ingest IEM ASOS/METAR weather for one or many stations, "
                    "one month at a time, with multi-station batching.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--station", default="ORD",
                     help="Single station id (default: ORD).")
    src.add_argument("--stations",
                     help="Comma-separated list of station ids.")
    src.add_argument("--stations-file",
                     help="Path to a text file with one station id per line.")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--start-month", type=int, default=1,
                   choices=range(1, 13), metavar="1-12")
    p.add_argument("--end-year", type=int, default=today.year)
    p.add_argument("--end-month", type=int, default=today.month,
                   choices=range(1, 13), metavar="1-12")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                   help=f"Default: {DEFAULT_RAW_ROOT}")
    p.add_argument("--batch-size", type=int, default=25,
                   help="Stations per IEM request when more than one is "
                        "requested (default: 25).")
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
    stations = _parse_stations(args)
    today = date.today()
    args.raw_root.mkdir(parents=True, exist_ok=True)

    if (args.start_year, args.start_month) > (args.end_year, args.end_month):
        logger.error("Start month is after end month; nothing to do.")
        return 1

    logger.info("Ingesting %d station(s) %04d-%02d -> %04d-%02d into %s "
                "(batch_size=%d)",
                len(stations), args.start_year, args.start_month,
                args.end_year, args.end_month, args.raw_root, args.batch_size)

    totals = {"downloaded": 0, "cached": 0, "missing": 0, "future": 0}
    failed_months: list[str] = []
    months = list(month_iter(args.start_year, args.start_month,
                             args.end_year, args.end_month))
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

    logger.info("Done. downloaded=%d cached=%d missing=%d future=%d "
                "failed_months=%d",
                totals["downloaded"], totals["cached"], totals["missing"],
                totals["future"], len(failed_months))
    if failed_months:
        logger.error("Failed months (re-run to retry): %s",
                      ", ".join(failed_months))
        return 2
    if totals["missing"]:
        logger.warning("%d (station, month) pairs were missing from IEM "
                        "responses (likely unknown station ids).",
                        totals["missing"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
