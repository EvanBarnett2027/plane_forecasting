"""Paced AeroDataBox backfill for the rolling recent-departures cache.

The dashboard refreshes a small recent window each run. This script is for
seeding a fuller cache while staying under the API.Market gateway's
~1 request/second limit: it fetches small time chunks and sleeps between
calls. Each chunk maps to one FIDS request, so keep ``--chunk-h`` at or below
the 12 h FIDS window cap.

Example
-------
    .venv/bin/python -m src.backfill_recent_cache --lookback-h 168
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingest_bts_ord import PROJECT_ROOT
from src.live_flights import (
    DEFAULT_RECENT_CACHE,
    DEFAULT_RECENT_CACHE_RETENTION_H,
    ORIGIN_DEFAULT,
    fetch_recent_departures,
    load_recent_departures_cache,
    merge_recent_departures_cache,
    write_recent_departures_cache,
)

logger = logging.getLogger("backfill_recent_cache")


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the AeroDataBox key from an arg, ``aerodatabox.key``, or env."""
    if explicit:
        return explicit.strip()
    key_file = PROJECT_ROOT / "aerodatabox.key"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key
    env = os.environ.get("AERODATABOX_KEY")
    return env.strip() if env else None


def _utc_timestamp(value: str | datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def iter_backfill_windows(
    now: str | datetime | pd.Timestamp | None,
    *,
    lookback_h: float,
    chunk_h: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return newest-first ``(start, end)`` UTC windows for a backfill."""
    if lookback_h <= 0:
        raise ValueError("lookback_h must be positive")
    if chunk_h <= 0:
        raise ValueError("chunk_h must be positive")

    end = _utc_timestamp(now)
    earliest = end - pd.Timedelta(hours=lookback_h)
    step = pd.Timedelta(hours=chunk_h)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while end > earliest:
        start = max(earliest, end - step)
        windows.append((start, end))
        end = start
    return windows


def run_backfill(
    *,
    api_key: str,
    origin: str = ORIGIN_DEFAULT,
    now: str | datetime | pd.Timestamp | None = None,
    lookback_h: float = DEFAULT_RECENT_CACHE_RETENTION_H,
    chunk_h: float = 12.0,
    sleep_s: float = 1.5,
    retention_h: float = DEFAULT_RECENT_CACHE_RETENTION_H,
    cache_path: Path = DEFAULT_RECENT_CACHE,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Backfill the recent-departures cache with one FIDS call per chunk."""
    now_ts = _utc_timestamp(now)
    windows = iter_backfill_windows(now_ts, lookback_h=lookback_h,
                                    chunk_h=chunk_h)
    logger.info("backfilling %d windows from %s back %.1fh "
                "(chunk=%.2fh, sleep=%.1fs)",
                len(windows), now_ts, lookback_h, chunk_h, sleep_s)

    cache = load_recent_departures_cache(cache_path)
    if dry_run:
        for i, (start, end) in enumerate(windows, start=1):
            logger.info("dry-run %03d/%03d: %s -> %s", i, len(windows),
                        start, end)
        return cache

    for i, (start, end) in enumerate(windows, start=1):
        chunk_hours = (end - start).total_seconds() / 3600.0
        try:
            fresh = fetch_recent_departures(
                api_key, origin=origin, lookback_h=chunk_hours,
                now=end.to_pydatetime())
        except Exception:
            logger.exception("failed chunk %d/%d: %s -> %s",
                             i, len(windows), start, end)
            if not continue_on_error:
                raise
            fresh = pd.DataFrame()

        cache = merge_recent_departures_cache(
            cache, fresh, now=now_ts.to_pydatetime(), retention_h=retention_h)
        write_recent_departures_cache(cache, cache_path)
        logger.info("chunk %03d/%03d: %s -> %s, fresh=%d, cache=%d",
                    i, len(windows), start, end, len(fresh), len(cache))

        if i < len(windows) and sleep_s > 0:
            time.sleep(sleep_s)

    return cache


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--origin", default=ORIGIN_DEFAULT)
    p.add_argument("--lookback-h", type=float, default=168.0,
                   help="Total history to backfill.")
    p.add_argument("--chunk-h", type=float, default=12.0,
                   help="Hours per FIDS call. Keep at or below the 12 h FIDS "
                        "window cap so each chunk is a single request.")
    p.add_argument("--sleep-s", type=float, default=1.5,
                   help="Pause between calls; stays under the gateway's "
                        "~1 request/second limit.")
    p.add_argument("--retention-h", type=float,
                   default=DEFAULT_RECENT_CACHE_RETENTION_H)
    p.add_argument("--cache", type=Path, default=DEFAULT_RECENT_CACHE)
    p.add_argument("--api-key", default=None)
    p.add_argument("--now", default=None,
                   help="UTC timestamp anchor, mainly for testing/resume.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue past failed chunks. Default stops so cache "
                        "coverage stays contiguous from now backward.")
    p.add_argument("--dry-run", action="store_true",
                   help="Log planned calls without using AeroDataBox.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    key = resolve_api_key(args.api_key)
    if not key and not args.dry_run:
        raise SystemExit(
            "No API key. Pass --api-key, create aerodatabox.key, or set "
            "AERODATABOX_KEY.")

    cache = run_backfill(
        api_key=key or "",
        origin=args.origin,
        now=args.now,
        lookback_h=args.lookback_h,
        chunk_h=args.chunk_h,
        sleep_s=args.sleep_s,
        retention_h=args.retention_h,
        cache_path=args.cache,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    logger.info("done; cache rows=%d path=%s", len(cache), args.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
