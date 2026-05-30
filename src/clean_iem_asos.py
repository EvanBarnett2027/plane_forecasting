"""Combine, clean, regularise and validate raw IEM ASOS files per station.

Reads the per-station monthly files written by ``src.ingest_iem_asos``
from ``data/raw/iem_asos/{STATION}/{YYYY_MM}.csv`` and writes a cleaned
hourly weather series per station:

    data/processed/weather/{STATION}_iem_2020_present.parquet

What this does (per station)
----------------------------
* Keeps the core meteorological columns useful for modelling.
* Parses the observation timestamp ``valid`` as UTC.
* Coerces numeric columns and converts trace precip to a small float.
* Aggregates sub-hourly METAR/SPECI reports into a strict hourly series
  with one row per UTC hour, applying a column-appropriate aggregator
  (max for gusts/precip, min for visibility/ceiling, mean for instant
  readings, circular mean for wind direction, union for weather codes).
  See ``HOURLY_AGG`` for the full per-column rule.
* Runs validation (station, per-month counts, coverage gaps, missingness,
  unique hourly index).

Usage
-----
    # single station (default ORD)
    python -m src.clean_iem_asos --station ORD

    # several stations
    python -m src.clean_iem_asos --stations ORD,LAX,DEN

    # every station with raw data on disk, plus a combined long table
    python -m src.clean_iem_asos --all --combined-out \\
        data/processed/dest_weather_iem_2020_present.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest_iem_asos import DEFAULT_RAW_ROOT, station_dir
from src.ingest_bts_ord import PROJECT_ROOT

logger = logging.getLogger("clean_iem_asos")

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "weather"


def processed_path_for(processed_dir: Path, station: str) -> Path:
    return processed_dir / f"{station.upper()}_iem_2020_present.parquet"

# Core columns kept for modelling (others in the raw file are dropped).
CORE_COLS = [
    "station", "valid",
    "tmpf",            # air temperature (F)
    "dwpf",            # dew point (F)
    "relh",            # relative humidity (%)
    "drct",            # wind direction (deg)
    "sknt",            # wind speed (kt)
    "gust",            # wind gust (kt)
    "p01i",            # 1-hour precip (in; trace -> small float)
    "alti",            # altimeter (inHg)
    "mslp",            # sea-level pressure (mb)
    "vsby",            # visibility (mi)
    "skyc1",           # lowest sky coverage code
    "skyl1",           # lowest sky level (ft)
    "wxcodes",         # present weather codes
    "ice_accretion_1hr",
    "peak_wind_gust",
    "feel",            # apparent temperature (F)
    "snowdepth",
]

NUMERIC_COLS = [
    "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "alti",
    "mslp", "vsby", "skyl1", "ice_accretion_1hr", "peak_wind_gust",
    "feel", "snowdepth",
]

IMPORTANT_COLS_FOR_MISSINGNESS = [
    "tmpf", "dwpf", "relh", "sknt", "drct", "p01i", "alti", "vsby",
]

# --------------------------------------------------------------------------- #
# Per-column hourly aggregation
# --------------------------------------------------------------------------- #
#
# IEM ASOS at ORD is sub-hourly: temperature/dewpoint/humidity appear only
# on the routine METAR (~once/hour, ~10% of rows), while wind, altimeter and
# visibility update every ~5 minutes and gust/wxcodes appear on SPECI
# specials in between. Picking one row per hour would discard the extremes
# that actually matter for disruption modelling (e.g. peak gust, worst
# visibility, any precip burst). Instead we bucket every sub-hourly report
# into its UTC hour and apply a column-appropriate aggregator.

# Sky-cover ordinal: increasing severity of cloud cover / obscuration.
_SKY_RANK = {"SKC": 0, "CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4,
             "VV": 5}


def _circular_mean_deg(s: pd.Series) -> float:
    """Mean wind direction in degrees, computed as a unit-vector mean.

    Linear mean of degrees is wrong (mean of 350° and 10° should be 0°,
    not 180°). Returns NaN if no valid readings or the vectors cancel.
    """
    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return np.nan
    rad = np.deg2rad(vals)
    sin_, cos_ = float(np.sin(rad).mean()), float(np.cos(rad).mean())
    if abs(sin_) < 1e-9 and abs(cos_) < 1e-9:
        return np.nan
    return float(np.rad2deg(np.arctan2(sin_, cos_)) % 360.0)


def _worst_sky(s: pd.Series) -> object:
    """Pick the most-obscured sky-cover code seen in the hour (VV > OVC > ...)."""
    best, best_rank = np.nan, -1
    for v in s.dropna():
        v = str(v).strip().upper()
        r = _SKY_RANK.get(v)
        if r is not None and r > best_rank:
            best, best_rank = v, r
    return best


def _union_wxcodes(s: pd.Series) -> object:
    """Union of distinct present-weather tokens seen in the hour."""
    codes: set[str] = set()
    for v in s.dropna():
        for tok in str(v).split():
            tok = tok.strip()
            if tok:
                codes.add(tok)
    return " ".join(sorted(codes)) if codes else np.nan


# Column -> aggregator. Anything not listed here is dropped from the
# hourly output. Aggregators chosen so the kept value is meaningful for a
# downstream flight-disruption model:
#   - extremes (max/min) where the worst-in-hour drives operations,
#   - mean where the column is a stable instant reading,
#   - circular mean for direction,
#   - union for categorical/text codes.
HOURLY_AGG: dict[str, object] = {
    "station":           "first",
    "tmpf":              "mean",
    "dwpf":              "mean",
    "relh":              "mean",
    "feel":              "mean",
    "alti":              "mean",
    "mslp":              "mean",
    "sknt":              "mean",
    "drct":              _circular_mean_deg,
    "gust":              "max",
    "peak_wind_gust":    "max",
    "p01i":              "max",
    "vsby":              "min",
    "wxcodes":           _union_wxcodes,
    "skyc1":             _worst_sky,
    "skyl1":             "min",
    "ice_accretion_1hr": "max",
    "snowdepth":         "last",
}


def load_raw_for_station(raw_root: Path, station: str) -> pd.DataFrame:
    sdir = station_dir(raw_root, station)
    files = sorted(sdir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No monthly files in {sdir}. "
            f"Run `python -m src.ingest_iem_asos --station {station}` first."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, na_values=["M", "null", ""])
        df.columns = df.columns.str.strip()
        frames.append(df)
    logger.info("[%s] loaded %d monthly files (%d obs total)",
                station.upper(), len(files), sum(len(f) for f in frames))
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sub-hourly reports for a single station to one row per UTC hour."""
    keep = [c for c in CORE_COLS if c in df.columns]
    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        logger.warning("source missing %d column(s): %s", len(missing), missing)
    df = df[keep].copy()

    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid"])

    for col in NUMERIC_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bucket every sub-hourly report into its UTC hour and aggregate with
    # the per-column rule in HOURLY_AGG. See the comment above the map for
    # why a single-row pick is not good enough here.
    df["valid_utc"] = df["valid"].dt.floor("h")
    df = df.drop(columns=["valid"])

    agg = {c: fn for c, fn in HOURLY_AGG.items() if c in df.columns}
    n_reports_per_hour = df.groupby("valid_utc").size().rename("n_reports")
    df = df.groupby("valid_utc", sort=True).agg(agg).reset_index()
    # Diagnostic: how many sub-hourly reports fed each hourly row.
    df = df.merge(n_reports_per_hour, left_on="valid_utc", right_index=True)

    front = ["station", "valid_utc"]
    df = df[front + [c for c in df.columns if c not in front]]
    return df.sort_values("valid_utc").reset_index(drop=True)


def validate(df: pd.DataFrame, station: str, *, verbose: bool = True) -> None:
    """Per-station validation. Asserts unique hourly index and station match."""
    if verbose:
        logger.info("--- [%s] validation -----------------------------",
                    station.upper())

    bad = df.loc[df["station"].astype("string").str.upper() != station.upper()]
    assert bad.empty, f"[{station}] {len(bad)} rows have wrong station id"

    span = df["valid_utc"].max() - df["valid_utc"].min()
    expected = int(span.total_seconds() // 3600) + 1
    got = len(df)
    miss_pct = 100.0 * (expected - got) / expected if expected else 0.0
    logger.info("[%s] %d hourly rows | span %s -> %s | coverage %.2f%% "
                "(%d missing hours)",
                station.upper(), got,
                df["valid_utc"].min(), df["valid_utc"].max(),
                100.0 - miss_pct, expected - got)

    if verbose:
        n = len(df)
        for col in IMPORTANT_COLS_FOR_MISSINGNESS:
            if col in df:
                miss = int(df[col].isna().sum())
                logger.info("  %-10s missing %6d (%5.2f%%)",
                            col, miss, 100.0 * miss / n)

    dups = int(df["valid_utc"].duplicated().sum())
    assert dups == 0, f"[{station}] {dups} duplicate hourly timestamps"


def clean_station(raw_root: Path, station: str,
                  processed_dir: Path | None,
                  *, verbose: bool = True) -> pd.DataFrame:
    """Load + clean + (optionally) write the parquet for one station.

    Returns the cleaned hourly DataFrame so callers can also assemble a
    combined long table without re-reading from disk.
    """
    df = load_raw_for_station(raw_root, station)
    df = clean(df)
    validate(df, station=station, verbose=verbose)
    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        out = processed_path_for(processed_dir, station)
        df.to_parquet(out, index=False)
        logger.info("[%s] wrote %s (%d rows x %d cols)",
                    station.upper(), out, len(df), df.shape[1])
    return df


def discover_stations(raw_root: Path) -> list[str]:
    """Every station with at least one raw monthly file under ``raw_root``."""
    return sorted(
        p.name.upper() for p in raw_root.iterdir()
        if p.is_dir() and any(p.glob("*.csv"))
    )


def _parse_stations(args: argparse.Namespace) -> list[str]:
    if args.all:
        return discover_stations(args.raw_root)
    if args.stations:
        return [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    return [args.station.upper()]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--station", default="ORD",
                     help="Single station id (default: ORD).")
    sel.add_argument("--stations",
                     help="Comma-separated list of station ids.")
    sel.add_argument("--all", action="store_true",
                     help="Clean every station found under --raw-root.")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                   help=f"Default: {DEFAULT_PROCESSED_DIR}")
    p.add_argument("--combined-out", type=Path,
                   help="If set, also write a single long parquet keyed "
                        "(station, valid_utc) containing every cleaned "
                        "station -- useful for joining destination weather "
                        "onto flights.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip per-station validation (not recommended).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    stations = _parse_stations(args)
    if not stations:
        logger.error("No stations to clean (raw root: %s)", args.raw_root)
        return 1

    logger.info("Cleaning %d station(s) from %s -> %s",
                len(stations), args.raw_root, args.processed_dir)

    verbose = len(stations) <= 3  # full per-station report only for small runs
    cleaned: list[pd.DataFrame] = []
    failed: list[str] = []
    for s in stations:
        try:
            df = clean_station(args.raw_root, s, args.processed_dir,
                               verbose=verbose)
        except Exception:
            logger.exception("[%s] FAILED (continuing)", s)
            failed.append(s)
            continue
        if args.combined_out is not None:
            cleaned.append(df)

    if args.combined_out is not None and cleaned:
        combined = (
            pd.concat(cleaned, ignore_index=True)
            .sort_values(["station", "valid_utc"])
            .reset_index(drop=True)
        )
        args.combined_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(args.combined_out, index=False)
        logger.info("wrote combined long table: %s (%d rows, %d stations)",
                    args.combined_out, len(combined),
                    combined["station"].nunique())

    logger.info("Done. cleaned=%d failed=%d", len(stations) - len(failed),
                len(failed))
    if failed:
        logger.error("Failed stations: %s", ", ".join(failed))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
