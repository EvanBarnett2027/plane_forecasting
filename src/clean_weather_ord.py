"""Combine, clean, regularise and validate the raw ORD ASOS files.

Reads every monthly file produced by ``src.ingest_weather_ord`` from
``data/raw/iem_asos_ord/`` and writes one cleaned hourly weather series:

    data/processed/ord_weather_iem_2020_present.parquet

What this does
--------------
* Keeps the core meteorological columns useful for modelling.
* Parses the observation timestamp ``valid`` as UTC and adds
  ``valid_local`` in **America/Chicago** so it lines up with the flight
  dataset's ``scheduled_dep_datetime``.
* Coerces numeric columns and converts trace precip to a small float.
* Regularises sub-hourly METAR/SPECI reports to a strict hourly series:
  one row per UTC hour, preferring the report that actually has a
  temperature reading.
* Runs validation (station, per-month counts, coverage gaps, missingness,
  unique hourly index).

Usage
-----
    python -m src.clean_weather_ord
    python -m src.clean_weather_ord --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.ingest_weather_ord import DEFAULT_RAW_DIR
from src.ingest_bts_ord import PROJECT_ROOT

logger = logging.getLogger("clean_weather_ord")

DEFAULT_PROCESSED = (
    PROJECT_ROOT / "data" / "processed"
    / "ord_weather_iem_2020_present.parquet"
)

LOCAL_TZ = "America/Chicago"

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


def load_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No monthly files in {raw_dir}. "
            f"Run `python -m src.ingest_weather_ord` first."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, na_values=["M", "null", ""])
        df.columns = df.columns.str.strip()
        logger.info("loaded %s (%d obs)", f.name, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
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

    # Regularise to a strict hourly series. Sub-hourly SPECI reports often
    # carry wind/altimeter but no temperature, so within each UTC hour keep
    # the row that has a temperature if one exists, else the first report.
    df["valid_hour_utc"] = df["valid"].dt.floor("h")
    df["_has_temp"] = df["tmpf"].notna()
    df = (
        df.sort_values(["valid_hour_utc", "_has_temp", "valid"],
                        ascending=[True, False, True])
        .drop_duplicates(subset=["valid_hour_utc"], keep="first")
        .drop(columns=["_has_temp", "valid"])
        .rename(columns={"valid_hour_utc": "valid_utc"})
        .reset_index(drop=True)
    )

    # Local time for joining with the flight dataset (America/Chicago).
    df["valid_local"] = df["valid_utc"].dt.tz_convert(LOCAL_TZ)

    front = ["station", "valid_utc", "valid_local"]
    df = df[front + [c for c in df.columns if c not in front]]
    return df.sort_values("valid_utc").reset_index(drop=True)


def validate(df: pd.DataFrame, station: str = "ORD") -> None:
    logger.info("=" * 64)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 64)

    bad = df.loc[df["station"].astype("string").str.upper() != station]
    assert bad.empty, f"{len(bad)} rows have station != {station}"
    logger.info("station check: all %d rows have station == %s",
                len(df), station)

    by_ym = (
        df.assign(_ym=df["valid_utc"].dt.strftime("%Y-%m"))
        .groupby("_ym").size()
    )
    logger.info("hourly obs per year-month:")
    for ym, n in by_ym.items():
        logger.info("  %s : %5d", ym, n)
    logger.info("  TOTAL  : %6d", len(df))

    # Coverage vs the number of hours actually spanned.
    span = df["valid_utc"].max() - df["valid_utc"].min()
    expected = int(span.total_seconds() // 3600) + 1
    got = len(df)
    logger.info("span: %s -> %s (UTC)",
                df["valid_utc"].min(), df["valid_utc"].max())
    logger.info("coverage: %d / %d hours (%.2f%%), %d missing hours",
                got, expected, 100.0 * got / expected, expected - got)

    n = len(df)
    logger.info("missingness (important columns):")
    for col in IMPORTANT_COLS_FOR_MISSINGNESS:
        if col in df:
            miss = int(df[col].isna().sum())
            logger.info("  %-10s %6d (%5.2f%%)", col, miss, 100.0 * miss / n)

    dups = int(df["valid_utc"].duplicated().sum())
    assert dups == 0, f"{dups} duplicate hourly timestamps after regularising"
    logger.info("hourly index check: %d unique hourly timestamps", len(df))
    logger.info("=" * 64)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--station", default="ORD",
                   help="Expected station id for the validation check.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the validation report (not recommended).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    df = load_raw(args.raw_dir)
    df = clean(df)

    if not args.no_validate:
        validate(df, station=args.station.upper())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("Wrote %d rows x %d cols to %s (%s -> %s)",
                len(df), df.shape[1], args.out,
                df["valid_utc"].min(), df["valid_utc"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
