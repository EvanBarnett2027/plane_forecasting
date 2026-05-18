"""Combine, clean, label and validate the monthly raw BTS ORD files.

Reads every monthly file produced by ``src.ingest_bts_ord`` from
``data/raw/bts_ontime_ord/`` and writes a single cleaned, labelled,
validated dataset:

    data/processed/ord_departures_bts_2020_present.parquet

What this does
--------------
* Standardises all column names to ``snake_case``.
* Parses ``flight_date`` as datetime.
* Builds ``scheduled_dep_datetime`` / ``scheduled_arr_datetime`` from
  ``crs_dep_time`` / ``crs_arr_time`` in **America/Chicago** local time
  (ORD's timezone; see the note in ``add_scheduled_datetimes``).
* Preserves cancelled and diverted flights, and does NOT drop rows just
  because actual departure/arrival fields are missing (those are blank by
  definition for cancelled flights).
* Adds basic labels: ``disrupted``, ``delayed``, ``canceled``,
  ``diverted_flag``, ``weather_related``.
* Runs validation checks and prints a data-quality report.

Leakage note
------------
For predictive modeling, only the columns in ``SAFE_FEATURE_COLS`` are
known before the flight operates. Everything in ``POST_FLIGHT_OUTCOME_COLS``
(actual times, delays, taxi/air time, delay-cause minutes, cancellation
code) is realised *after* departure and must NOT be used as a feature -
they are kept only as labels/diagnostics. See README for the full list.

Usage
-----
    python -m src.clean_bts_ord
    python -m src.clean_bts_ord --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.ingest_bts_ord import DEFAULT_RAW_DIR, PROJECT_ROOT

logger = logging.getLogger("clean_bts_ord")

DEFAULT_PROCESSED = (
    PROJECT_ROOT / "data" / "processed"
    / "ord_departures_bts_2020_present.parquet"
)

ORIGIN_TZ = "America/Chicago"  # ORD local time

# --------------------------------------------------------------------------- #
# Column-name standardisation (explicit map = predictable & maintainable)
# --------------------------------------------------------------------------- #

SNAKE_CASE = {
    "Year": "year",
    "Quarter": "quarter",
    "Month": "month",
    "DayofMonth": "day_of_month",
    "DayOfWeek": "day_of_week",
    "FlightDate": "flight_date",
    "Reporting_Airline": "reporting_airline",
    "DOT_ID_Reporting_Airline": "dot_id_reporting_airline",
    "IATA_CODE_Reporting_Airline": "iata_code_reporting_airline",
    "Tail_Number": "tail_number",
    "Flight_Number_Reporting_Airline": "flight_number_reporting_airline",
    "Origin": "origin",
    "OriginCityName": "origin_city_name",
    "OriginState": "origin_state",
    "Dest": "dest",
    "DestCityName": "dest_city_name",
    "DestState": "dest_state",
    "CRSDepTime": "crs_dep_time",
    "DepTime": "dep_time",
    "DepDelay": "dep_delay",
    "DepDelayMinutes": "dep_delay_minutes",
    "DepDel15": "dep_del15",
    "DepartureDelayGroups": "departure_delay_groups",
    "TaxiOut": "taxi_out",
    "WheelsOff": "wheels_off",
    "WheelsOn": "wheels_on",
    "TaxiIn": "taxi_in",
    "CRSArrTime": "crs_arr_time",
    "ArrTime": "arr_time",
    "ArrDelay": "arr_delay",
    "ArrDelayMinutes": "arr_delay_minutes",
    "ArrDel15": "arr_del15",
    "ArrivalDelayGroups": "arrival_delay_groups",
    "Cancelled": "cancelled",
    "CancellationCode": "cancellation_code",
    "Diverted": "diverted",
    "CRSElapsedTime": "crs_elapsed_time",
    "ActualElapsedTime": "actual_elapsed_time",
    "AirTime": "air_time",
    "Distance": "distance",
    "CarrierDelay": "carrier_delay",
    "WeatherDelay": "weather_delay",
    "NASDelay": "nas_delay",
    "SecurityDelay": "security_delay",
    "LateAircraftDelay": "late_aircraft_delay",
}

# --------------------------------------------------------------------------- #
# Leakage classification (snake_case names)
# --------------------------------------------------------------------------- #

# Known before the flight operates -> safe to use as model features.
SAFE_FEATURE_COLS = [
    "year", "quarter", "month", "day_of_month", "day_of_week", "flight_date",
    "reporting_airline", "dot_id_reporting_airline",
    "iata_code_reporting_airline", "tail_number",
    "flight_number_reporting_airline",
    "origin", "origin_city_name", "origin_state",
    "dest", "dest_city_name", "dest_state",
    "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance",
    "scheduled_dep_datetime", "scheduled_arr_datetime",
]

# Realised only after the flight operates -> labels/diagnostics, NOT features.
POST_FLIGHT_OUTCOME_COLS = [
    "dep_time", "dep_delay", "dep_delay_minutes", "dep_del15",
    "departure_delay_groups", "taxi_out", "wheels_off", "wheels_on",
    "taxi_in", "arr_time", "arr_delay", "arr_delay_minutes", "arr_del15",
    "arrival_delay_groups", "cancelled", "cancellation_code", "diverted",
    "actual_elapsed_time", "air_time",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay",
    "late_aircraft_delay",
]

LABEL_COLS = ["disrupted", "delayed", "canceled", "diverted_flag",
              "weather_related"]

# Numeric indicator columns to coerce to nullable Int.
INDICATOR_COLS = ["cancelled", "diverted", "dep_del15", "arr_del15"]
CAUSE_DELAY_COLS = ["carrier_delay", "weather_delay", "nas_delay",
                    "security_delay", "late_aircraft_delay"]

# BTS CancellationCode: A=Carrier, B=Weather, C=National Air System, D=Security
WEATHER_CANCELLATION_CODE = "B"

DEDUPE_KEY = ["flight_date", "reporting_airline",
              "flight_number_reporting_airline", "origin", "dest",
              "crs_dep_time"]

IMPORTANT_COLS_FOR_MISSINGNESS = [
    "flight_date", "scheduled_dep_datetime", "scheduled_arr_datetime",
    "reporting_airline", "origin", "dest", "crs_dep_time", "crs_arr_time",
    "dep_delay", "dep_del15", "arr_del15", "cancelled", "diverted",
    "distance",
]

# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #


def load_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted([*raw_dir.glob("*.parquet"), *raw_dir.glob("*.csv")])
    if not files:
        raise FileNotFoundError(
            f"No monthly files in {raw_dir}. "
            f"Run `python -m src.ingest_bts_ord` first."
        )
    frames = []
    for f in files:
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        logger.info("loaded %s (%d rows)", f.name, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=SNAKE_CASE)


def _hhmm_to_local_datetime(flight_date: pd.Series, hhmm: pd.Series,
                            tz: str) -> pd.Series:
    """Combine a date with an HHMM clock value into a tz-aware datetime.

    BTS stores scheduled times as integers (e.g. 830 -> 08:30, 1830 ->
    18:30, 2400 -> midnight). 2400 is handled naturally: hour 24 + 0 min
    rolls the timestamp to 00:00 of the following day.
    """
    t = pd.to_numeric(hhmm, errors="coerce")
    hours = (t // 100)
    minutes = (t % 100)
    base = pd.to_datetime(flight_date, errors="coerce")
    naive = (
        base
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    naive = naive.where(t.notna() & base.notna())
    # ambiguous/nonexistent: scheduled clock times can't be DST-resolved, so
    # mark the once-a-year ambiguous hour NaT and shift the spring-forward gap.
    return naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")


def add_scheduled_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add scheduled departure/arrival datetimes in America/Chicago.

    Note: ORD is the origin for every row, so ``scheduled_dep_datetime`` is
    genuinely ORD-local. ``crs_arr_time`` is destination-local in the BTS
    source; per the project requirement it is expressed here in
    America/Chicago as well, which is correct only for same-tz destinations.
    Treat ``scheduled_arr_datetime`` as a Chicago-frame reference, not the
    literal wall clock at the destination.
    """
    df["scheduled_dep_datetime"] = _hhmm_to_local_datetime(
        df["flight_date"], df["crs_dep_time"], ORIGIN_TZ)
    df["scheduled_arr_datetime"] = _hhmm_to_local_datetime(
        df["flight_date"], df["crs_arr_time"], ORIGIN_TZ)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")

    for col in INDICATOR_COLS:
        if col in df:
            df[col] = df[col].astype("Float64").astype("Int64")

    # Cause-delay minutes are blank when not applicable -> 0 (post-flight).
    for col in CAUSE_DELAY_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = add_scheduled_datetimes(df)

    # Drop only exact full-row duplicates (BTS occasionally re-emits a row);
    # rows are NOT dropped for missing actual dep/arr fields.
    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        logger.info("dropped %d exact-duplicate rows", before - len(df))

    df = add_labels(df)

    ordered = (
        [c for c in SAFE_FEATURE_COLS if c in df.columns]
        + [c for c in POST_FLIGHT_OUTCOME_COLS if c in df.columns]
        + [c for c in LABEL_COLS if c in df.columns]
    )
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    sort_keys = [k for k in ("flight_date", "crs_dep_time") if k in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic disruption labels (all stored as nullable Int 0/1)."""
    dep15 = df["dep_del15"].fillna(0).astype("int64")
    arr15 = df["arr_del15"].fillna(0).astype("int64")
    cancelled = df["cancelled"].fillna(0).astype("int64")
    diverted = df["diverted"].fillna(0).astype("int64")

    weather_minutes = pd.to_numeric(
        df.get("weather_delay", 0), errors="coerce").fillna(0)
    weather_cancel = (
        df.get("cancellation_code", pd.Series(index=df.index, dtype="object"))
        .astype("string").str.upper().str.strip()
        == WEATHER_CANCELLATION_CODE
    )

    df["delayed"] = ((dep15 == 1) | (arr15 == 1)).astype("Int64")
    df["disrupted"] = (
        (dep15 == 1) | (arr15 == 1) | (cancelled == 1) | (diverted == 1)
    ).astype("Int64")
    df["canceled"] = cancelled.astype("Int64")
    df["diverted_flag"] = diverted.astype("Int64")
    df["weather_related"] = (
        (weather_minutes > 0) | weather_cancel.fillna(False)
    ).astype("Int64")
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(df: pd.DataFrame, origin: str = "ORD") -> None:
    logger.info("=" * 64)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 64)

    # 1. All rows must be the requested origin.
    bad_origin = df.loc[df["origin"].astype("string").str.upper() != origin]
    assert bad_origin.empty, (
        f"{len(bad_origin)} rows have origin != {origin}"
    )
    logger.info("origin check: all %d rows have origin == %s", len(df), origin)

    # 2. Rows per year/month.
    by_ym = (
        df.assign(_ym=df["flight_date"].dt.strftime("%Y-%m"))
        .groupby("_ym").size()
    )
    logger.info("rows per year-month:")
    for ym, n in by_ym.items():
        logger.info("  %s : %7d", ym, n)
    logger.info("  TOTAL  : %7d", len(df))

    # 3. Rates.
    n = len(df)
    def rate(col):
        return 100.0 * df[col].fillna(0).astype("int64").sum() / n if n else 0.0
    logger.info("disruption rate : %6.2f%%", rate("disrupted"))
    logger.info("delay rate      : %6.2f%%", rate("delayed"))
    logger.info("cancellation rate: %5.2f%%", rate("canceled"))
    logger.info("diversion rate  : %6.2f%%", rate("diverted_flag"))
    logger.info("weather-related : %6.2f%%", rate("weather_related"))

    # 4. Missingness for important columns.
    logger.info("missingness (important columns):")
    for col in IMPORTANT_COLS_FOR_MISSINGNESS:
        if col in df:
            miss = df[col].isna().sum()
            logger.info("  %-24s %7d (%5.2f%%)", col, miss, 100.0 * miss / n)

    # 5. Duplicate flight records on the natural key.
    key = [k for k in DEDUPE_KEY if k in df.columns]
    dup_mask = df.duplicated(subset=key, keep=False)
    n_dups = int(dup_mask.sum())
    if n_dups:
        sample = (df.loc[dup_mask, key]
                  .sort_values(key).head(10).to_string(index=False))
        logger.error("found %d rows sharing key %s. Sample:\n%s",
                      n_dups, key, sample)
    assert n_dups == 0, (
        f"{n_dups} duplicate flight records on {key}. "
        f"Exact-duplicate rows are already dropped, so these are not "
        f"explainable as simple re-emits - inspect the source months."
    )
    logger.info("duplicate check: no duplicate flight records on %s", key)
    logger.info("=" * 64)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--origin", default="ORD",
                   help="Expected origin for the validation check.")
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
        validate(df, origin=args.origin.upper())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    span = ""
    if df["flight_date"].notna().any():
        span = (f" ({df['flight_date'].min().date()} -> "
                f"{df['flight_date'].max().date()})")
    logger.info("Wrote %d rows x %d cols to %s%s",
                len(df), df.shape[1], args.out, span)
    return 0


if __name__ == "__main__":
    sys.exit(main())
