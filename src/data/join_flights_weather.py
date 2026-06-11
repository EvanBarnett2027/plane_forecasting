"""Join ORD-departing flights with origin and destination weather.

Produces the final modelling table::

    data/processed/ord_flights_with_weather_2020_present.parquet

For every flight row this attaches:

* ``origin_wx_*`` - ORD weather at the **scheduled-departure** UTC hour.
* ``dest_wx_*``   - destination-airport weather at the **scheduled-arrival**
  UTC hour.

Correct destination-arrival timestamp
-------------------------------------
The flight cleaner builds ``scheduled_arr_datetime`` by combining
``flight_date`` with ``crs_arr_time`` and labelling it America/Chicago,
but BTS's ``crs_arr_time`` is the **destination-local** wall clock. For
non-CT destinations the UTC instant derived from that column is off by
1-3 hours, which would mis-align the destination-weather join.

The cleanest, timezone-independent fix uses ``crs_elapsed_time``
(scheduled gate-to-gate minutes, 100%% populated in the BTS data):

    scheduled_dep_utc = scheduled_dep_datetime.tz_convert("UTC")
    scheduled_arr_utc = scheduled_dep_utc + crs_elapsed_time minutes

Both joins use the UTC instant floored to the hour, which is exactly what
the weather tables are already keyed on.

Usage
-----
    python -m src.join_flights_weather
    python -m src.join_flights_weather --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.ingest_bts_ord import PROJECT_ROOT

logger = logging.getLogger("join_flights_weather")

DEFAULT_FLIGHTS = (
    PROJECT_ROOT / "data" / "processed"
    / "ord_departures_bts_2020_present.parquet"
)
DEFAULT_ORIGIN_WEATHER = (
    PROJECT_ROOT / "data" / "processed" / "weather"
    / "ORD_iem_2020_present.parquet"
)
DEFAULT_DEST_WEATHER = (
    PROJECT_ROOT / "data" / "processed"
    / "dest_weather_iem_2020_present.parquet"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "data" / "processed"
    / "ord_flights_with_weather_2020_present.parquet"
)

# Drop these from the weather table before joining (already on the flight
# row or not useful as a feature).
_DROP_FROM_WEATHER = ["station"]


def add_utc_join_keys(flights: pd.DataFrame) -> pd.DataFrame:
    """Add ``scheduled_dep_utc_hour`` and ``scheduled_arr_utc_hour``.

    These are the join keys against the hourly weather tables. Built from
    ``scheduled_dep_datetime`` + ``crs_elapsed_time`` so the destination
    timestamp is correct regardless of the destination's timezone.
    """
    if flights["scheduled_dep_datetime"].dt.tz is None:
        raise ValueError("scheduled_dep_datetime must be tz-aware")

    dep_utc = flights["scheduled_dep_datetime"].dt.tz_convert("UTC")
    elapsed = pd.to_numeric(flights["crs_elapsed_time"], errors="coerce")
    arr_utc = dep_utc + pd.to_timedelta(elapsed, unit="m")

    flights = flights.copy()
    flights["scheduled_dep_utc_hour"] = dep_utc.dt.floor("h")
    flights["scheduled_arr_utc_hour"] = arr_utc.dt.floor("h")
    return flights


def _prepare_origin(wx: pd.DataFrame, prefix: str) -> pd.DataFrame:
    wx = wx.drop(columns=[c for c in _DROP_FROM_WEATHER if c in wx.columns])
    wx = wx.rename(columns={"valid_utc": "_join_hour"})
    rename = {c: f"{prefix}{c}" for c in wx.columns if c != "_join_hour"}
    return wx.rename(columns=rename)


def _prepare_dest(wx: pd.DataFrame, prefix: str) -> pd.DataFrame:
    wx = wx.rename(columns={"valid_utc": "_join_hour",
                            "station": "_join_station"})
    rename = {c: f"{prefix}{c}" for c in wx.columns
              if c not in ("_join_hour", "_join_station")}
    return wx.rename(columns=rename)


def join(flights: pd.DataFrame,
         origin_wx: pd.DataFrame,
         dest_wx: pd.DataFrame,
         *, origin_prefix: str = "origin_wx_",
         dest_prefix: str = "dest_wx_") -> pd.DataFrame:
    flights = add_utc_join_keys(flights)

    o = _prepare_origin(origin_wx, origin_prefix)
    flights = flights.merge(
        o, how="left",
        left_on="scheduled_dep_utc_hour", right_on="_join_hour",
    ).drop(columns=["_join_hour"])

    d = _prepare_dest(dest_wx, dest_prefix)
    flights = flights.merge(
        d, how="left",
        left_on=["dest", "scheduled_arr_utc_hour"],
        right_on=["_join_station", "_join_hour"],
    ).drop(columns=["_join_hour", "_join_station"])

    return flights


def report(flights: pd.DataFrame, origin_prefix: str,
           dest_prefix: str) -> None:
    n = len(flights)
    logger.info("=" * 64)
    logger.info("JOIN REPORT")
    logger.info("=" * 64)
    logger.info("rows: %d", n)

    o_tmpf = f"{origin_prefix}tmpf"
    d_tmpf = f"{dest_prefix}tmpf"
    if o_tmpf in flights:
        miss = int(flights[o_tmpf].isna().sum())
        logger.info("origin weather attached: %d (%.2f%% missing)",
                    n - miss, 100.0 * miss / n)
    if d_tmpf in flights:
        miss = int(flights[d_tmpf].isna().sum())
        logger.info("dest weather attached:   %d (%.2f%% missing)",
                    n - miss, 100.0 * miss / n)
        # Which dests are missing weather?
        miss_by_dest = (
            flights.loc[flights[d_tmpf].isna(), "dest"]
            .value_counts().head(10)
        )
        if len(miss_by_dest):
            logger.info("top dests missing weather:")
            for d, c in miss_by_dest.items():
                logger.info("  %s : %d", d, c)
    logger.info("=" * 64)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--flights", type=Path, default=DEFAULT_FLIGHTS)
    p.add_argument("--origin-weather", type=Path,
                   default=DEFAULT_ORIGIN_WEATHER)
    p.add_argument("--dest-weather", type=Path,
                   default=DEFAULT_DEST_WEATHER)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--origin-prefix", default="origin_wx_")
    p.add_argument("--dest-prefix", default="dest_wx_")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("loading flights: %s", args.flights)
    flights = pd.read_parquet(args.flights)
    logger.info("  %d rows x %d cols", len(flights), flights.shape[1])

    logger.info("loading origin weather: %s", args.origin_weather)
    origin_wx = pd.read_parquet(args.origin_weather)
    logger.info("  %d hourly rows", len(origin_wx))

    logger.info("loading dest weather:   %s", args.dest_weather)
    dest_wx = pd.read_parquet(args.dest_weather)
    logger.info("  %d hourly rows across %d stations",
                len(dest_wx), dest_wx["station"].nunique())

    joined = join(flights, origin_wx, dest_wx,
                  origin_prefix=args.origin_prefix,
                  dest_prefix=args.dest_prefix)
    report(joined, args.origin_prefix, args.dest_prefix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.out, index=False)
    logger.info("Wrote %d rows x %d cols to %s",
                len(joined), joined.shape[1], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
