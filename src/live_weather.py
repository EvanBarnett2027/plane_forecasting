"""Live weather-forecast collection for the real-time dashboard.

At inference the model needs a *forecast* of the same hourly weather
fields it trained on (the training pipeline simulated forecasts by adding
lead-time noise to historical observations; here we fetch real forecasts
and add no noise). This module pulls hourly forecasts from the US
National Weather Service and maps them into the exact column schema the
flight/weather join expects, so the output drops into
:func:`src.join_flights_weather.join` unchanged.

Provider
--------
**NWS / api.weather.gov** -- free, no API key, US airports only. Flow per
station:

    IATA -> (lat, lon)            via airportsdata
    GET /points/{lat},{lon}       -> gridpoint forecast URL
    GET /gridpoints/{wfo}/{x},{y} -> raw multi-variable time series
    expand ISO-8601 intervals     -> one row per UTC hour
    convert units + rename        -> model weather schema

Output schema (long, one row per (station, hour)) matches
``src.clean_iem_asos``::

    station, valid_utc, tmpf, dwpf, relh, feel, alti, mslp, sknt, drct,
    gust, peak_wind_gust, p01i, vsby, skyl1, ice_accretion_1hr,
    snowdepth, n_reports

Train/serve skew (documented, intentional)
------------------------------------------
NWS does not forecast every field the ASOS observations carried. These
come back **NaN** and the model handles NaN natively, but be aware they
were *present* at train time:

* ``vsby``      -- NWS gridpoints ``visibility`` is usually empty.
* ``mslp`` / ``alti`` -- no public pressure forecast.
* ``peak_wind_gust`` -- METAR-only; ``gust`` (forecast) is the proxy.
* ``snowdepth`` -- NWS forecasts snow*fall*, not depth on the ground.
* ``n_reports`` -- a count of sub-hourly METARs; meaningless for a
  forecast. It ranked high in importance, so its absence is the single
  biggest skew; consider retraining without it for the live model.

Usage
-----
    # forecasts for every airport in a flights file
    python -m src.live_weather --flights data/live/flights_next48h.parquet

    # ad-hoc stations
    python -m src.live_weather --stations ORD,LAX,DEN
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.ingest_bts_ord import PROJECT_ROOT
from src.live_flights import DEFAULT_LIVE_DIR, airport_info

logger = logging.getLogger("live_weather")

NWS_BASE = "https://api.weather.gov"
DEFAULT_OUT = DEFAULT_LIVE_DIR / "weather_forecast.parquet"

# NWS only forecasts US locations (airportsdata `country`): the 50 states,
# DC, Puerto Rico and the USVI all report as "US"; the Pacific territories
# use their own ISO codes. International destinations (AMS, LHR, ...) are
# outside coverage and are skipped -- their flights get NaN dest weather,
# which the model handles natively.
NWS_COVERED_COUNTRIES = {"US", "GU", "AS", "MP", "PR", "VI"}

# Contact string NWS asks every client to send (their docs require a
# self-identifying User-Agent; an email is the conventional form).
USER_AGENT = "de300-plane-forecasting (evan387264@gmail.com)"

# The full hourly weather schema the model/join expects.
WX_COLS = [
    "tmpf", "dwpf", "relh", "feel", "alti", "mslp", "sknt", "drct",
    "gust", "peak_wind_gust", "p01i", "vsby", "skyl1",
    "ice_accretion_1hr", "snowdepth", "n_reports",
]

_ISO_DUR = re.compile(
    r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", re.IGNORECASE)


def _duration_hours(dur: str) -> int:
    """ISO-8601 duration (the NWS subset: P#DT#H#M) -> whole hours, >= 1."""
    m = _ISO_DUR.fullmatch(dur)
    if not m:
        return 1
    days, hours, minutes = (int(g) if g else 0 for g in m.groups())
    total = days * 24 + hours + (1 if minutes else 0)
    return max(total, 1)


def _expand_series(values: list[dict], conv=lambda v: v) -> pd.Series:
    """Expand NWS ``{validTime: "<start>/<dur>", value: x}`` into hourly.

    Each interval covers ``dur`` hours; every hour in it gets that value.
    Returns a Series indexed by tz-aware UTC hour timestamps.
    """
    idx, vals = [], []
    for item in values:
        vt = item.get("validTime")
        val = item.get("value")
        if vt is None or val is None:
            continue
        start_s, _, dur_s = vt.partition("/")
        start = pd.Timestamp(start_s).tz_convert("UTC").floor("h")
        for h in range(_duration_hours(dur_s)):
            idx.append(start + pd.Timedelta(hours=h))
            vals.append(conv(val))
    if not idx:
        return pd.Series(dtype="float64")
    s = pd.Series(vals, index=idx, dtype="float64")
    # collapse any overlap (later interval wins)
    return s[~s.index.duplicated(keep="last")].sort_index()


# Unit conversions to the model's schema.
def _c_to_f(c):       return c * 9.0 / 5.0 + 32.0
def _kmh_to_kt(k):    return k / 1.852
def _mm_to_in(mm):    return mm / 25.4


def gridpoints_to_hourly(props: dict) -> pd.DataFrame:
    """Map a raw NWS gridpoints ``properties`` dict to the hourly wx schema.

    Pure function (no network) so it is unit-testable with a captured or
    synthetic payload.
    """
    def grab(key, conv=lambda v: v):
        block = props.get(key) or {}
        return _expand_series(block.get("values", []), conv)

    cols = {
        "tmpf": grab("temperature", _c_to_f),
        "dwpf": grab("dewpoint", _c_to_f),
        "relh": grab("relativeHumidity"),
        "feel": grab("apparentTemperature", _c_to_f),
        "sknt": grab("windSpeed", _kmh_to_kt),
        "drct": grab("windDirection"),
        "gust": grab("windGust", _kmh_to_kt),
        "p01i": grab("quantitativePrecipitation", _mm_to_in),
        "ice_accretion_1hr": grab("iceAccumulation", _mm_to_in),
        # NWS-unavailable forecast fields -> NaN (see module docstring).
        # mslp, alti, peak_wind_gust, vsby, skyl1, snowdepth, n_reports
    }
    frame = pd.DataFrame(cols)
    for c in WX_COLS:
        if c not in frame.columns:
            frame[c] = np.nan
    frame = frame[WX_COLS]
    frame.index.name = "valid_utc"
    return frame.reset_index()


class NWSClient:
    """Thin NWS forecast client with retry/backoff."""

    def __init__(self, user_agent: str = USER_AGENT, timeout: int = 30,
                 retries: int = 4):
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept": "application/geo+json"})

    def _get(self, url: str, params: dict | None = None) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, params=params,
                                        timeout=(15, self.timeout))
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as err:
                status = getattr(getattr(err, "response", None),
                                 "status_code", None)
                # 4xx (except 429) is a permanent client error -- e.g. the
                # location is outside NWS coverage. Fail fast instead of
                # burning four backoff attempts on something that can't work.
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_err = err
                if attempt < self.retries:
                    backoff = min(60, 5 * 2 ** (attempt - 1))
                    logger.debug("  NWS attempt %d/%d failed (%s); retry in %ds",
                                 attempt, self.retries,
                                 type(err).__name__, backoff)
                    time.sleep(backoff)
        raise RuntimeError(f"giving up after {self.retries}: {url}") from last_err

    def gridpoint_url(self, lat: float, lon: float) -> str:
        data = self._get(f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}")
        return data["properties"]["forecastGridData"]

    def station_forecast(self, station: str) -> pd.DataFrame:
        """Hourly forecast for one airport, in the model wx schema."""
        empty = pd.DataFrame(columns=["station", "valid_utc", *WX_COLS])
        info = airport_info(station)
        country = info.get("country")
        if country not in NWS_COVERED_COUNTRIES:
            # International destination -- NWS can't forecast it. Skip
            # quietly (no retries); the flight gets NaN dest weather.
            logger.debug("%s: outside NWS coverage (country=%s); skipping",
                         station, country)
            return empty
        lat, lon = info.get("lat"), info.get("lon")
        if lat is None or lon is None:
            logger.warning("%s: no coordinates; skipping", station)
            return empty
        try:
            grid = self.gridpoint_url(lat, lon)
            props = self._get(grid)["properties"]
        except Exception as err:  # noqa: BLE001 - NWS is US-only / flaky abroad
            logger.warning("%s: forecast fetch failed (%s); skipping",
                           station, type(err).__name__)
            return empty
        hourly = gridpoints_to_hourly(props)
        hourly.insert(0, "station", station.upper())
        return hourly


def fetch_forecasts(stations: list[str],
                    client: NWSClient | None = None,
                    max_workers: int = 1) -> pd.DataFrame:
    """Hourly forecasts for several airports, concatenated long.

    ``max_workers > 1`` fetches stations concurrently (each NWS station is
    two independent GETs); useful when pulling weather for the ~150
    destinations of a full next-48 h schedule.
    """
    client = client or NWSClient()
    if max_workers and max_workers > 1 and len(stations) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(client.station_forecast, stations))
    else:
        results = [client.station_forecast(st) for st in stations]
    frames = [df for df in results if len(df)]
    logger.info("fetched forecasts for %d/%d stations",
                len(frames), len(stations))
    if not frames:
        return pd.DataFrame(columns=["station", "valid_utc", *WX_COLS])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["station", "valid_utc"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--flights", type=Path,
                   help="A flights parquet; weather is fetched for ORD + "
                        "every distinct dest.")
    g.add_argument("--stations", type=str,
                   help="Comma-separated IATA codes, e.g. ORD,LAX,DEN.")
    p.add_argument("--origin", default="ORD")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.flights:
        from src.live_flights import distinct_stations
        feats = pd.read_parquet(args.flights)
        stations = distinct_stations(feats, origin=args.origin)
    else:
        stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    logger.info("fetching forecasts for %d stations", len(stations))

    wx = fetch_forecasts(stations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(args.out, index=False)
    logger.info("wrote %d (station, hour) rows to %s", len(wx), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
