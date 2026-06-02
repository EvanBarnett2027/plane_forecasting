"""Live flight collection for the real-time disruption dashboard.

Two jobs, both for ORD departures:

1. **Forward schedule** -- every flight scheduled to depart in the next
   48 h. This is the population the model scores each refresh.
2. **Recent completed** -- flights that already departed/landed in the
   last N hours, with their realised disruption status. This feeds the
   ``trailing_disruption_rate_*`` operational features
   (see :mod:`src.trailing_state`).

Design
------
The API is abstracted behind a small *normalized schema* so the rest of
the pipeline never sees provider-specific JSON. An adapter's only job is
to turn a provider response into a DataFrame with these columns::

    ident                flight identifier (optional, diagnostics)
    reporting_airline    IATA carrier code, e.g. "UA"  (matches BTS)
    dest                 destination IATA, e.g. "LAX"
    scheduled_dep_utc    tz-aware UTC timestamp (gate-out)
    scheduled_arr_utc    tz-aware UTC timestamp (gate-in)
    cancelled            0/1            (completed flights only)
    diverted             0/1            (completed flights only)
    dep_del15            0/1 or NaN     (completed flights only)
    arr_del15            0/1 or NaN     (completed flights only)

:func:`build_flight_features` then derives the exact model feature columns
(crs_dep_time, crs_arr_time, crs_elapsed_time, distance, calendar parts,
dest_state, and the ``scheduled_*_utc_hour`` join keys) from that schema,
using :mod:`airportsdata` for per-airport timezone / state / coordinates.

The reference adapter targets **AeroDataBox** (via the API.Market gateway).
AeroDataBox's FIDS "airport departures/arrivals by local time range"
endpoint returns *all* departures in a time window in a single request --
unlike FlightAware AeroAPI's 15-record result sets, which rate-limit ORD's
busy schedule into minutes of paging. The window is capped at 12 h by the
API, so the next-48 h schedule is fetched as four 12 h chunks (and a
``lookback_h`` history as ``ceil(lookback_h / 12)`` chunks). To swap
providers, write a new ``*_to_normalized`` function returning the schema
above; the feature builder is unchanged.

Usage
-----
    # forward schedule for the next 48 h -> data/live/flights_next48h.parquet
    AERODATABOX_KEY=... python -m src.live_flights schedule

    # recent completed flights (last 168 h) -> data/live/flights_recent.parquet
    AERODATABOX_KEY=... python -m src.live_flights recent --lookback-h 168

    # if your key is from RapidAPI (not API.Market), select that gateway:
    AERODATABOX_KEY=... python -m src.live_flights schedule --provider rapidapi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from src.ingest_bts_ord import PROJECT_ROOT

logger = logging.getLogger("live_flights")

DEFAULT_LIVE_DIR = PROJECT_ROOT / "data" / "live"
ORIGIN_DEFAULT = "ORD"
WINDOW_H = 48
EARTH_RADIUS_MI = 3958.7613  # mean Earth radius in statute miles

# --------------------------------------------------------------------------- #
# AeroDataBox gateway configuration
# --------------------------------------------------------------------------- #
# AeroDataBox is resold through several marketplaces that proxy the *same* API
# (identical paths, params, and response schema) but differ in base URL and
# auth headers. Pick one with AERODATABOX_PROVIDER (default: apimarket);
# override the base URL with AERODATABOX_BASE. Both verified against the
# respective OpenAPI specs.
_GATEWAYS: dict[str, dict] = {
    "apimarket": {
        "base": "https://prod.api.market/api/v1/aedbx/aerodatabox",
        "key_header": "x-magicapi-key",
        "host": None,
    },
    "rapidapi": {
        "base": "https://aerodatabox.p.rapidapi.com",
        "key_header": "X-RapidAPI-Key",
        "host": "aerodatabox.p.rapidapi.com",   # required X-RapidAPI-Host
    },
}
# FIDS "by local time range" is capped at a 12 h window per request.
FIDS_MAX_WINDOW_H = 12
DELAY15_SECONDS = 15 * 60


def _provider() -> str:
    """Selected marketplace (read at call time so the env var can change)."""
    return os.environ.get("AERODATABOX_PROVIDER", "apimarket").strip().lower()


def _gateway() -> dict:
    name = _provider()
    gw = _GATEWAYS.get(name)
    if gw is None:
        raise AeroDataBoxError(
            f"unknown AERODATABOX_PROVIDER={name!r}; "
            f"choose one of {sorted(_GATEWAYS)}")
    return gw


def _base_url() -> str:
    return (os.environ.get("AERODATABOX_BASE") or _gateway()["base"]).rstrip("/")


def _auth_headers(api_key: str) -> dict:
    """Gateway-specific auth headers for the selected marketplace."""
    gw = _gateway()
    headers = {gw["key_header"]: api_key, "Accept": "application/json"}
    if gw["host"]:
        headers["X-RapidAPI-Host"] = gw["host"]
    return headers

# Raw-response disk cache (avoids re-billing identical FIDS calls within a
# short window -- e.g. the four schedule chunks of one dashboard refresh, or
# repeated CLI runs). The durable cost-saver for the 168 h history is the
# rolling normalized parquet cache below; this TTL cache is intentionally
# short so each genuine refresh still updates.
DEFAULT_CACHE_TTL_S = int(os.environ.get("AERODATABOX_CACHE_TTL_S", "600"))
DEFAULT_RESPONSE_CACHE_DIR = DEFAULT_LIVE_DIR / "cache" / "aerodatabox"

# Rolling recent-status cache (provider-agnostic; stores normalized rows).
DEFAULT_RECENT_CACHE = DEFAULT_LIVE_DIR / "flights_recent_cache.parquet"
DEFAULT_RECENT_CACHE_REFRESH_H = int(
    os.environ.get("AERODATABOX_RECENT_CACHE_REFRESH_H", "3"))
DEFAULT_RECENT_CACHE_RETENTION_H = int(
    os.environ.get("AERODATABOX_RECENT_CACHE_RETENTION_H", "168"))

NORMALIZED_COLS = [
    "ident", "reporting_airline", "dest",
    "scheduled_dep_utc", "scheduled_arr_utc",
    "cancelled", "diverted", "dep_del15", "arr_del15",
]

RECENT_CACHE_DEDUPE_COLS = [
    "ident", "reporting_airline", "dest", "scheduled_dep_utc",
]

# Full US subdivision name (airportsdata `subd`) -> 2-letter code, to match
# the BTS ``dest_state`` column the model was trained on.
US_STATE_ABBREV: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR", "Virgin Islands": "VI", "U.S. Virgin Islands": "VI",
    "Guam": "GU", "American Samoa": "AS",
    "Northern Mariana Islands": "MP",
}


# --------------------------------------------------------------------------- #
# Airport metadata (timezone / state / coordinates)
# --------------------------------------------------------------------------- #

_AIRPORTS_CACHE: dict | None = None


def _airports() -> dict:
    """Lazily load the IATA-keyed airport database (offline, bundled)."""
    global _AIRPORTS_CACHE
    if _AIRPORTS_CACHE is None:
        import airportsdata
        _AIRPORTS_CACHE = airportsdata.load("IATA")
    return _AIRPORTS_CACHE


def airport_info(iata: str) -> dict:
    return _airports().get(iata.upper(), {})


def haversine_miles(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles (matches BTS ``distance``)."""
    rlat1, rlat2 = np.radians(lat1), np.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2)
    return float(2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a)))


def state_abbrev(subd: str | None) -> str | None:
    if not subd:
        return None
    return US_STATE_ABBREV.get(subd, None)


# --------------------------------------------------------------------------- #
# Feature engineering (provider-agnostic; operates on the normalized schema)
# --------------------------------------------------------------------------- #


def _hhmm_local(utc_ts: pd.Series, tz: str | None) -> pd.Series:
    """tz-aware UTC -> local HHMM integer (e.g. 0830 -> 830, 1830 -> 1830)."""
    if tz is None:
        return pd.Series([pd.NA] * len(utc_ts), index=utc_ts.index, dtype="Int64")
    local = utc_ts.dt.tz_convert(tz)
    return (local.dt.hour * 100 + local.dt.minute).astype("Int64")


def build_flight_features(norm: pd.DataFrame, *,
                          origin: str = ORIGIN_DEFAULT) -> pd.DataFrame:
    """Derive the model's schedule/route feature columns + join keys.

    Input: a DataFrame in the :data:`NORMALIZED_COLS` schema. Output: one
    row per flight with the calendar/schedule/route features the model
    consumes, plus ``scheduled_dep_utc_hour`` / ``scheduled_arr_utc_hour``
    (the weather-join keys and lead-time anchors). Weather and operational
    features are attached downstream.
    """
    if not len(norm):
        return norm.assign(**{c: [] for c in (
            "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance",
            "month", "day_of_month", "day_of_week", "quarter",
            "dest_state", "scheduled_dep_utc_hour", "scheduled_arr_utc_hour")})

    df = norm.copy()
    dep_utc = pd.to_datetime(df["scheduled_dep_utc"], utc=True)
    arr_utc = pd.to_datetime(df["scheduled_arr_utc"], utc=True)

    origin_tz = airport_info(origin).get("tz", "America/Chicago")
    df["crs_dep_time"] = _hhmm_local(dep_utc, origin_tz)

    # crs_arr_time is destination-local wall clock in BTS; convert per-dest.
    dest_tz = df["dest"].map(lambda d: airport_info(d).get("tz"))
    arr_hhmm = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for tz, grp in arr_utc.groupby(dest_tz):
        if tz is None:
            continue
        local = grp.dt.tz_convert(tz)
        arr_hhmm.loc[grp.index] = (local.dt.hour * 100
                                   + local.dt.minute).astype("Int64")
    df["crs_arr_time"] = arr_hhmm

    df["crs_elapsed_time"] = ((arr_utc - dep_utc).dt.total_seconds()
                              / 60.0).round().astype("Int64")

    # Distance: great-circle ORD -> dest (BTS reports great-circle miles).
    o = airport_info(origin)
    o_lat, o_lon = o.get("lat"), o.get("lon")

    def _dist(dest: str) -> float:
        a = airport_info(dest)
        if o_lat is None or a.get("lat") is None:
            return np.nan
        return haversine_miles(o_lat, o_lon, a["lat"], a["lon"])

    df["distance"] = df["dest"].map(_dist)

    # Calendar parts from ORIGIN-local scheduled departure (BTS flight_date
    # is the ORD-local operating day). day_of_week: 1=Mon..7=Sun (BTS/ISO).
    dep_local = dep_utc.dt.tz_convert(origin_tz)
    df["month"] = dep_local.dt.month.astype("Int64")
    df["day_of_month"] = dep_local.dt.day.astype("Int64")
    df["day_of_week"] = dep_local.dt.isocalendar().day.astype("Int64")
    df["quarter"] = dep_local.dt.quarter.astype("Int64")

    df["dest_state"] = df["dest"].map(
        lambda d: state_abbrev(airport_info(d).get("subd")))

    df["scheduled_dep_utc_hour"] = dep_utc.dt.floor("h")
    df["scheduled_arr_utc_hour"] = arr_utc.dt.floor("h")
    return df


def distinct_stations(features: pd.DataFrame, *,
                      origin: str = ORIGIN_DEFAULT) -> list[str]:
    """Every airport whose weather we need: ORD plus all destinations."""
    dests = features["dest"].dropna().astype("string").str.upper().unique()
    return sorted({origin.upper(), *dests})


# --------------------------------------------------------------------------- #
# AeroDataBox adapter (reference provider)
# --------------------------------------------------------------------------- #


class AeroDataBoxError(RuntimeError):
    """AeroDataBox request failed with a useful, UI-safe message."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 detail: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _aerodatabox_error(resp: requests.Response, path: str) -> AeroDataBoxError:
    """Build a concise error from the gateway's JSON (or text) error payload."""
    detail = None
    title = None
    try:
        payload = resp.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        title = payload.get("title") or payload.get("error")
        detail = (payload.get("message") or payload.get("detail")
                  or payload.get("description"))
    else:
        detail = resp.text.strip()[:500]

    bits = [f"AeroDataBox {resp.status_code} {resp.reason}"]
    if title:
        bits.append(str(title))
    if detail:
        bits.append(str(detail))
    bits.append(f"path={path}")
    return AeroDataBoxError(": ".join(bits), status_code=resp.status_code,
                            detail=detail)


def _retry_after_seconds(resp: requests.Response | None) -> float | None:
    if resp is None:
        return None
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Raw-response TTL cache
# --------------------------------------------------------------------------- #


def _response_cache_path(path: str, params: dict,
                         cache_dir: Path = DEFAULT_RESPONSE_CACHE_DIR) -> Path:
    """Stable on-disk path for a (gateway, path, params) request (key excluded)."""
    key = _base_url() + "|" + path + "?" + urlencode(sorted(params.items()))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _read_response_cache(cache_file: Path, ttl_s: float) -> dict | None:
    if ttl_s <= 0 or not cache_file.exists():
        return None
    age = time.time() - cache_file.stat().st_mtime
    if age > ttl_s:
        return None
    try:
        return json.loads(cache_file.read_text())
    except Exception as err:  # noqa: BLE001 - corrupt cache should not kill UI
        logger.warning("ignoring corrupt response cache %s (%s)",
                       cache_file, err)
        return None


def _write_response_cache(cache_file: Path, data: dict) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    except Exception as err:  # noqa: BLE001 - caching is best-effort
        logger.warning("could not write response cache %s (%s)",
                       cache_file, err)


def _aerodatabox_get(path: str, params: dict, api_key: str,
                     *, timeout: int = 30, retries: int = 4,
                     cache_ttl: float = DEFAULT_CACHE_TTL_S,
                     cache_dir: Path = DEFAULT_RESPONSE_CACHE_DIR) -> dict:
    """GET an AeroDataBox endpoint with a TTL response cache + retry/backoff."""
    cache_file = _response_cache_path(path, params, cache_dir)
    cached = _read_response_cache(cache_file, cache_ttl)
    if cached is not None:
        logger.debug("AeroDataBox cache hit %s", path)
        return cached

    url = f"{_base_url()}{path}"
    headers = _auth_headers(api_key)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params,
                                timeout=(15, timeout))
            if resp.status_code == 204:        # empty window -> no flights
                data: dict = {"departures": [], "arrivals": []}
            elif resp.status_code >= 400:
                err = _aerodatabox_error(resp, path)
                status = resp.status_code
                permanent = 400 <= status < 500 and status != 429
                if permanent:
                    raise err
                raise requests.exceptions.HTTPError(str(err), response=resp)
            else:
                data = resp.json()
            if cache_ttl > 0:
                _write_response_cache(cache_file, data)
            return data
        except AeroDataBoxError:
            raise
        except requests.exceptions.RequestException as err:
            last_err = err
            if attempt < retries:
                response = getattr(err, "response", None)
                backoff = _retry_after_seconds(response)
                if backoff is None:
                    backoff = min(60, 5 * 2 ** (attempt - 1))
                logger.warning("  AeroDataBox attempt %d/%d failed (%s); "
                               "retry in %.0fs", attempt, retries, err,
                               backoff)
                time.sleep(backoff)
    msg = f"AeroDataBox request failed after {retries} attempts: path={path}"
    if last_err is not None:
        msg = f"{msg}: {last_err}"
    raise AeroDataBoxError(msg) from last_err


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def _utc(node: dict | None, field: str) -> str | None:
    """Pull the UTC string from an AeroDataBox DateTimeContract sub-field."""
    if not node:
        return None
    dt = node.get(field)
    if isinstance(dt, dict):
        return dt.get("utc")
    return None


def _delay15(sched: str | None, actual: str | None) -> float:
    """1 if actual is >= 15 min after scheduled, 0 if earlier, NaN if unknown."""
    if not sched or not actual:
        return np.nan
    s = pd.to_datetime(sched, utc=True, errors="coerce")
    a = pd.to_datetime(actual, utc=True, errors="coerce")
    if pd.isna(s) or pd.isna(a):
        return np.nan
    return int((a - s).total_seconds() >= DELAY15_SECONDS)


def _adb_flight_to_row(f: dict) -> dict:
    """Normalize one AeroDataBox AirportFlightContract into NORMALIZED schema.

    For a departing flight requested with ``withLeg=true`` the contract has
    ``departure`` (origin = ORD, departure times) and ``arrival``
    (destination, arrival times). When leg info is unavailable the API sets
    a single ``movement`` block instead, whose ``airport`` is the *opposite*
    (destination) airport but which carries no arrival time -- such rows lack
    ``scheduled_arr_utc`` and are dropped downstream.
    """
    dep_node = f.get("departure") or f.get("movement") or {}
    arr_node = f.get("arrival") or {}

    if f.get("arrival"):
        dest = (arr_node.get("airport") or {}).get("iata")
    elif f.get("movement"):
        dest = ((f["movement"].get("airport") or {}).get("iata"))
    else:
        dest = None

    dep_sched = _utc(dep_node, "scheduledTime")
    arr_sched = _utc(arr_node, "scheduledTime")
    airline = f.get("airline") or {}
    status = f.get("status")
    return {
        "ident": f.get("number") or f.get("callSign"),
        "reporting_airline": airline.get("iata") or airline.get("icao"),
        "dest": dest,
        "scheduled_dep_utc": dep_sched,
        "scheduled_arr_utc": arr_sched,
        "cancelled": int(status in ("Canceled", "CanceledUncertain")),
        "diverted": int(status == "Diverted"),
        "dep_del15": _delay15(dep_sched, _utc(dep_node, "revisedTime")),
        "arr_del15": _delay15(arr_sched, _utc(arr_node, "revisedTime")),
    }


def aerodatabox_to_normalized(flights: list[dict]) -> pd.DataFrame:
    """Turn a list of AeroDataBox flight objects into the normalized schema."""
    rows = [_adb_flight_to_row(f) for f in flights]
    df = pd.DataFrame(rows, columns=NORMALIZED_COLS)
    df["scheduled_dep_utc"] = pd.to_datetime(
        df["scheduled_dep_utc"], utc=True, errors="coerce")
    df["scheduled_arr_utc"] = pd.to_datetime(
        df["scheduled_arr_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dest", "scheduled_dep_utc", "scheduled_arr_utc"])
    # Dedupe flights that straddle two adjacent 12 h chunk boundaries.
    df = df.drop_duplicates(subset=["ident", "dest", "scheduled_dep_utc"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# FIDS fetch (by local time range, chunked to <= 12 h windows)
# --------------------------------------------------------------------------- #


def _local_time_windows(start_utc: datetime, end_utc: datetime, tz: str,
                        *, max_window_h: int = FIDS_MAX_WINDOW_H
                        ) -> list[tuple[str, str]]:
    """Split ``[start_utc, end_utc)`` into <= ``max_window_h`` local windows.

    Returned bounds are naive local wall-clock strings (``YYYY-MM-DDTHH:mm``)
    in the airport timezone, the format the FIDS endpoint expects. Stepping
    the naive local clock keeps every window <= ``max_window_h`` by string
    arithmetic regardless of DST.
    """
    start_local = pd.Timestamp(start_utc).tz_convert(tz).tz_localize(None)
    end_local = pd.Timestamp(end_utc).tz_convert(tz).tz_localize(None)
    if end_local <= start_local:
        return []
    step = pd.Timedelta(hours=max_window_h)
    windows: list[tuple[str, str]] = []
    cur = start_local
    while cur < end_local:
        nxt = min(cur + step, end_local)
        windows.append((cur.strftime("%Y-%m-%dT%H:%M"),
                        nxt.strftime("%Y-%m-%dT%H:%M")))
        cur = nxt
    return windows


def fetch_fids_departures(api_key: str, *, origin: str = ORIGIN_DEFAULT,
                          start_utc: datetime, end_utc: datetime,
                          cache_ttl: float = DEFAULT_CACHE_TTL_S,
                          sleep_s: float = 0.0) -> list[dict]:
    """All ORD departures in ``[start_utc, end_utc)`` via chunked FIDS calls."""
    tz = airport_info(origin).get("tz", "America/Chicago")
    windows = _local_time_windows(start_utc, end_utc, tz)
    params = {
        "direction": "Departure",
        "withLeg": "true",          # needed for the destination + arrival time
        "withCancelled": "true",
        "withCodeshared": "false",  # dedupe marketing/codeshare duplicates
        "withCargo": "false",
        "withPrivate": "false",
        "withLocation": "false",
    }
    flights: list[dict] = []
    for i, (from_local, to_local) in enumerate(windows):
        path = (f"/flights/airports/iata/{origin.upper()}"
                f"/{from_local}/{to_local}")
        data = _aerodatabox_get(path, params, api_key, cache_ttl=cache_ttl)
        chunk = data.get("departures") or []
        flights.extend(chunk)
        logger.info("FIDS %s %s..%s -> %d departures",
                    origin.upper(), from_local, to_local, len(chunk))
        if sleep_s > 0 and i < len(windows) - 1:
            time.sleep(sleep_s)
    return flights


def fetch_scheduled_departures(api_key: str, *, origin: str = ORIGIN_DEFAULT,
                               window_h: int = WINDOW_H,
                               now: datetime | None = None,
                               cache_ttl: float = DEFAULT_CACHE_TTL_S
                               ) -> pd.DataFrame:
    """Forward schedule: ORD departures in ``[now, now + window_h)``."""
    now = now or datetime.now(timezone.utc)
    end = now + timedelta(hours=window_h)
    flights = fetch_fids_departures(api_key, origin=origin, start_utc=now,
                                    end_utc=end, cache_ttl=cache_ttl)
    norm = aerodatabox_to_normalized(flights)
    logger.info("fetched %d scheduled departures from %s (%dh window)",
                len(norm), origin.upper(), window_h)
    return norm


def fetch_recent_departures(api_key: str, *, origin: str = ORIGIN_DEFAULT,
                            lookback_h: int = 168,
                            now: datetime | None = None,
                            cache_ttl: float = DEFAULT_CACHE_TTL_S,
                            sleep_s: float = 0.0) -> pd.DataFrame:
    """Completed departures in ``[now - lookback_h, now)`` for trailing rate."""
    if lookback_h <= 0:
        logger.info("skipping recent departures (lookback_h=%d)", lookback_h)
        return pd.DataFrame(columns=NORMALIZED_COLS)
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=lookback_h)
    flights = fetch_fids_departures(api_key, origin=origin, start_utc=start,
                                    end_utc=now, cache_ttl=cache_ttl,
                                    sleep_s=sleep_s)
    norm = aerodatabox_to_normalized(flights)
    logger.info("fetched %d recent departures from %s (%dh lookback)",
                len(norm), origin.upper(), lookback_h)
    return norm


# --------------------------------------------------------------------------- #
# Rolling recent-status cache (provider-agnostic)
# --------------------------------------------------------------------------- #


def _empty_recent_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=NORMALIZED_COLS)


def _coerce_recent_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cache dtypes after reading from disk or appending new rows."""
    if not len(df):
        return _empty_recent_cache()
    out = df.copy()
    for c in NORMALIZED_COLS:
        if c not in out:
            out[c] = pd.NA
    out = out[NORMALIZED_COLS]
    out["scheduled_dep_utc"] = pd.to_datetime(
        out["scheduled_dep_utc"], utc=True, errors="coerce")
    out["scheduled_arr_utc"] = pd.to_datetime(
        out["scheduled_arr_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["dest", "scheduled_dep_utc",
                             "scheduled_arr_utc"])
    for c in ("cancelled", "diverted"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int8")
    for c in ("dep_del15", "arr_del15"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


def load_recent_departures_cache(
    cache_path: Path = DEFAULT_RECENT_CACHE,
) -> pd.DataFrame:
    """Read the rolling normalized recent-departures cache if it exists."""
    if not cache_path.exists():
        return _empty_recent_cache()
    try:
        return _coerce_recent_cache(pd.read_parquet(cache_path))
    except Exception as err:  # noqa: BLE001 - corrupt cache should not kill UI
        logger.warning("could not read recent departures cache %s (%s)",
                       cache_path, err)
        return _empty_recent_cache()


def write_recent_departures_cache(
    cache: pd.DataFrame,
    cache_path: Path = DEFAULT_RECENT_CACHE,
) -> None:
    """Persist the normalized rolling recent-departures cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_recent_cache(cache).to_parquet(cache_path, index=False)


def _prune_recent_cache(df: pd.DataFrame, now: datetime,
                        retention_h: int) -> pd.DataFrame:
    if not len(df):
        return _empty_recent_cache()
    now_ts = pd.Timestamp(now)
    if now_ts.tz is None:
        now_ts = now_ts.tz_localize("UTC")
    cutoff = now_ts - pd.Timedelta(hours=retention_h)
    arr = pd.to_datetime(df["scheduled_arr_utc"], utc=True, errors="coerce")
    pruned = df.loc[arr >= cutoff].copy()
    return _coerce_recent_cache(pruned)


def merge_recent_departures_cache(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
    *,
    now: datetime,
    retention_h: int = DEFAULT_RECENT_CACHE_RETENTION_H,
) -> pd.DataFrame:
    """Append fresh statuses, prefer newest rows, and prune old rows."""
    pieces = [_coerce_recent_cache(f) for f in (existing, fresh) if len(f)]
    if not pieces:
        return _empty_recent_cache()
    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=RECENT_CACHE_DEDUPE_COLS, keep="last")
    combined = _prune_recent_cache(combined, now, retention_h)
    combined = combined.sort_values("scheduled_arr_utc").reset_index(drop=True)
    return combined


def update_recent_departures_cache(
    api_key: str,
    *,
    origin: str = ORIGIN_DEFAULT,
    now: datetime | None = None,
    refresh_h: int = DEFAULT_RECENT_CACHE_REFRESH_H,
    retention_h: int = DEFAULT_RECENT_CACHE_RETENTION_H,
    cache_path: Path = DEFAULT_RECENT_CACHE,
) -> pd.DataFrame:
    """Refresh and return the rolling recent-departures cache.

    The dashboard only needs a small incremental pull on each refresh. The
    cache accumulates those rows locally and keeps a rolling ``retention_h``
    history for the trailing-disruption features.
    """
    now = now or datetime.now(timezone.utc)
    existing = _prune_recent_cache(
        load_recent_departures_cache(cache_path), now, retention_h)

    fresh = _empty_recent_cache()
    if refresh_h > 0:
        try:
            fresh = fetch_recent_departures(
                api_key, origin=origin, lookback_h=refresh_h, now=now)
        except Exception as err:  # noqa: BLE001 - serve can use stale cache
            logger.warning("recent cache refresh failed (%s); using %d "
                           "cached rows", err, len(existing))

    cache = merge_recent_departures_cache(
        existing, fresh, now=now, retention_h=retention_h)
    if len(cache):
        write_recent_departures_cache(cache, cache_path)
        logger.info("recent cache has %d rows at %s", len(cache), cache_path)
    return cache


def backfill_recent_departures_cache(
    api_key: str,
    *,
    origin: str = ORIGIN_DEFAULT,
    now: datetime | None = None,
    lookback_h: int = DEFAULT_RECENT_CACHE_RETENTION_H,
    retention_h: int = DEFAULT_RECENT_CACHE_RETENTION_H,
    cache_path: Path = DEFAULT_RECENT_CACHE,
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """Seed the rolling recent-departures cache from a large FIDS pull.

    Unlike the dashboard's incremental refresh, this is a deliberate CLI
    operation and lets failures bubble up so the operator knows the backfill
    did not happen. ``sleep_s`` paces the chunked calls under the gateway's
    ~1 request/second limit.
    """
    now = now or datetime.now(timezone.utc)
    existing = _prune_recent_cache(
        load_recent_departures_cache(cache_path), now, retention_h)
    fresh = fetch_recent_departures(
        api_key, origin=origin, lookback_h=lookback_h, now=now,
        sleep_s=sleep_s)
    cache = merge_recent_departures_cache(
        existing, fresh, now=now, retention_h=retention_h)
    write_recent_departures_cache(cache, cache_path)
    logger.info("backfilled recent cache with %d fresh rows; cache now has "
                "%d rows at %s", len(fresh), len(cache), cache_path)
    return cache


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _require_key(args) -> str:
    key = args.api_key or os.environ.get("AERODATABOX_KEY")
    key_file = PROJECT_ROOT / "aerodatabox.key"
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    if not key:
        raise SystemExit(
            "No API key. Pass --api-key, set AERODATABOX_KEY, or create "
            "aerodatabox.key. (AeroDataBox via API.Market: "
            "https://api.market/store/aedbx/aerodatabox)")
    return key.strip()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["schedule", "recent", "backfill-cache"],
                   help="'schedule' = next-48h forward; "
                        "'recent' = completed export; "
                        "'backfill-cache' = seed rolling recent cache.")
    p.add_argument("--origin", default=ORIGIN_DEFAULT)
    p.add_argument("--window-h", type=int, default=WINDOW_H)
    p.add_argument("--lookback-h", type=int, default=168)
    p.add_argument("--retention-h", type=int,
                   default=DEFAULT_RECENT_CACHE_RETENTION_H,
                   help="Rolling cache retention for backfill-cache.")
    p.add_argument("--api-key", default=None,
                   help="AeroDataBox key (else $AERODATABOX_KEY / "
                        "aerodatabox.key).")
    p.add_argument("--provider", default=None,
                   choices=sorted(_GATEWAYS),
                   help="Marketplace your key is from (else "
                        "$AERODATABOX_PROVIDER, default apimarket).")
    p.add_argument("--sleep-s", type=float, default=1.1,
                   help="Pause between chunked FIDS calls (gateway allows "
                        "~1 request/second). Applies to recent/backfill-cache.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=DEFAULT_RECENT_CACHE,
                   help="Rolling cache path for backfill-cache.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.provider:
        os.environ["AERODATABOX_PROVIDER"] = args.provider
    key = _require_key(args)
    logger.info("using AeroDataBox gateway: %s", _provider())
    DEFAULT_LIVE_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "schedule":
        norm = fetch_scheduled_departures(
            key, origin=args.origin, window_h=args.window_h)
        feats = build_flight_features(norm, origin=args.origin)
        out = args.out or DEFAULT_LIVE_DIR / "flights_next48h.parquet"
        feats.to_parquet(out, index=False)
        logger.info("wrote %d flights x %d cols to %s",
                    len(feats), feats.shape[1], out)
        logger.info("distinct airports needing weather: %d",
                    len(distinct_stations(feats, origin=args.origin)))
    elif args.mode == "recent":
        norm = fetch_recent_departures(
            key, origin=args.origin, lookback_h=args.lookback_h,
            sleep_s=args.sleep_s)
        feats = build_flight_features(norm, origin=args.origin)
        # keep realised-status columns for trailing-rate labelling
        for c in ("cancelled", "diverted", "dep_del15", "arr_del15"):
            feats[c] = norm[c].to_numpy()
        out = args.out or DEFAULT_LIVE_DIR / "flights_recent.parquet"
        feats.to_parquet(out, index=False)
        logger.info("wrote %d flights x %d cols to %s",
                    len(feats), feats.shape[1], out)
    else:
        cache = backfill_recent_departures_cache(
            key, origin=args.origin, lookback_h=args.lookback_h,
            retention_h=args.retention_h, cache_path=args.cache,
            sleep_s=args.sleep_s)
        if len(cache):
            arr = pd.to_datetime(cache["scheduled_arr_utc"], utc=True)
            coverage_h = (
                (pd.Timestamp.now(tz="UTC") - arr.min()).total_seconds()
                / 3600.0)
            logger.info("cache arrival coverage: %.1fh (%s -> %s)",
                        max(0.0, coverage_h), arr.min(), arr.max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
