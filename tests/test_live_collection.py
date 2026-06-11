"""Tests for the live collection modules.

The HTTP layers (AeroDataBox / NWS) are not exercised over the network --
only the pure transforms are, against synthetic provider payloads. These are
exactly the pieces that, if wrong, silently break the model at serve time
(train/serve skew), so they are the ones worth pinning. The one networked
helper (:func:`_aerodatabox_get`) is tested only for its TTL-cache behaviour
with ``requests.get`` stubbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.live.backfill_recent_cache import iter_backfill_windows, run_backfill
from src.live.live_flights import (
    aerodatabox_to_normalized,
    backfill_recent_departures_cache,
    build_flight_features,
    distinct_stations,
    fetch_fids_departures,
    _aerodatabox_get,
    _auth_headers,
    _base_url,
    _local_time_windows,
    haversine_miles,
    load_recent_departures_cache,
    update_recent_departures_cache,
    state_abbrev,
)
from src.live.live_weather import (
    WX_COLS,
    _duration_hours,
    gridpoints_to_hourly,
)
from src.live.trailing_state import (
    compute_trailing_rates_asof,
    label_disrupted,
)


# --------------------------------------------------------------------------- #
# live_flights
# --------------------------------------------------------------------------- #


def test_haversine_matches_bts_distance():
    # ORD (41.977,-87.908) -> LAX (33.942,-118.408); BTS reports 1745 mi.
    d = haversine_miles(41.977, -87.908, 33.942, -118.408)
    assert abs(d - 1745) < 20


def test_state_abbrev():
    assert state_abbrev("Illinois") == "IL"
    assert state_abbrev("California") == "CA"
    assert state_abbrev("Puerto Rico") == "PR"
    assert state_abbrev(None) is None
    assert state_abbrev("Nowhere") is None


def _adb_flight(number, op_iata, dest, out_iso, in_iso,
                status="Departed", dep_revised=None, arr_revised=None):
    """Build one AeroDataBox AirportFlightContract for an ORD departure.

    Mirrors the ``withLeg=true`` shape: a ``departure`` block (origin = ORD)
    and an ``arrival`` block (destination), each with a DateTimeContract.
    """
    departure: dict = {
        "airport": {"iata": "ORD", "name": "Chicago O'Hare"},
        "scheduledTime": {"utc": out_iso, "local": out_iso},
    }
    if dep_revised is not None:
        departure["revisedTime"] = {"utc": dep_revised, "local": dep_revised}
    arrival: dict = {
        "airport": ({"iata": dest, "name": dest} if dest else {}),
        "scheduledTime": {"utc": in_iso, "local": in_iso},
    }
    if arr_revised is not None:
        arrival["revisedTime"] = {"utc": arr_revised, "local": arr_revised}
    return {
        "number": number,
        "callSign": None,
        "status": status,
        "codeshareStatus": "Unknown",
        "isCargo": False,
        "airline": {"name": op_iata, "iata": op_iata},
        "departure": departure,
        "arrival": arrival,
    }


def test_aerodatabox_to_normalized_and_features():
    flights = [
        _adb_flight("UA 123", "UA", "LAX",
                    "2026-05-25T14:30:00Z", "2026-05-25T19:05:00Z"),
        _adb_flight("AA 456", "AA", "JFK",
                    "2026-05-25T22:00:00Z", "2026-05-26T01:10:00Z"),
        # malformed -> dropped (no dest)
        _adb_flight("XX 999", "XX", None,
                    "2026-05-25T10:00:00Z", "2026-05-25T12:00:00Z"),
    ]
    norm = aerodatabox_to_normalized(flights)
    assert len(norm) == 2
    assert set(norm["dest"]) == {"LAX", "JFK"}

    feats = build_flight_features(norm, origin="ORD")
    lax = feats.loc[feats["dest"] == "LAX"].iloc[0]

    # crs_dep_time: 14:30 UTC in America/Chicago (CDT, -5h) = 09:30 -> 930
    assert lax["crs_dep_time"] == 930
    # crs_arr_time: 19:05 UTC in America/Los_Angeles (PDT, -7h) = 12:05 -> 1205
    assert lax["crs_arr_time"] == 1205
    # elapsed 14:30 -> 19:05 = 275 min
    assert lax["crs_elapsed_time"] == 275
    assert abs(lax["distance"] - 1745) < 20
    assert lax["dest_state"] == "CA"
    # day_of_week: 2026-05-25 is a Monday -> ISO 1
    assert lax["day_of_week"] == 1
    assert lax["month"] == 5 and lax["quarter"] == 2
    # join keys floored to the hour
    assert lax["scheduled_dep_utc_hour"] == pd.Timestamp("2026-05-25T14:00:00Z")
    assert lax["scheduled_arr_utc_hour"] == pd.Timestamp("2026-05-25T19:00:00Z")


def test_aerodatabox_status_and_delays():
    flights = [
        # on time (revised == scheduled)
        _adb_flight("UA 1", "UA", "DEN", "2026-05-25T14:00:00Z",
                    "2026-05-25T16:00:00Z", status="Arrived",
                    dep_revised="2026-05-25T14:00:00Z",
                    arr_revised="2026-05-25T16:00:00Z"),
        # 20 min late arrival -> arr_del15 = 1
        _adb_flight("UA 2", "UA", "SFO", "2026-05-25T15:00:00Z",
                    "2026-05-25T18:00:00Z", status="Arrived",
                    dep_revised="2026-05-25T15:05:00Z",   # 5 min -> 0
                    arr_revised="2026-05-25T18:20:00Z"),  # 20 min -> 1
        # cancelled
        _adb_flight("UA 3", "UA", "BOS", "2026-05-25T16:00:00Z",
                    "2026-05-25T19:00:00Z", status="Canceled"),
    ]
    norm = aerodatabox_to_normalized(flights).set_index("ident")

    assert norm.loc["UA 1", "dep_del15"] == 0   # revised == scheduled
    assert norm.loc["UA 1", "arr_del15"] == 0
    assert norm.loc["UA 2", "dep_del15"] == 0
    assert norm.loc["UA 2", "arr_del15"] == 1
    assert norm.loc["UA 3", "cancelled"] == 1
    # cancelled row has no revisedTime -> NaN delays
    assert np.isnan(norm.loc["UA 3", "arr_del15"])
    # label_disrupted recovers BTS disrupted from the realised status
    disrupted = label_disrupted(norm.reset_index())
    assert disrupted.tolist() == [0, 1, 1]


def test_distinct_stations_includes_origin():
    norm = aerodatabox_to_normalized([
        _adb_flight("UA 1", "UA", "LAX",
                    "2026-05-25T14:00:00Z", "2026-05-25T18:00:00Z"),
        _adb_flight("UA 2", "UA", "DEN",
                    "2026-05-25T15:00:00Z", "2026-05-25T17:00:00Z"),
    ])
    feats = build_flight_features(norm)
    assert distinct_stations(feats, origin="ORD") == ["DEN", "LAX", "ORD"]


def test_local_time_windows_chunk_to_12h():
    # 48 h from 2026-05-25T12:00Z (ORD CDT = 07:00 local) -> four 12 h windows.
    now = pd.Timestamp("2026-05-25T12:00:00Z")
    windows = _local_time_windows(now, now + pd.Timedelta(hours=48),
                                  "America/Chicago")
    assert len(windows) == 4
    assert windows[0] == ("2026-05-25T07:00", "2026-05-25T19:00")
    assert windows[1] == ("2026-05-25T19:00", "2026-05-26T07:00")
    # every window is naive local wall-clock and spans <= 12 h
    for lo, hi in windows:
        span = pd.Timestamp(hi) - pd.Timestamp(lo)
        assert span <= pd.Timedelta(hours=12)


def test_fetch_fids_departures_chunks_and_dedupes(monkeypatch):
    calls = []

    def fake_get(path, params, api_key, *, cache_ttl):
        calls.append({"path": path, "params": params, "api_key": api_key})
        # Same flight in every chunk so we can prove cross-chunk dedupe.
        return {"departures": [
            _adb_flight("UA 1", "UA", "DEN",
                        "2026-05-25T14:00:00Z", "2026-05-25T16:00:00Z")]}

    monkeypatch.setattr("src.live.live_flights._aerodatabox_get", fake_get)
    now = pd.Timestamp("2026-05-25T12:00:00Z").to_pydatetime()
    flights = fetch_fids_departures(
        "secret", origin="ORD", start_utc=now,
        end_utc=now + pd.Timedelta(hours=48))

    # 48 h -> four 12 h FIDS calls, all to the IATA airport path.
    assert len(calls) == 4
    assert all(c["api_key"] == "secret" for c in calls)
    assert calls[0]["path"].startswith("/flights/airports/iata/ORD/")
    p = calls[0]["params"]
    assert p["direction"] == "Departure"
    assert p["withLeg"] == "true"
    assert p["withCodeshared"] == "false"
    assert p["withCargo"] == "false"
    # four identical raw flights returned; normalization dedupes to one.
    assert len(flights) == 4
    assert len(aerodatabox_to_normalized(flights)) == 1


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.reason = "OK"
        self.headers = {}

    def json(self):
        return self._payload


def test_gateway_selects_base_and_auth_headers(monkeypatch):
    # default marketplace = API.Market
    monkeypatch.delenv("AERODATABOX_PROVIDER", raising=False)
    monkeypatch.delenv("AERODATABOX_BASE", raising=False)
    assert _base_url() == "https://prod.api.market/api/v1/aedbx/aerodatabox"
    assert _auth_headers("k")["x-magicapi-key"] == "k"
    assert "X-RapidAPI-Host" not in _auth_headers("k")

    # RapidAPI key -> RapidAPI host + two headers
    monkeypatch.setenv("AERODATABOX_PROVIDER", "rapidapi")
    assert _base_url() == "https://aerodatabox.p.rapidapi.com"
    h = _auth_headers("k")
    assert h["X-RapidAPI-Key"] == "k"
    assert h["X-RapidAPI-Host"] == "aerodatabox.p.rapidapi.com"


def test_aerodatabox_get_uses_ttl_cache(monkeypatch, tmp_path):
    n = {"calls": 0}

    def fake_requests_get(url, headers, params, timeout):
        n["calls"] += 1
        # the auth header must be the API.Market header
        assert "x-magicapi-key" in headers
        return _FakeResp({"departures": [{"number": "UA 1"}]})

    monkeypatch.setattr("src.live.live_flights.requests.get", fake_requests_get)

    kwargs = dict(timeout=5, retries=1, cache_ttl=600, cache_dir=tmp_path)
    a = _aerodatabox_get("/p", {"x": "1"}, "secret", **kwargs)
    b = _aerodatabox_get("/p", {"x": "1"}, "secret", **kwargs)  # cache hit
    assert a == b == {"departures": [{"number": "UA 1"}]}
    assert n["calls"] == 1                       # second call served from cache

    # ttl=0 disables the cache -> a real second request
    _aerodatabox_get("/p", {"x": "1"}, "secret", timeout=5, retries=1,
                     cache_ttl=0, cache_dir=tmp_path)
    assert n["calls"] == 2


def test_update_recent_departures_cache_appends_dedupes_and_prunes(
    monkeypatch, tmp_path,
):
    now = pd.Timestamp("2026-05-25T12:00:00Z")
    cache_path = tmp_path / "recent_cache.parquet"
    existing = pd.DataFrame([
        {
            "ident": "OLD1", "reporting_airline": "UA", "dest": "LAX",
            "scheduled_dep_utc": "2026-05-24T22:00:00Z",
            "scheduled_arr_utc": "2026-05-25T04:00:00Z",
            "cancelled": 0, "diverted": 0, "dep_del15": 0, "arr_del15": 0,
        },
        {
            "ident": "UAL1", "reporting_airline": "UA", "dest": "DEN",
            "scheduled_dep_utc": "2026-05-25T09:00:00Z",
            "scheduled_arr_utc": "2026-05-25T11:00:00Z",
            "cancelled": 0, "diverted": 0, "dep_del15": 0, "arr_del15": 0,
        },
    ])
    existing.to_parquet(cache_path, index=False)

    fresh = pd.DataFrame([
        {
            "ident": "UAL1", "reporting_airline": "UA", "dest": "DEN",
            "scheduled_dep_utc": "2026-05-25T09:00:00Z",
            "scheduled_arr_utc": "2026-05-25T11:00:00Z",
            "cancelled": 0, "diverted": 0, "dep_del15": 1, "arr_del15": 0,
        },
        {
            "ident": "AAL2", "reporting_airline": "AA", "dest": "DFW",
            "scheduled_dep_utc": "2026-05-25T10:00:00Z",
            "scheduled_arr_utc": "2026-05-25T12:30:00Z",
            "cancelled": 0, "diverted": 0, "dep_del15": 0, "arr_del15": 1,
        },
    ])
    seen = {}

    def fake_recent(api_key, *, origin, lookback_h, now):
        seen.update(api_key=api_key, origin=origin, lookback_h=lookback_h)
        return fresh

    monkeypatch.setattr("src.live.live_flights.fetch_recent_departures",
                        fake_recent)
    cache = update_recent_departures_cache(
        "secret", origin="ORD", now=now.to_pydatetime(), refresh_h=3,
        retention_h=6, cache_path=cache_path)

    assert seen == {"api_key": "secret", "origin": "ORD", "lookback_h": 3}
    assert set(cache["ident"]) == {"UAL1", "AAL2"}
    assert cache.loc[cache["ident"] == "UAL1", "dep_del15"].iloc[0] == 1
    reread = load_recent_departures_cache(cache_path)
    assert set(reread["ident"]) == {"UAL1", "AAL2"}


def test_backfill_recent_departures_cache_writes(monkeypatch, tmp_path):
    now = pd.Timestamp("2026-05-25T12:00:00Z")
    cache_path = tmp_path / "recent_cache.parquet"
    fresh = pd.DataFrame([
        {
            "ident": "UAL1", "reporting_airline": "UA", "dest": "DEN",
            "scheduled_dep_utc": "2026-05-24T10:00:00Z",
            "scheduled_arr_utc": "2026-05-24T12:00:00Z",
            "cancelled": 0, "diverted": 0, "dep_del15": 1, "arr_del15": 0,
        },
        {
            "ident": "AAL2", "reporting_airline": "AA", "dest": "DFW",
            "scheduled_dep_utc": "2026-05-25T09:00:00Z",
            "scheduled_arr_utc": "2026-05-25T11:00:00Z",
            "cancelled": 0, "diverted": 0, "dep_del15": 0, "arr_del15": 1,
        },
    ])
    seen = {}

    def fake_recent(api_key, *, origin, lookback_h, now, sleep_s=0.0):
        seen.update(api_key=api_key, origin=origin, lookback_h=lookback_h)
        return fresh

    monkeypatch.setattr("src.live.live_flights.fetch_recent_departures",
                        fake_recent)
    cache = backfill_recent_departures_cache(
        "secret", origin="ORD", now=now.to_pydatetime(), lookback_h=168,
        retention_h=168, cache_path=cache_path)

    assert seen == {"api_key": "secret", "origin": "ORD", "lookback_h": 168}
    assert set(cache["ident"]) == {"UAL1", "AAL2"}
    assert cache_path.exists()
    reread = load_recent_departures_cache(cache_path)
    assert set(reread["ident"]) == {"UAL1", "AAL2"}


def test_slow_backfill_windows_are_newest_first():
    windows = iter_backfill_windows(
        "2026-05-25T12:00:00Z", lookback_h=3, chunk_h=1)

    assert windows == [
        (pd.Timestamp("2026-05-25T11:00:00Z"),
         pd.Timestamp("2026-05-25T12:00:00Z")),
        (pd.Timestamp("2026-05-25T10:00:00Z"),
         pd.Timestamp("2026-05-25T11:00:00Z")),
        (pd.Timestamp("2026-05-25T09:00:00Z"),
         pd.Timestamp("2026-05-25T10:00:00Z")),
    ]


def test_run_slow_backfill_writes_each_chunk(monkeypatch, tmp_path):
    calls = []

    def fake_recent(api_key, *, origin, lookback_h, now):
        end = pd.Timestamp(now)
        calls.append((api_key, origin, lookback_h, end))
        return pd.DataFrame([{
            "ident": f"UAL{len(calls)}",
            "reporting_airline": "UA",
            "dest": "DEN",
            "scheduled_dep_utc": end - pd.Timedelta(minutes=50),
            "scheduled_arr_utc": end - pd.Timedelta(minutes=10),
            "cancelled": 0,
            "diverted": 0,
            "dep_del15": int(len(calls) == 1),
            "arr_del15": 0,
        }])

    monkeypatch.setattr("src.live.backfill_recent_cache.fetch_recent_departures",
                        fake_recent)
    cache_path = tmp_path / "recent_cache.parquet"
    cache = run_backfill(
        api_key="secret", now="2026-05-25T12:00:00Z", lookback_h=2,
        chunk_h=1, sleep_s=0, cache_path=cache_path)

    assert len(calls) == 2
    assert [c[3] for c in calls] == [
        pd.Timestamp("2026-05-25T12:00:00Z"),
        pd.Timestamp("2026-05-25T11:00:00Z"),
    ]
    assert set(cache["ident"]) == {"UAL1", "UAL2"}
    assert cache_path.exists()


def test_build_features_empty():
    empty = pd.DataFrame(columns=["ident", "reporting_airline", "dest",
                                   "scheduled_dep_utc", "scheduled_arr_utc"])
    feats = build_flight_features(empty)
    assert len(feats) == 0
    assert "scheduled_dep_utc_hour" in feats.columns


# --------------------------------------------------------------------------- #
# live_weather
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dur,hours", [
    ("PT1H", 1), ("PT3H", 3), ("P1DT14H", 38), ("PT0H", 1), ("P2D", 48),
    ("PT30M", 1),
])
def test_duration_hours(dur, hours):
    assert _duration_hours(dur) == hours


def test_gridpoints_to_hourly_units_and_expansion():
    props = {
        "temperature": {"uom": "wmoUnit:degC", "values": [
            {"validTime": "2026-05-25T12:00:00+00:00/PT2H", "value": 20.0}]},
        "dewpoint": {"values": [
            {"validTime": "2026-05-25T12:00:00+00:00/PT2H", "value": 10.0}]},
        "windSpeed": {"values": [        # km/h -> kt
            {"validTime": "2026-05-25T12:00:00+00:00/PT1H", "value": 18.52}]},
        "windGust": {"values": [
            {"validTime": "2026-05-25T12:00:00+00:00/PT1H", "value": 37.04}]},
        "windDirection": {"values": [
            {"validTime": "2026-05-25T12:00:00+00:00/PT2H", "value": 270}]},
        "quantitativePrecipitation": {"values": [   # mm -> in
            {"validTime": "2026-05-25T12:00:00+00:00/PT1H", "value": 25.4}]},
        "relativeHumidity": {"values": [
            {"validTime": "2026-05-25T12:00:00+00:00/PT2H", "value": 80}]},
    }
    hourly = gridpoints_to_hourly(props)

    # PT2H temperature expands to two hourly rows
    assert len(hourly) == 2
    assert list(hourly.columns) == ["valid_utc", *WX_COLS]

    row0 = hourly.iloc[0]
    assert row0["valid_utc"] == pd.Timestamp("2026-05-25T12:00:00Z")
    assert row0["tmpf"] == pytest.approx(68.0)      # 20C
    assert row0["dwpf"] == pytest.approx(50.0)      # 10C
    assert row0["sknt"] == pytest.approx(10.0, abs=0.01)   # 18.52 km/h
    assert row0["gust"] == pytest.approx(20.0, abs=0.01)
    assert row0["drct"] == 270
    assert row0["p01i"] == pytest.approx(1.0, abs=0.001)   # 25.4 mm
    assert row0["relh"] == 80

    # Unavailable forecast fields are present but NaN.
    for col in ["vsby", "mslp", "alti", "peak_wind_gust",
                "snowdepth", "n_reports", "skyl1"]:
        assert np.isnan(row0[col])

    # windSpeed only had a PT1H interval, so hour 2 is NaN for sknt.
    assert np.isnan(hourly.iloc[1]["sknt"])
    assert hourly.iloc[1]["tmpf"] == pytest.approx(68.0)  # temp covered both


# --------------------------------------------------------------------------- #
# trailing_state
# --------------------------------------------------------------------------- #


def test_label_disrupted():
    df = pd.DataFrame({
        "dep_del15": [1, 0, 0, 0, np.nan],
        "arr_del15": [0, 1, 0, 0, 0],
        "cancelled": [0, 0, 1, 0, 0],
        "diverted":  [0, 0, 0, 1, 0],
    })
    assert label_disrupted(df).tolist() == [1, 1, 1, 1, 0]


def _completed(now: pd.Timestamp):
    """Five completed flights at known offsets before `now`, by sched arr."""
    return pd.DataFrame({
        "scheduled_arr_utc_hour": [
            now - pd.Timedelta(hours=3),    # in 6h & 24h windows
            now - pd.Timedelta(hours=10),   # in 24h window
            now - pd.Timedelta(hours=50),   # in 72h window
            now - pd.Timedelta(hours=100),  # in 168h window
            now,                            # AT now -> excluded (strict <)
        ],
        "disrupted": [1, 0, 1, 0, 1],
    })


def test_trailing_rate_windows_and_strict_bound():
    now = pd.Timestamp("2026-05-25T12:00:00Z")
    rates = compute_trailing_rates_asof(now, _completed(now))
    # 6h: only the -3h flight (disrupted) -> 1.0
    assert rates[6] == pytest.approx(1.0)
    # 24h: -3h (1) and -10h (0) -> 0.5
    assert rates[24] == pytest.approx(0.5)
    # 72h: -3,-10,-50 -> (1+0+1)/3
    assert rates[72] == pytest.approx(2 / 3)
    # 168h: first four -> (1+0+1+0)/4 = 0.5  (the AT-now flight is excluded)
    assert rates[168] == pytest.approx(0.5)


def test_trailing_rate_excludes_flight_at_now():
    # Put the ONLY disrupted flight exactly at `now`: strict < excludes it.
    now = pd.Timestamp("2026-05-25T12:00:00Z")
    pool = pd.DataFrame({
        "scheduled_arr_utc_hour": [now, now - pd.Timedelta(hours=1)],
        "disrupted": [1, 0],
    })
    rates = compute_trailing_rates_asof(now, pool)
    assert rates[6] == pytest.approx(0.0)  # only the -1h (clean) flight counts


def test_trailing_rate_empty_pool_is_nan():
    now = pd.Timestamp("2026-05-25T12:00:00Z")
    pool = pd.DataFrame({"scheduled_arr_utc_hour": pd.to_datetime([], utc=True),
                         "disrupted": []})
    rates = compute_trailing_rates_asof(now, pool)
    assert all(np.isnan(v) for v in rates.values())
