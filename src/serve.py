"""Serving pipeline: assemble features and score the production model.

This is the "prediction API" the dashboard calls. :func:`predict_next_48h`
runs the whole chain end to end and returns per-flight probabilities plus a
48 h summary:

    flights (next 48h)   -> live_flights (AeroDataBox)  | demo replay
    + weather forecast   -> live_weather (NWS)          | already joined
    + trailing op-state  -> trailing_state              | from history
    -> prepare_features  -> production LightGBM -> P(disrupted) per flight
    -> aggregate         -> 48 h occasion disruption rate

Two data sources feed one scoring core (:func:`score`):

* ``source="live"`` -- real next-48 h ORD schedule + NWS forecasts +
  recent completed flights. Needs an AeroDataBox (API.Market) key.
* ``source="demo"`` -- replay a real window from the joined parquet through
  the identical scoring path. No API key, no network; also carries the
  realised ``disrupted`` labels so the UI can show predicted-vs-actual.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ingest_bts_ord import PROJECT_ROOT
from src.training_pipeline import (
    DEFAULT_JOINED, WINDOW_H, _REFERENCE_DATE, prepare_features,
)
from src.trailing_state import (
    attach_operational_features, compute_trailing_rates_asof, label_disrupted,
)

logger = logging.getLogger("serve")

PRODUCTION_BUNDLE = (
    PROJECT_ROOT / "artifacts" / "production_model" / "model.joblib")

# AeroDataBox (API.Market) key file (git-ignored). One line, the key only.
AERODATABOX_KEY_FILE = PROJECT_ROOT / "aerodatabox.key"


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the AeroDataBox key: explicit arg -> ``aerodatabox.key`` -> env.

    Returns ``None`` if no key is found anywhere.
    """
    if explicit:
        return explicit.strip()
    if AERODATABOX_KEY_FILE.exists():
        key = AERODATABOX_KEY_FILE.read_text().strip()
        if key:
            return key
    env = os.environ.get("AERODATABOX_KEY")
    return env.strip() if env else None

# A demo "now" whose next-48 h window straddles the Feb 2026 disruption
# spike, so the replayed dashboard shows real variation out of the box.
DEMO_DEFAULT_NOW = pd.Timestamp("2026-02-21 06:00", tz="UTC")

# Risk banding for display.
RISK_BANDS = [(0.0, 0.20, "Low"), (0.20, 0.40, "Moderate"),
              (0.40, 0.60, "Elevated"), (0.60, 1.01, "High")]


def risk_band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "Unknown"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def load_model_bundle(path: Path = PRODUCTION_BUNDLE) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"production model not found at {path}. "
            f"Run `python -m src.train_production_model` first.")
    bundle = joblib.load(path)
    logger.info("loaded production model (%d features, isotonic=%s)",
                len(bundle["feature_names"]), bundle.get("uses_isotonic"))
    return bundle


# --------------------------------------------------------------------------- #
# Scoring core (shared by live + demo)
# --------------------------------------------------------------------------- #


def score(flights: pd.DataFrame, now: pd.Timestamp, bundle: dict, *,
          completed_pool: pd.DataFrame | None = None,
          trailing_rates: dict[int, float] | None = None) -> pd.DataFrame:
    """Score an assembled flight frame; returns it with a ``p_hat`` column.

    ``flights`` must carry the schedule/route columns, the ``origin_wx_*`` /
    ``dest_wx_*`` weather columns, and ``scheduled_dep_utc_hour`` /
    ``scheduled_arr_utc_hour``.

    Trailing operational state comes from exactly one of:
      * ``completed_pool`` -- compute the trailing rate as of ``now`` from a
        pool of completed flights (live / replay, where recent flights exist
        before ``now``); or
      * ``trailing_rates`` -- a precomputed ``{window_h: rate}`` snapshot
        (upcoming forecast, where ``now`` sits *after* the labelled data so
        the as-of-now window is empty -- use the most recent known state).
    """
    now = pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("UTC")
    df = flights.copy()

    # Real leads from the single issue time `now`.
    dep = pd.to_datetime(df["scheduled_dep_utc_hour"], utc=True)
    df["lead_origin_h"] = ((dep - now).dt.total_seconds() / 3600.0)
    elapsed_h = pd.to_numeric(df["crs_elapsed_time"], errors="coerce") / 60.0
    df["lead_dest_h"] = df["lead_origin_h"] + elapsed_h.to_numpy()

    # Operational state (trailing rate snapshot + time trend).
    if trailing_rates is not None:
        for w, r in trailing_rates.items():
            df[f"trailing_disruption_rate_{w}h"] = r
        df["days_since_2021"] = (
            (dep - _REFERENCE_DATE).dt.total_seconds() / 86400.0)
    else:
        df = attach_operational_features(df, completed_pool, now=now)

    X, _ = prepare_features(df, cat_categories=bundle["categories"])
    X = X[bundle["feature_names"]]          # drops n_reports, fixes order
    df["p_hat"] = bundle["model"].predict_proba(X)[:, 1]
    df["risk"] = df["p_hat"].map(risk_band)
    return df


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class Prediction:
    as_of: pd.Timestamp
    source: str
    flights: pd.DataFrame                  # tidy, display-ready
    predicted_rate: float                  # mean P over the 48 h window
    n_flights: int
    actual_rate: float | None = None       # demo only
    by_hour: pd.DataFrame = field(default_factory=pd.DataFrame)


def _airport_name(iata: str) -> str:
    from src.live_flights import airport_info
    info = airport_info(iata)
    city = info.get("city") or info.get("name") or ""
    return str(city)


def _tidy(scored: pd.DataFrame, now: pd.Timestamp,
          origin: str = "ORD") -> pd.DataFrame:
    """Build the display table the dashboard renders.

    Departure time is shown as the **exact** scheduled clock time in
    **America/Chicago** (ORD local), reconstructed from ``crs_dep_time``
    (the local HHMM) rather than the hour-floored UTC join key.
    """
    out = pd.DataFrame()
    # Exact Central-time departure: take the date from the (whole-hour)
    # UTC key converted to Chicago, then overlay the exact HH:MM from
    # crs_dep_time (Central offset is whole hours, so the date is preserved).
    ct_date = (pd.to_datetime(scored["scheduled_dep_utc_hour"], utc=True)
               .dt.tz_convert("America/Chicago").dt.normalize())
    hhmm = pd.to_numeric(scored["crs_dep_time"], errors="coerce")
    hh = (hhmm // 100).fillna(0).astype(int)
    mm = (hhmm % 100).fillna(0).astype(int)
    dep_ct = (ct_date + pd.to_timedelta(hh, unit="h")
              + pd.to_timedelta(mm, unit="m")).dt.tz_localize(None)
    air = scored["reporting_airline"].astype("string")
    if "ident" in scored and scored["ident"].notna().any():
        flight = scored["ident"].astype("string")
    elif "flight_number_reporting_airline" in scored:
        num = (pd.to_numeric(scored["flight_number_reporting_airline"],
                             errors="coerce")
               .astype("Int64").astype("string"))
        flight = air.str.cat(num, sep=" ")
    else:
        flight = air
    out["flight"] = flight
    out["airline"] = air
    out["dest"] = scored["dest"].astype("string")
    out["dest_city"] = out["dest"].map(_airport_name)
    out["dep_ct"] = dep_ct
    out["lead_h"] = scored["lead_origin_h"].round(1)
    out["p_disrupted"] = scored["p_hat"].round(4)
    out["risk"] = scored["risk"].astype("string")
    if "disrupted" in scored:
        out["actual_disrupted"] = pd.to_numeric(
            scored["disrupted"], errors="coerce").astype("Int64")
    return out.sort_values("dep_ct").reset_index(drop=True)


def _summarize(scored: pd.DataFrame, tidy: pd.DataFrame,
               now: pd.Timestamp, source: str) -> Prediction:
    by_hour = (tidy.assign(dep_hour=tidy["dep_ct"].dt.floor("h"))
                   .groupby("dep_hour")
                   .agg(predicted_rate=("p_disrupted", "mean"),
                        n_flights=("p_disrupted", "size"))
                   .reset_index())
    actual = (float(pd.to_numeric(scored["disrupted"], errors="coerce").mean())
              if "disrupted" in scored else None)
    return Prediction(
        as_of=pd.Timestamp(now),
        source=source,
        flights=tidy,
        predicted_rate=float(scored["p_hat"].mean()),
        n_flights=int(len(scored)),
        actual_rate=actual,
        by_hour=by_hour,
    )


# --------------------------------------------------------------------------- #
# Demo source: replay a real window from the joined parquet
# --------------------------------------------------------------------------- #


def predict_demo(now: pd.Timestamp | None = None,
                 *, joined: Path = DEFAULT_JOINED,
                 bundle: dict | None = None,
                 origin: str = "ORD") -> Prediction:
    now = pd.Timestamp(now or DEMO_DEFAULT_NOW)
    if now.tz is None:
        now = now.tz_localize("UTC")
    bundle = bundle or load_model_bundle()

    df = pd.read_parquet(joined)
    dep = pd.to_datetime(df["scheduled_dep_utc_hour"], utc=True)
    df["scheduled_dep_utc_hour"] = dep
    window = (dep >= now) & (dep < now + pd.Timedelta(hours=WINDOW_H))
    flights = df.loc[window].reset_index(drop=True)
    if not len(flights):
        raise ValueError(
            f"demo: no flights in [{now}, +{WINDOW_H}h]; pick a now within "
            f"{dep.min().date()}..{dep.max().date()}")
    logger.info("demo: %d flights in [%s, +%dh]", len(flights), now, WINDOW_H)

    scored = score(flights, now, bundle, completed_pool=df)
    tidy = _tidy(scored, now, origin)
    return _summarize(scored, tidy, now, source="replay")


# --------------------------------------------------------------------------- #
# Upcoming source: real next-48h dates, schedule projected from recent history
# --------------------------------------------------------------------------- #

# Columns copied verbatim from the template (real carrier / route / clock).
_SCHEDULE_COLS = [
    "reporting_airline", "flight_number_reporting_airline", "dest",
    "dest_state", "crs_dep_time", "crs_arr_time", "crs_elapsed_time",
    "distance",
]


def project_schedule_next_48h(now: pd.Timestamp, history: pd.DataFrame, *,
                              window_h: int = WINDOW_H,
                              origin_tz: str = "America/Chicago") -> pd.DataFrame:
    """Project ORD's recurring schedule onto the real upcoming window.

    There is no free, no-key source for the *future* ORD schedule, but ORD's
    operation is strongly weekly-periodic. For each ORD-local date the
    window touches, take the most recent historical date with the same ISO
    weekday and re-stamp its flights -- real carriers, flight numbers,
    routes, and local clock times -- onto that date. Times are rebuilt in
    America/Chicago and converted to UTC, yielding a realistic (not
    guaranteed) next-48 h schedule on genuine upcoming timestamps.
    """
    from src.clean_bts_ord import _hhmm_to_local_datetime

    now = pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("UTC")
    end = now + pd.Timedelta(hours=window_h)

    hist = history.copy()
    hist["flight_date"] = pd.to_datetime(hist["flight_date"])
    hist_iso = hist["flight_date"].dt.isocalendar().day  # 1=Mon..7=Sun
    templates = {iso: sub.max() for iso in range(1, 8)
                 if len(sub := hist.loc[hist_iso == iso, "flight_date"])}

    lo = now.tz_convert(origin_tz).normalize()
    hi = end.tz_convert(origin_tz).normalize()
    pieces = []
    for d in pd.date_range(lo, hi, freq="D"):
        iso = pd.Timestamp(d).isoweekday()
        tmpl_date = templates.get(iso)
        if tmpl_date is None:
            continue
        tmpl = hist.loc[hist["flight_date"] == tmpl_date, _SCHEDULE_COLS].copy()
        if not len(tmpl):
            continue
        D = pd.Timestamp(d.date())
        dep_local = _hhmm_to_local_datetime(
            pd.Series([D] * len(tmpl), index=tmpl.index),
            tmpl["crs_dep_time"], origin_tz)
        dep_utc = dep_local.dt.tz_convert("UTC")
        elapsed = pd.to_numeric(tmpl["crs_elapsed_time"], errors="coerce")
        tmpl["scheduled_dep_utc_hour"] = dep_utc.dt.floor("h")
        tmpl["scheduled_arr_utc_hour"] = (
            dep_utc + pd.to_timedelta(elapsed, unit="m")).dt.floor("h")
        tmpl["month"] = D.month
        tmpl["day_of_month"] = D.day
        tmpl["day_of_week"] = pd.Timestamp(D).isoweekday()
        tmpl["quarter"] = (D.month - 1) // 3 + 1
        pieces.append(tmpl)

    if not pieces:
        raise ValueError("could not project a schedule (empty history?)")
    proj = pd.concat(pieces, ignore_index=True)
    dep_all = pd.to_datetime(proj["scheduled_dep_utc_hour"], utc=True)
    proj = proj.loc[(dep_all >= now) & (dep_all < end)].reset_index(drop=True)
    return proj


def predict_upcoming(now: pd.Timestamp | None = None,
                     *, joined: Path = DEFAULT_JOINED,
                     bundle: dict | None = None, origin: str = "ORD",
                     weather: bool = True, weather_workers: int = 8) -> Prediction:
    """Predict the genuine next 48 h: projected schedule + live NWS forecast.

    The flight list is projected from ORD's recent recurring schedule
    (see :func:`project_schedule_next_48h`); weather is fetched live from
    NWS for the real upcoming hours. There is no ground truth (the window
    is in the future), so ``actual_rate`` is ``None``.
    """
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tz is None:
        now = now.tz_localize("UTC")
    bundle = bundle or load_model_bundle()

    history = pd.read_parquet(joined)
    history["scheduled_dep_utc_hour"] = pd.to_datetime(
        history["scheduled_dep_utc_hour"], utc=True)
    flights = project_schedule_next_48h(now, history, window_h=WINDOW_H)
    logger.info("upcoming: projected %d flights for [%s, +%dh]",
                len(flights), now, WINDOW_H)

    if weather:
        from src.live_flights import distinct_stations
        from src.live_weather import fetch_forecasts
        stations = distinct_stations(flights, origin=origin)
        logger.info("upcoming: fetching live NWS forecasts for %d airports",
                    len(stations))
        wx = fetch_forecasts(stations, max_workers=weather_workers)
        flights = _join_weather(flights, wx, origin=origin)

    # `now` is after the labelled data, so the as-of-now trailing window is
    # empty -- use the most recent known airport state instead.
    if "disrupted" not in history:
        history["disrupted"] = label_disrupted(history)
    hist_end = history["scheduled_dep_utc_hour"].max()
    tr = compute_trailing_rates_asof(hist_end, history)
    logger.info("upcoming: trailing snapshot as of %s: %s", hist_end,
                {w: round(r, 3) for w, r in tr.items()})

    scored = score(flights, now, bundle, trailing_rates=tr)
    tidy = _tidy(scored, now, origin)
    return _summarize(scored, tidy, now, source="upcoming")


# --------------------------------------------------------------------------- #
# Live source: AeroDataBox schedule + NWS forecast + recent completed
# --------------------------------------------------------------------------- #


def predict_live(api_key: str, now: pd.Timestamp | None = None,
                 *, bundle: dict | None = None,
                 origin: str = "ORD", lookback_h: int = 168,
                 recent_refresh_h: int | None = None,
                 joined: Path = DEFAULT_JOINED) -> Prediction:
    from src.live_flights import (
        DEFAULT_RECENT_CACHE_REFRESH_H,
        build_flight_features, distinct_stations, fetch_scheduled_departures,
        update_recent_departures_cache,
    )
    from src.live_weather import fetch_forecasts

    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tz is None:
        now = now.tz_localize("UTC")
    bundle = bundle or load_model_bundle()
    now_dt = now.to_pydatetime()

    # 1. forward schedule -> features
    norm = fetch_scheduled_departures(api_key, origin=origin,
                                      window_h=WINDOW_H, now=now_dt)
    flights = build_flight_features(norm, origin=origin)
    if not len(flights):
        raise ValueError("live: AeroDataBox returned no scheduled departures")

    # 2. weather forecast for ORD + every dest, joined on the hour
    wx = fetch_forecasts(distinct_stations(flights, origin=origin))
    flights = _join_weather(flights, wx, origin=origin)

    # 3. Operational state. Refresh a small rolling cache instead of pulling
    # the full 168h ORD history on every dashboard refresh. During cold start,
    # use cached rates only for windows the cache actually covers and fall back
    # to the latest local BTS snapshot for the longer windows.
    refresh_h = (DEFAULT_RECENT_CACHE_REFRESH_H if recent_refresh_h is None
                 else recent_refresh_h)
    recent_norm = update_recent_departures_cache(
        api_key, origin=origin, now=now_dt, refresh_h=refresh_h,
        retention_h=lookback_h)
    recent = build_flight_features(recent_norm, origin=origin)
    for c in ("cancelled", "diverted", "dep_del15", "arr_del15"):
        if c in recent_norm:
            recent[c] = recent_norm[c].to_numpy()
    if len(recent):
        recent["disrupted"] = label_disrupted(recent)

    history = pd.read_parquet(joined)
    history["scheduled_dep_utc_hour"] = pd.to_datetime(
        history["scheduled_dep_utc_hour"], utc=True)
    history["scheduled_arr_utc_hour"] = pd.to_datetime(
        history["scheduled_arr_utc_hour"], utc=True)
    if "disrupted" not in history:
        history["disrupted"] = label_disrupted(history)
    hist_end = history["scheduled_dep_utc_hour"].max()
    fallback_rates = compute_trailing_rates_asof(hist_end, history)

    trailing_rates = fallback_rates
    if len(recent):
        cache_rates = compute_trailing_rates_asof(now, recent)
        oldest_arr = pd.to_datetime(
            recent["scheduled_arr_utc_hour"], utc=True).min()
        coverage_h = max(0.0, (now - oldest_arr).total_seconds() / 3600.0)
        trailing_rates = {
            w: (cache_rates[w]
                if coverage_h >= w and not np.isnan(cache_rates[w])
                else fallback_rates[w])
            for w in fallback_rates
        }
        logger.info("live: recent cache coverage %.1fh; trailing rates: %s",
                    coverage_h, {w: round(r, 3)
                                 for w, r in trailing_rates.items()})
    else:
        logger.info("live: recent cache empty; using historical trailing "
                    "snapshot as of %s: %s", hist_end,
                    {w: round(r, 3) for w, r in trailing_rates.items()})

    scored = score(flights, now, bundle, trailing_rates=trailing_rates)
    tidy = _tidy(scored, now, origin)
    return _summarize(scored, tidy, now, source="live")


def _join_weather(flights: pd.DataFrame, wx: pd.DataFrame,
                  *, origin: str = "ORD") -> pd.DataFrame:
    """Attach ``origin_wx_*`` (ORD@dep hour) and ``dest_wx_*`` (dest@arr hour)."""
    wxcols = [c for c in wx.columns if c not in ("station", "valid_utc")]
    o = (wx.loc[wx["station"] == origin.upper(), ["valid_utc", *wxcols]]
           .rename(columns={"valid_utc": "_h",
                            **{c: f"origin_wx_{c}" for c in wxcols}}))
    out = flights.merge(o, left_on="scheduled_dep_utc_hour",
                        right_on="_h", how="left").drop(columns=["_h"])
    d = wx.rename(columns={"valid_utc": "_h", "station": "_st",
                           **{c: f"dest_wx_{c}" for c in wxcols}})
    out = out.merge(d, left_on=["dest", "scheduled_arr_utc_hour"],
                    right_on=["_st", "_h"], how="left").drop(
        columns=["_h", "_st"])
    return out


def predict_next_48h(source: str = "upcoming", *, api_key: str | None = None,
                     now: pd.Timestamp | None = None,
                     bundle: dict | None = None,
                     weather: bool = True) -> Prediction:
    """Dispatch to a prediction source. Returns a :class:`Prediction`.

    ``source``:
      * ``"upcoming"`` -- genuine next 48 h: schedule projected from ORD's
        recent history + live NWS forecast (no key, no ground truth).
      * ``"replay"`` -- replay a real labelled window (has ground truth).
      * ``"live"``    -- AeroDataBox schedule + NWS forecast (needs key).
    """
    if source == "live":
        api_key = resolve_api_key(api_key)
        if not api_key:
            raise ValueError(
                "live source requires an AeroDataBox API key "
                "(put it in aerodatabox.key, set $AERODATABOX_KEY, or "
                "pass api_key)")
        return predict_live(api_key, now=now, bundle=bundle)
    if source == "replay":
        return predict_demo(now=now, bundle=bundle)
    return predict_upcoming(now=now, bundle=bundle, weather=weather)
