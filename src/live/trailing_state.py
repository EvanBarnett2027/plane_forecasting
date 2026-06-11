"""As-of-now trailing disruption rate for the real-time dashboard.

This is the production counterpart to
:func:`src.training_pipeline.add_trailing_disruption_features`. It answers
the question that confuses people about this feature: *does it look into
the future?* It does **not**.

The key fact
------------
At a single forecast issue time, every flight in the next 48 h shares the
same ``t_now`` -- the issue time itself. In the backtest each flight's
``t_now = scheduled_dep - lead_origin_h``; for a real occasion issued at
``now``, ``lead_origin_h = scheduled_dep - now``, so ``t_now`` collapses
to ``now`` for *every* flight. The trailing rate is therefore a single
snapshot of how ORD has been running over the last
``{6, 24, 72, 168} h`` **up to now**, broadcast to all flights -- the
flight 2 h out and the flight 36 h out get the same value::

        [now-168h ........... now-6h)   |  now  |  --- next 48h flights ---
        \\___ realised, already landed __/    ^ t_now (issue time)

You never need to know whether flights between ``now`` and the 36 h
flight will be cancelled; those are in the future and are not in the
window. You only need the realised disruption status of flights that
**already completed** before ``now`` -- which a live flight-status feed
(e.g. AeroDataBox's FIDS departures-by-time-range endpoint) provides.

Leakage guard
-------------
The lookback window is indexed by **scheduled arrival** with a strict
``< now`` upper bound, identical to the training feature. That guarantees
every flight averaged into the rate has had time to realise its full
``disrupted`` outcome (which includes arrival-delay and diversion) before
``now``.

This module reuses :func:`src.training_pipeline._trailing_rates_vectorized`
so the serving math is *literally the same code* as training -- the
surest way to avoid train/serve skew.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.model.training_pipeline import (
    TRAILING_DISRUPTION_WINDOWS_H,
    _REFERENCE_DATE,
    _SECONDS_PER_HOUR,
    _series_to_int_microseconds,
    _trailing_rates_vectorized,
)

logger = logging.getLogger("trailing_state")


def label_disrupted(df: pd.DataFrame) -> pd.Series:
    """Reconstruct the BTS ``disrupted`` label from realised status.

    ``disrupted = dep_del15 OR arr_del15 OR cancelled OR diverted`` --
    identical to :func:`src.clean_bts_ord.add_labels`.
    """
    def col(name):
        if name in df:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0, index=df.index)

    disrupted = ((col("dep_del15") == 1) | (col("arr_del15") == 1)
                 | (col("cancelled") == 1) | (col("diverted") == 1))
    return disrupted.astype("int8")


def compute_trailing_rates_asof(
    now: datetime,
    completed: pd.DataFrame,
    *,
    windows_hours: tuple[int, ...] = TRAILING_DISRUPTION_WINDOWS_H,
) -> dict[int, float]:
    """Trailing disruption rate(s) as of ``now`` over completed flights.

    ``completed`` must have ``scheduled_arr_utc_hour`` (tz-aware UTC) and
    either ``disrupted`` or the raw status columns to derive it.

    Returns ``{window_hours: rate}``; a window with no completed flights
    yields ``nan``.
    """
    now = pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("UTC")

    pool = completed.copy()
    if "disrupted" not in pool:
        pool["disrupted"] = label_disrupted(pool)
    pool = pool.dropna(subset=["scheduled_arr_utc_hour", "disrupted"])
    pool = pool.sort_values("scheduled_arr_utc_hour")
    if not len(pool):
        return {w: float("nan") for w in windows_hours}

    arr_series = pool["scheduled_arr_utc_hour"]
    pool_arr_us = _series_to_int_microseconds(arr_series)
    pool_labels = pool["disrupted"].astype("int8").to_numpy()
    # Build the query instant with the SAME dtype as the pool so the
    # int64 conversion lands in the same time unit (microseconds here).
    query_series = pd.Series([now], dtype=arr_series.dtype)
    query = _series_to_int_microseconds(query_series)
    rates = _trailing_rates_vectorized(
        query, pool_arr_us, pool_labels, windows_hours)
    return {w: float(arr[0]) for w, arr in rates.items()}


def attach_operational_features(
    flights: pd.DataFrame,
    completed: pd.DataFrame,
    *,
    now: datetime | None = None,
    windows_hours: tuple[int, ...] = TRAILING_DISRUPTION_WINDOWS_H,
) -> pd.DataFrame:
    """Add ``trailing_disruption_rate_{W}h`` + ``days_since_2021`` columns.

    The trailing rates are a single as-of-``now`` snapshot broadcast to
    every flight (see module docstring). ``days_since_2021`` is the
    per-flight linear time trend the model also expects, computed from
    each flight's scheduled departure (matching
    :func:`src.training_pipeline.add_long_term_trend`).
    """
    now = now or datetime.now(timezone.utc)
    out = flights.copy()

    rates = compute_trailing_rates_asof(now, completed,
                                        windows_hours=windows_hours)
    for w, r in rates.items():
        out[f"trailing_disruption_rate_{w}h"] = r
    logger.info("trailing rates as of %s: %s", pd.Timestamp(now),
                {w: round(r, 4) for w, r in rates.items()})

    dep = pd.to_datetime(out["scheduled_dep_utc_hour"], utc=True)
    out["days_since_2021"] = (
        (dep - _REFERENCE_DATE).dt.total_seconds() / 86400.0).astype("float64")
    return out
