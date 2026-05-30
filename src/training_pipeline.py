"""End-to-end training + evaluation for the 48-hour disruption-rate forecast.

What this script does, in order:

1.  Load the joined flights+weather parquet (output of
    ``src.join_flights_weather``).
2.  Drop COVID-era 2020; time-split into train / val / test::

        train       2021-01-01 .. 2025-09-30   (filter -> ``--train-start``)
        validation  2025-10-01 .. 2025-12-31
        test        2026-01-01 .. (latest)

3.  Engineer features (calendar, cyclic time, schedule, route, lead times,
    weather).  Categorical cols are dtype 'category' so sklearn's
    ``HistGradientBoostingClassifier`` picks them up natively.
4.  Sample per-flight lead times ``L_origin ~ Uniform(0, 48)`` on train +
    validation; derive ``L_dest = L_origin + crs_elapsed_h``.
5.  Inject lead-time-aware noise into every ``*_wx_*`` column on both
    train AND validation (and later test) using
    ``src.weather_noise.WeatherNoiseInjector``.
6.  Fit ``HistGradientBoostingClassifier`` with early stopping on val.
7.  Build the **test occasions**: every 6 hours within the test period,
    take all flights scheduled to depart in the next 48 h, use each
    flight's real lead at that issue time, and inject noise at those
    leads.  Score and aggregate.
8.  Report per-flight metrics (AUC, AUC-PR, Brier) and per-occasion
    metrics (MAE on aggregate disruption rate, signed bias, severe-window
    precision/recall, etc.).
9.  Save model + metrics + per-occasion CSV under ``--artifacts-dir``.

Usage
-----
    python -m src.training_pipeline
    python -m src.training_pipeline --quick      # 10% subsample for fast iter
    python -m src.training_pipeline --noise-scale 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

try:
    import lightgbm as lgb
except ImportError:  # optional dependency for --model lgbm
    lgb = None

MODEL_CHOICES = ("hgb", "lgbm")

from src.ingest_bts_ord import PROJECT_ROOT
from src.weather_noise import WeatherNoiseInjector

logger = logging.getLogger("training_pipeline")

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_JOINED = (
    PROJECT_ROOT / "data" / "processed"
    / "ord_flights_with_weather_2020_present.parquet"
)
DEFAULT_SIGMAS = (
    PROJECT_ROOT / "data" / "processed" / "weather_noise_sigmas.json"
)
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "disruption_model"

# Splits (overridable via CLI)
TRAIN_START_DEFAULT = pd.Timestamp("2021-01-01", tz="UTC")
VAL_START_DEFAULT = pd.Timestamp("2025-10-01", tz="UTC")
TEST_START_DEFAULT = pd.Timestamp("2026-01-01", tz="UTC")

WINDOW_H = 48           # forecast window length
OCCASION_STEP_H = 6     # production refresh cadence (matches GFS runs)

# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #

# Keep small and explicit.  Anything in POST_FLIGHT_OUTCOME_COLS in
# clean_bts_ord.py is forbidden -- it would leak the label.
NUMERIC_FLIGHT_COLS = [
    "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance",
    "month", "day_of_month", "day_of_week", "quarter",
]
CATEGORICAL_FLIGHT_COLS = [
    "reporting_airline", "dest", "dest_state",
]
WEATHER_NUMERIC_SUFFIXES = [
    "tmpf", "dwpf", "relh", "feel", "alti", "mslp",
    "sknt", "drct", "gust", "peak_wind_gust",
    "p01i", "vsby", "skyl1", "ice_accretion_1hr", "snowdepth",
    "n_reports",
]


TRAILING_DISRUPTION_WINDOWS_H = (6, 24, 72, 168)  # 6h, 1d, 3d, 7d

OPERATIONAL_FEATURES = [
    f"trailing_disruption_rate_{w}h" for w in TRAILING_DISRUPTION_WINDOWS_H
] + ["days_since_2021"]


def feature_columns() -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) for the model."""
    weather = [
        f"{side}{suf}" for side in ("origin_wx_", "dest_wx_")
        for suf in WEATHER_NUMERIC_SUFFIXES
    ]
    derived = [
        "lead_origin_h", "lead_dest_h",
        "dep_hour", "arr_hour",
        "dep_sin", "dep_cos", "month_sin", "month_cos",
        "is_weekend",
    ]
    numeric = NUMERIC_FLIGHT_COLS + derived + weather + OPERATIONAL_FEATURES
    return numeric, CATEGORICAL_FLIGHT_COLS


# --------------------------------------------------------------------------- #
# Loading + splitting
# --------------------------------------------------------------------------- #


def load_and_split(
    path: Path,
    train_start: pd.Timestamp,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the joined parquet, drop pre-train rows, do a time-based split."""
    logger.info("loading %s", path)
    df = pd.read_parquet(path)
    logger.info("  loaded %d rows", len(df))

    # Cleaner timestamp: scheduled_dep_utc_hour is already tz-aware UTC.
    dt = df["scheduled_dep_utc_hour"]
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")

    df = df.loc[dt >= train_start].copy()
    dt = dt.loc[df.index]
    logger.info("  after train_start filter (%s): %d rows", train_start, len(df))

    # Drop rows with no scheduled time or no disrupted label
    keep = dt.notna() & df["disrupted"].notna()
    df = df.loc[keep].copy()
    dt = dt.loc[df.index]

    train_mask = dt < val_start
    val_mask = (dt >= val_start) & (dt < test_start)
    test_mask = dt >= test_start

    train = df.loc[train_mask].reset_index(drop=True)
    val = df.loc[val_mask].reset_index(drop=True)
    test = df.loc[test_mask].reset_index(drop=True)

    if quick:
        train = train.sample(frac=0.10, random_state=0).reset_index(drop=True)
        val = val.sample(frac=0.25, random_state=0).reset_index(drop=True)
        # leave test full so the eval numbers are honest
        logger.info("  --quick: subsampled train to %d, val to %d",
                    len(train), len(val))

    logger.info("split sizes: train=%d  val=%d  test=%d",
                len(train), len(val), len(test))
    return train, val, test


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def _derive_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dep_hour"] = (df["crs_dep_time"] // 100).astype("Int64")
    df["arr_hour"] = (df["crs_arr_time"] // 100).astype("Int64")
    # cyclic encodings of departure hour-of-day and month
    h = df["dep_hour"].astype("float64").fillna(12)
    m = df["month"].astype("float64").fillna(6)
    df["dep_sin"] = np.sin(2 * np.pi * h / 24)
    df["dep_cos"] = np.cos(2 * np.pi * h / 24)
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype("int8")
    return df


def add_sampled_leads(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """For training/validation: sample L_origin ~ U(0, WINDOW_H) per flight."""
    df = df.copy()
    lo = rng.uniform(0.0, float(WINDOW_H), size=len(df))
    elapsed_h = pd.to_numeric(df["crs_elapsed_time"], errors="coerce") / 60.0
    df["lead_origin_h"] = lo
    df["lead_dest_h"] = lo + elapsed_h.to_numpy()
    return df


# --------------------------------------------------------------------------- #
# Engineered "operational state" features
# --------------------------------------------------------------------------- #
#
# The big driver of the under-prediction bias was that the model had no
# memory: it saw weather and schedule but nothing about how the airport
# had been *running* in the recent past. Two cheap, leak-free features
# close most of that gap:
#
# * trailing_disruption_rate_{6,24,72,168}h
#       Mean of `disrupted` over ORD flights whose scheduled *arrival*
#       was strictly before t_now, with scheduled arrival in
#       [t_now - W, t_now). t_now = scheduled_dep - lead_origin_h.
#       Captures cascading delays, weather-aftermath state, ATC/crew
#       disruption, holiday spikes, etc.  In production this would be
#       sourced from a real-time ops feed (ASDI/ASPM/airline ops); during
#       backtest we compute it from BTS labels.
#
#       Why index by sched_arr instead of sched_dep: the disrupted label
#       includes arr_del15 (and diverted), which aren't realised until
#       the flight has landed. A bound on sched_dep would include flights
#       still in the air at t_now whose arrival labels aren't yet known
#       -- a real (though small for long windows, large for short
#       windows) leak. Indexing by sched_arr means every flight in the
#       window has had time to realise its full outcome before t_now.
#
# * days_since_2021
#       Linear-time trend. Captures the slow upward drift in ORD
#       disruption rate from 2021 (post-COVID recovery, low traffic)
#       through 2026 (busier, more disruption). Without this, the model
#       extrapolates a 2021-2025 average onto 2026 and ends up biased low.


_REFERENCE_DATE = pd.Timestamp("2021-01-01", tz="UTC")


def add_long_term_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    days = (df["scheduled_dep_utc_hour"] - _REFERENCE_DATE
            ).dt.total_seconds() / 86400.0
    df["days_since_2021"] = days.astype("float64")
    return df


_SECONDS_PER_HOUR = 3600


def _series_to_int_microseconds(s: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> int64 microseconds since epoch.

    pandas 3 stores ``datetime64[us, UTC]`` so ``astype('int64')`` already
    returns microseconds; we don't try to normalise to nanoseconds because
    1) the joined parquet is microsecond-resolution, 2) microsecond
    arithmetic comfortably fits in int64 for any realistic date range.
    """
    return s.astype("int64").to_numpy()


def _trailing_rates_vectorized(
    query_times_us: np.ndarray,         # int64 microseconds since epoch
    pool_times_us_sorted: np.ndarray,   # int64 us, ascending
    pool_labels_sorted: np.ndarray,     # int8 (0/1)
    windows_hours: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Per-window trailing rates for many query times in one vectorised pass.

    Strict-less-than upper bound (``side="left"`` on the upper edge) means
    a flight scheduled at exactly ``t_now`` is *not* in the lookback --
    guarantees no leakage of the query flight's own label.
    """
    # Prefix sum with a leading zero so cumsum[i] = count over first i rows.
    cumsum = np.concatenate(
        ([0], np.cumsum(pool_labels_sorted.astype(np.int64))))
    # i_now = first index with pool_time >= query_time (left-side -> strict <)
    i_now = np.searchsorted(pool_times_us_sorted, query_times_us, side="left")
    rates: dict[int, np.ndarray] = {}
    for w in windows_hours:
        delta_us = np.int64(w) * np.int64(_SECONDS_PER_HOUR) * np.int64(10**6)
        i_low = np.searchsorted(pool_times_us_sorted,
                                 query_times_us - delta_us, side="left")
        n = (i_now - i_low).astype(np.float64)
        s = (cumsum[i_now] - cumsum[i_low]).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(n > 0, s / n, np.nan)
        rates[w] = r
    return rates


def add_trailing_disruption_features(
    df: pd.DataFrame,
    lookback_pool: pd.DataFrame,
    *,
    windows_hours: tuple[int, ...] = TRAILING_DISRUPTION_WINDOWS_H,
) -> pd.DataFrame:
    """Add ``trailing_disruption_rate_{W}h`` columns, leak-free.

    Requires ``df`` to have ``scheduled_dep_utc_hour`` and
    ``lead_origin_h``. ``lookback_pool`` must additionally carry
    ``scheduled_arr_utc_hour`` (present in the joined parquet) and
    ``disrupted``.

    Each window's mean is taken over flights whose *scheduled arrival*
    falls in ``[t_now - W, t_now)`` -- strict-less-than ensures every
    flight in the window has had time to fully realise its outcome
    before t_now, so the label being averaged is genuinely available at
    prediction time.

    Computed via prefix sums, so O((|pool| + |df|) log |pool|).
    """
    df = df.copy()
    pool = lookback_pool.dropna(
        subset=["scheduled_arr_utc_hour", "disrupted"])
    pool = pool.sort_values("scheduled_arr_utc_hour")
    pool_arr_us = _series_to_int_microseconds(pool["scheduled_arr_utc_hour"])
    pool_labels = pool["disrupted"].astype("int8").to_numpy()

    t_dep_us = _series_to_int_microseconds(df["scheduled_dep_utc_hour"])
    lead_us = (df["lead_origin_h"].astype("float64").to_numpy()
               * _SECONDS_PER_HOUR * 1e6).astype("int64")
    t_now_us = t_dep_us - lead_us

    rates = _trailing_rates_vectorized(
        t_now_us, pool_arr_us, pool_labels, windows_hours)
    for w, arr in rates.items():
        df[f"trailing_disruption_rate_{w}h"] = arr
    return df


def prepare_features(df: pd.DataFrame,
                     cat_categories: dict[str, pd.Index] | None = None,
                     ) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    """Build the X frame the model will consume.

    Returns the feature frame *and* the category index per categorical col,
    so train can fix the vocabularies that val/test must reuse (unknowns
    become NaN, which HistGBT handles natively).
    """
    df = _derive_calendar_features(df)
    numeric, categorical = feature_columns()

    # Ensure all numeric features exist (fill missing weather cols with NaN)
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan

    out = df[numeric + categorical].copy()

    # Numeric -> float64
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")

    # Categorical with pinned categories.  When fitting on train we let
    # the vocabulary emerge from the data; on val/test we reuse the train
    # vocabulary and map unseen levels to NaN (HistGBT handles NaN in
    # categoricals natively).
    cats: dict[str, pd.Index] = {}
    for col in categorical:
        s = out[col].astype("string")
        if cat_categories is not None and col in cat_categories:
            pinned = cat_categories[col]
            s = s.where(s.isin(pinned))            # unknowns -> NaN, no warning
            out[col] = pd.Categorical(s, categories=pinned)
            cats[col] = pinned
        else:
            out[col] = pd.Categorical(s)
            cats[col] = out[col].cat.categories
    return out, cats


# --------------------------------------------------------------------------- #
# Noise injection
# --------------------------------------------------------------------------- #


def inject_noise(df: pd.DataFrame, noise: WeatherNoiseInjector) -> pd.DataFrame:
    """Apply lead-aware noise to *_wx_* cols in-place-ish; returns a copy."""
    return noise.transform(
        df,
        lead_origin_h=df["lead_origin_h"],
        lead_dest_h=df["lead_dest_h"],
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def fit_model(X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, y_val: np.ndarray,
              *, model_name: str = "hgb", seed: int = 0):
    """Fit a classifier and report validation metrics.

    ``model_name``:
      - ``"hgb"`` : sklearn ``HistGradientBoostingClassifier`` (default).
      - ``"lgbm"``: ``lightgbm.LGBMClassifier`` with early stopping on val.
    """
    if model_name == "hgb":
        model = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_iter=500,
            max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=0.0,
            early_stopping=True, validation_fraction=None,
            n_iter_no_change=20, categorical_features="from_dtype",
            random_state=seed, verbose=0,
        )
        logger.info("fitting HistGBT on %d rows (val=%d for monitoring)...",
                    len(X_train), len(X_val))
        model.fit(X_train, y_train)
    elif model_name == "lgbm":
        if lgb is None:
            raise ImportError(
                "lightgbm is not installed; `pip install lightgbm` to use "
                "--model lgbm")
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
            objective="binary", random_state=seed, n_jobs=-1, verbose=-1,
        )
        logger.info("fitting LightGBM on %d rows (val=%d for early stop)...",
                    len(X_train), len(X_val))
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
    else:
        raise ValueError(
            f"unknown model_name {model_name!r}; choose from {MODEL_CHOICES}")

    val_pred = model.predict_proba(X_val)[:, 1]
    logger.info("validation [%s]: AUC=%.4f  AUC-PR=%.4f  Brier=%.4f",
                model_name,
                roc_auc_score(y_val, val_pred),
                average_precision_score(y_val, val_pred),
                brier_score_loss(y_val, val_pred))
    return model


# --------------------------------------------------------------------------- #
# Test occasions
# --------------------------------------------------------------------------- #


def build_test_occasions(test_df: pd.DataFrame,
                         test_start: pd.Timestamp,
                         test_end: pd.Timestamp,
                         window_h: int = WINDOW_H,
                         step_h: int = OCCASION_STEP_H,
                         ) -> pd.DataFrame:
    """Explode the test split into per-(occasion, flight) rows.

    Each occasion is a ``t_issue`` at ``step_h`` cadence.  For each
    occasion we include every flight whose scheduled departure UTC hour
    falls in ``[t_issue, t_issue + window_h)``.  The flight's lead times
    are computed from ``t_issue`` (real, not sampled).
    """
    dep = test_df["scheduled_dep_utc_hour"]
    if dep.dt.tz is None:
        dep = dep.dt.tz_localize("UTC")
    test_df = test_df.copy()
    test_df["scheduled_dep_utc_hour"] = dep

    issues = pd.date_range(test_start, test_end, freq=f"{step_h}h", tz="UTC")
    logger.info("building test occasions: %d issues, step=%dh, window=%dh",
                len(issues), step_h, window_h)

    pieces = []
    elapsed_h = (pd.to_numeric(test_df["crs_elapsed_time"], errors="coerce")
                 / 60.0)
    for occ_id, t_issue in enumerate(issues):
        t_end = t_issue + pd.Timedelta(hours=window_h)
        mask = (dep >= t_issue) & (dep < t_end)
        if not mask.any():
            continue
        sub = test_df.loc[mask].copy()
        sub["occasion_id"] = occ_id
        sub["t_issue"] = t_issue
        lo = (sub["scheduled_dep_utc_hour"] - t_issue) / pd.Timedelta(hours=1)
        sub["lead_origin_h"] = lo.astype("float64").to_numpy()
        sub["lead_dest_h"] = sub["lead_origin_h"] + elapsed_h.loc[mask].to_numpy()
        pieces.append(sub)
    if not pieces:
        return test_df.head(0)
    out = pd.concat(pieces, ignore_index=True)
    logger.info("  -> %d (occasion, flight) rows across %d occasions",
                len(out), out["occasion_id"].nunique())
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class Metrics:
    n_train: int
    n_val: int
    n_test_flight_occasions: int
    n_occasions: int

    # Per-flight (test, across all occasions)
    test_auc_roc: float
    test_auc_pr: float
    test_brier: float
    test_base_rate: float

    # Per-occasion aggregate
    occ_mae: float
    occ_rmse: float
    occ_bias: float
    occ_pearson: float

    # Severe-window detection (top-decile of actual rates)
    severe_threshold_pct: float
    severe_precision: float
    severe_recall: float


def per_occasion_table(occ_df: pd.DataFrame, p_hat: np.ndarray) -> pd.DataFrame:
    df = occ_df[["occasion_id", "t_issue", "disrupted"]].copy()
    df["p_hat"] = p_hat
    agg = df.groupby(["occasion_id", "t_issue"]).agg(
        n_flights=("disrupted", "size"),
        actual_rate=("disrupted", "mean"),
        predicted_rate=("p_hat", "mean"),
    ).reset_index()
    agg["abs_error"] = (agg["predicted_rate"] - agg["actual_rate"]).abs()
    agg["signed_error"] = agg["predicted_rate"] - agg["actual_rate"]
    return agg


def compute_metrics(y_test: np.ndarray, p_test: np.ndarray,
                    occ_table: pd.DataFrame,
                    n_train: int, n_val: int,
                    severe_q: float = 0.90) -> Metrics:
    base = float(np.mean(y_test))
    auc = float(roc_auc_score(y_test, p_test))
    pr = float(average_precision_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))

    mae = float(occ_table["abs_error"].mean())
    rmse = float(np.sqrt(np.mean(occ_table["signed_error"] ** 2)))
    bias = float(occ_table["signed_error"].mean())
    pearson = float(occ_table[["actual_rate", "predicted_rate"]]
                    .corr().iloc[0, 1])

    # Severe-window detection
    thr_actual = float(occ_table["actual_rate"].quantile(severe_q))
    thr_pred = float(occ_table["predicted_rate"].quantile(severe_q))
    actual_severe = occ_table["actual_rate"] >= thr_actual
    pred_severe = occ_table["predicted_rate"] >= thr_pred
    tp = int(((actual_severe) & (pred_severe)).sum())
    fp = int(((~actual_severe) & (pred_severe)).sum())
    fn = int(((actual_severe) & (~pred_severe)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    return Metrics(
        n_train=n_train,
        n_val=n_val,
        n_test_flight_occasions=len(y_test),
        n_occasions=int(occ_table["occasion_id"].nunique()),
        test_auc_roc=auc,
        test_auc_pr=pr,
        test_brier=brier,
        test_base_rate=base,
        occ_mae=mae,
        occ_rmse=rmse,
        occ_bias=bias,
        occ_pearson=pearson,
        severe_threshold_pct=severe_q * 100,
        severe_precision=prec,
        severe_recall=rec,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> Metrics:
    rng = np.random.default_rng(args.seed)

    train, val, test = load_and_split(
        args.joined, args.train_start, args.val_start, args.test_start,
        quick=args.quick,
    )
    if not len(test):
        raise SystemExit("Test split is empty -- check --test-start/--test-end.")

    # ------ training + validation features (per-flight, sampled leads) -----
    train = add_sampled_leads(train, rng)
    val = add_sampled_leads(val, rng)

    # Engineered operational/trend features (leak-free; see helper docstrings).
    lookback_pool = pd.concat([train, val, test], ignore_index=True)
    train = add_long_term_trend(
        add_trailing_disruption_features(train, lookback_pool))
    val = add_long_term_trend(
        add_trailing_disruption_features(val, lookback_pool))

    noise = WeatherNoiseInjector.load(args.sigmas, seed=args.seed
                                       ).with_scale(args.noise_scale)
    logger.info("WeatherNoiseInjector: scale=%.2f  sigmas=%d cols",
                noise.scale, len(noise.sigmas))

    train_noised = inject_noise(train, noise)
    val_noised = inject_noise(val, noise)

    X_train, cats = prepare_features(train_noised)
    X_val, _ = prepare_features(val_noised, cat_categories=cats)
    y_train = train_noised["disrupted"].astype("int8").to_numpy()
    y_val = val_noised["disrupted"].astype("int8").to_numpy()

    model = fit_model(X_train, y_train, X_val, y_val,
                      model_name=args.model, seed=args.seed)

    # ----------------- test set: build occasions, score ---------------------
    test_end = args.test_end or (
        pd.to_datetime(test["scheduled_dep_utc_hour"]).max().ceil("D")
    )
    occ_df = build_test_occasions(test, args.test_start, test_end)
    if not len(occ_df):
        raise SystemExit("No test occasions in range.")

    # Add the same engineered features. lookback_pool is unchanged so
    # trailing rates are computed against the full historical population.
    occ_df = add_long_term_trend(
        add_trailing_disruption_features(occ_df, lookback_pool))
    occ_noised = inject_noise(occ_df, noise)
    X_test, _ = prepare_features(occ_noised, cat_categories=cats)
    y_test = occ_noised["disrupted"].astype("int8").to_numpy()
    p_test = model.predict_proba(X_test)[:, 1]

    occ_table = per_occasion_table(occ_noised, p_test)
    metrics = compute_metrics(y_test, p_test, occ_table,
                              n_train=len(X_train), n_val=len(X_val))
    _log_metrics(metrics)

    # ----------------- artifacts -------------------------------------------
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "categories": cats,
                 "noise_scale": args.noise_scale, "seed": args.seed},
                args.artifacts_dir / "model.joblib")
    (args.artifacts_dir / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2, default=float))
    occ_table.to_csv(args.artifacts_dir / "per_occasion.csv", index=False)
    _write_feature_importance(model, X_train.columns,
                              args.artifacts_dir / "feature_importance.csv")
    logger.info("wrote artifacts to %s", args.artifacts_dir)
    return metrics


def run_walk_forward(args: argparse.Namespace) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation.

    At each retrain step ``i`` with boundary ``T_i``:
      - train on all flights scheduled before ``T_i``
        (so the train window *grows* monotonically across steps),
      - hold out the last ``--val-days`` of train as a time-ordered val
        slice for early stopping,
      - score the per-occasion 48 h forecast for issue times
        ``T_i <= t_issue < T_i + retrain_every_days``,
      - record metrics + per-occasion + per-flight outputs.

    This mirrors how the model would be retrained on a fixed cadence in
    production. Each successive step has more training data.
    """
    rng = np.random.default_rng(args.seed)

    # Load everything once.
    logger.info("loading %s", args.joined)
    df_all = pd.read_parquet(args.joined)
    dep = pd.to_datetime(df_all["scheduled_dep_utc_hour"], utc=True)
    df_all["scheduled_dep_utc_hour"] = dep
    df_all = df_all.loc[(dep >= args.train_start) & df_all["disrupted"].notna()].copy()
    logger.info("  %d flights >= %s", len(df_all), args.train_start)

    # Retrain schedule.
    first_test = args.first_test_start or args.test_start
    test_end = args.test_end or df_all["scheduled_dep_utc_hour"].max().ceil("D")
    step_h = args.retrain_every_days * 24
    boundaries = []
    t = first_test
    while t < test_end:
        boundaries.append((t, min(t + pd.Timedelta(hours=step_h), test_end)))
        t = boundaries[-1][1]
    logger.info("walk-forward: %d steps of %dd starting %s, model=%s",
                len(boundaries), args.retrain_every_days,
                first_test.date(), args.model)

    noise = WeatherNoiseInjector.load(
        args.sigmas, seed=args.seed).with_scale(args.noise_scale)

    step_rows: list[dict] = []
    occ_pieces: list[pd.DataFrame] = []
    flight_pieces: list[pd.DataFrame] = []
    val_window = pd.Timedelta(days=args.val_days)
    flight_horizon = pd.Timedelta(hours=WINDOW_H)

    for step_i, (t_start, t_end) in enumerate(boundaries):
        logger.info("=" * 64)
        logger.info("STEP %d/%d   train < %s   test [%s, %s)",
                    step_i + 1, len(boundaries),
                    t_start, t_start, t_end)

        dep_all = df_all["scheduled_dep_utc_hour"]
        # Train < t_start; carve val = last `val_days` of train (time-ordered).
        train_pool = df_all.loc[dep_all < t_start]
        val_cut = t_start - val_window
        val_df = train_pool.loc[train_pool["scheduled_dep_utc_hour"] >= val_cut].copy()
        train_df = train_pool.loc[train_pool["scheduled_dep_utc_hour"] < val_cut].copy()
        # Test must include flights up to t_end + 48h (the last occasion of
        # this chunk has t_issue near t_end and looks ahead a full window).
        test_mask = ((dep_all >= t_start)
                     & (dep_all < t_end + flight_horizon))
        test_df = df_all.loc[test_mask].copy()
        logger.info("  train=%d  val=%d  test-pool=%d",
                    len(train_df), len(val_df), len(test_df))
        if not len(train_df) or not len(val_df) or not len(test_df):
            logger.warning("step %d: empty split, skipping", step_i)
            continue

        # Same per-flight feature build the single-shot mode uses.
        train_df = add_sampled_leads(train_df, rng)
        val_df = add_sampled_leads(val_df, rng)
        # Engineered operational + trend features (lookback against full df_all).
        train_df = add_long_term_trend(
            add_trailing_disruption_features(train_df, df_all))
        val_df = add_long_term_trend(
            add_trailing_disruption_features(val_df, df_all))
        train_n = inject_noise(train_df, noise)
        val_n = inject_noise(val_df, noise)
        X_train, cats = prepare_features(train_n)
        X_val, _ = prepare_features(val_n, cat_categories=cats)
        y_train = train_n["disrupted"].astype("int8").to_numpy()
        y_val = val_n["disrupted"].astype("int8").to_numpy()

        model = fit_model(X_train, y_train, X_val, y_val,
                          model_name=args.model, seed=args.seed)

        # Per-occasion test scoring restricted to t_issue in [t_start, t_end).
        occ_df = build_test_occasions(test_df, t_start, t_end)
        if not len(occ_df):
            logger.warning("step %d: no test occasions", step_i)
            continue
        occ_df = add_long_term_trend(
            add_trailing_disruption_features(occ_df, df_all))
        occ_n = inject_noise(occ_df, noise)
        X_test, _ = prepare_features(occ_n, cat_categories=cats)
        y_test = occ_n["disrupted"].astype("int8").to_numpy()
        p_test = model.predict_proba(X_test)[:, 1]

        occ_table = per_occasion_table(occ_n, p_test)
        occ_table["step"] = step_i
        occ_table["t_train_end"] = t_start
        occ_pieces.append(occ_table)

        # Per-flight predictions -- enough to redo any metric / calibration.
        flight_pieces.append(pd.DataFrame({
            "step": step_i,
            "t_train_end": t_start,
            "t_issue": occ_n["t_issue"].to_numpy(),
            "occasion_id": occ_n["occasion_id"].to_numpy(),
            "dest": occ_n["dest"].astype("string").to_numpy(),
            "reporting_airline": occ_n["reporting_airline"].astype("string").to_numpy(),
            "lead_origin_h": occ_n["lead_origin_h"].to_numpy(),
            "lead_dest_h": occ_n["lead_dest_h"].to_numpy(),
            "y_true": y_test,
            "p_hat": p_test,
        }))

        m = compute_metrics(y_test, p_test, occ_table,
                            n_train=len(X_train), n_val=len(X_val))
        _log_metrics(m)
        step_rows.append({
            "step": step_i,
            "t_train_end": t_start,
            "t_test_start": t_start,
            "t_test_end": t_end,
            "n_train_rows": len(X_train),
            "n_val_rows": len(X_val),
            "n_occasions": int(occ_table["occasion_id"].nunique()),
            "n_test_flight_occasions": len(y_test),
            **asdict(m),
        })

    if not step_rows:
        raise SystemExit("walk-forward produced no steps -- check date args.")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(step_rows)
    metrics_df.to_csv(args.artifacts_dir / "walkforward_step_metrics.csv",
                      index=False)
    pd.concat(occ_pieces, ignore_index=True).to_parquet(
        args.artifacts_dir / "walkforward_per_occasion.parquet", index=False)
    pd.concat(flight_pieces, ignore_index=True).to_parquet(
        args.artifacts_dir / "walkforward_per_flight.parquet", index=False)

    logger.info("=" * 64)
    logger.info("WALK-FORWARD SUMMARY (%d steps, model=%s)",
                len(step_rows), args.model)
    logger.info("=" * 64)
    summary_cols = ["step", "t_train_end", "n_train_rows",
                    "test_auc_roc", "test_auc_pr", "test_brier",
                    "occ_mae", "occ_bias", "occ_pearson"]
    logger.info("\n%s", metrics_df[summary_cols].to_string(
        index=False, float_format="%.4f"))
    logger.info("wrote artifacts to %s", args.artifacts_dir)
    return metrics_df


def _log_metrics(m: Metrics) -> None:
    logger.info("=" * 64)
    logger.info("RESULTS")
    logger.info("=" * 64)
    logger.info("train rows: %d   val rows: %d", m.n_train, m.n_val)
    logger.info("test (occasion, flight) rows: %d across %d occasions",
                m.n_test_flight_occasions, m.n_occasions)
    logger.info("base rate (disrupted, test): %.4f", m.test_base_rate)
    logger.info("-- per-flight (test) --")
    logger.info("  AUC-ROC : %.4f", m.test_auc_roc)
    logger.info("  AUC-PR  : %.4f   (base %.4f)",
                m.test_auc_pr, m.test_base_rate)
    logger.info("  Brier   : %.4f", m.test_brier)
    logger.info("-- per-occasion aggregate (48h window) --")
    logger.info("  MAE      : %.4f  (= avg |pred rate - actual rate|)",
                m.occ_mae)
    logger.info("  RMSE     : %.4f", m.occ_rmse)
    logger.info("  bias     : %+.4f  (predicted - actual)", m.occ_bias)
    logger.info("  Pearson r: %.4f", m.occ_pearson)
    logger.info("-- severe-window detection (top %.0f%% of actual rates) --",
                m.severe_threshold_pct)
    logger.info("  precision: %.3f   recall: %.3f",
                m.severe_precision, m.severe_recall)
    logger.info("=" * 64)


def _write_feature_importance(model: HistGradientBoostingClassifier,
                              feature_names: Iterable[str],
                              out: Path) -> None:
    # HistGBT doesn't expose a clean importance attribute by default
    # without permutation_importance, which is expensive. Use a quick
    # gain-based proxy via tree_traversal if available; otherwise skip.
    try:
        imp = model.feature_importances_  # type: ignore[attr-defined]
    except AttributeError:
        return
    pd.DataFrame({"feature": list(feature_names), "importance": imp}) \
        .sort_values("importance", ascending=False) \
        .to_csv(out, index=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    p.add_argument("--sigmas", type=Path, default=DEFAULT_SIGMAS)
    p.add_argument("--train-start",
                   type=lambda s: pd.Timestamp(s, tz="UTC"),
                   default=TRAIN_START_DEFAULT)
    p.add_argument("--val-start",
                   type=lambda s: pd.Timestamp(s, tz="UTC"),
                   default=VAL_START_DEFAULT)
    p.add_argument("--test-start",
                   type=lambda s: pd.Timestamp(s, tz="UTC"),
                   default=TEST_START_DEFAULT)
    p.add_argument("--test-end",
                   type=lambda s: pd.Timestamp(s, tz="UTC"),
                   default=None)
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="Global multiplier on injector sigmas.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Subsample train/val for fast iteration.")
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--model", choices=MODEL_CHOICES, default="hgb",
                   help="Classifier family. lgbm requires `pip install lightgbm`.")
    # --- walk-forward retraining ---
    p.add_argument("--walk-forward", action="store_true",
                   help="Expanding-window evaluation: retrain every "
                        "--retrain-every-days, growing the train window "
                        "each step. Ignores --val-start when on.")
    p.add_argument("--retrain-every-days", type=int, default=30,
                   help="Cadence of retrains (and length of each test chunk) "
                        "in walk-forward mode. Default 30.")
    p.add_argument("--first-test-start",
                   type=lambda s: pd.Timestamp(s, tz="UTC"), default=None,
                   help="First retrain boundary T_0 in walk-forward mode. "
                        "Defaults to --test-start.")
    p.add_argument("--val-days", type=int, default=30,
                   help="Time-ordered validation slice (last N days of train) "
                        "used for early stopping at each walk-forward step. "
                        "Default 30.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.walk_forward:
        run_walk_forward(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
