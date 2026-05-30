"""Full-pipeline validation of CatBoost + damped local isotonic.

Reproduces the §15 experiment from
``notebooks/lightgbm_investigation.ipynb`` but on the **full** training
set (not the 10% subsample) so we can confirm the empirical findings
hold at production scale.

What this does
--------------
1. Load + split flights (same as `training_pipeline.run`), full train.
2. Sample lead times, add operational/trend features, inject noise.
3. Fit four stacks on the full pipeline:
   - LightGBM on full train (baseline, no calibration carve-out).
   - CatBoost on full train (raw).
   - LightGBM on `train_core` (full train minus last 14 days).
   - CatBoost on `train_core`.
4. For the two core-trained variants, fit isotonic on the 14-day
   calibration slice and sweep `alpha in [0, 0.25, 0.5, 0.75, 1.0]`.
5. Score per-occasion on the test set and dump:
   - ``artifacts/disruption_model/full_calibration_validation.json``
   - ``artifacts/disruption_model/full_calibration_validation.parquet``
     (per-(stack, alpha, occasion) rows, for the notebook to plot).

Usage
-----
    .venv/bin/python -m src.validate_calibration_full

Or with a few rough timing knobs:
    .venv/bin/python -m src.validate_calibration_full \\
        --catboost-iters 800 --catboost-depth 7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

import lightgbm as lgb
from catboost import CatBoostClassifier

from src.ingest_bts_ord import PROJECT_ROOT
from src.training_pipeline import (
    DEFAULT_JOINED,
    DEFAULT_SIGMAS,
    TRAIN_START_DEFAULT,
    VAL_START_DEFAULT,
    TEST_START_DEFAULT,
    WINDOW_H,
    OCCASION_STEP_H,
    add_long_term_trend,
    add_sampled_leads,
    add_trailing_disruption_features,
    build_test_occasions,
    inject_noise,
    load_and_split,
    per_occasion_table,
    prepare_features,
)
from src.weather_noise import WeatherNoiseInjector

logger = logging.getLogger("validate_calibration_full")

ARTIFACTS = PROJECT_ROOT / "artifacts" / "disruption_model"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _occ_metrics(y_true: np.ndarray,
                 p_hat: np.ndarray,
                 occ_n: pd.DataFrame) -> dict:
    t = per_occasion_table(occ_n, p_hat)
    return {
        "AUC":          float(roc_auc_score(y_true, p_hat)),
        "AUC_PR":       float(average_precision_score(y_true, p_hat)),
        "Brier":        float(brier_score_loss(y_true, p_hat)),
        "occ_MAE":      float(t["abs_error"].mean()),
        "occ_RMSE":     float(np.sqrt((t["signed_error"] ** 2).mean())),
        "occ_bias":     float(t["signed_error"].mean()),
        "occ_pearson":  float(
            t[["actual_rate", "predicted_rate"]].corr().iloc[0, 1]),
        "mean_p":       float(p_hat.mean()),
        "n_test_rows":  int(len(y_true)),
        "n_occasions":  int(t["occasion_id"].nunique()),
    }


def _per_occasion_rows(stack: str, alpha: float,
                       y_true: np.ndarray, p_hat: np.ndarray,
                       occ_n: pd.DataFrame) -> pd.DataFrame:
    t = per_occasion_table(occ_n, p_hat)
    t["stack"] = stack
    t["alpha"] = float(alpha)
    return t


def _for_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__missing__").astype("string")
    return X


def _fit_lgbm(X_tr, y_tr, X_va, y_va, *, seed: int = 42) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
        objective="binary", random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(20, verbose=False)])
    return m


def _fit_catboost(X_tr, y_tr, X_va, y_va, cat_cols: list[str],
                  *, seed: int = 42, iters: int = 1500,
                  depth: int = 8) -> CatBoostClassifier:
    m = CatBoostClassifier(
        iterations=iters, learning_rate=0.05, depth=depth,
        l2_leaf_reg=3.0, bagging_temperature=1.0,
        cat_features=cat_cols, random_seed=seed, verbose=False,
        early_stopping_rounds=30, eval_metric="Logloss",
    )
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return m


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    p.add_argument("--sigmas", type=Path, default=DEFAULT_SIGMAS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-days", type=int, default=14,
                   help="Days of train carved off as calibration slice.")
    p.add_argument("--catboost-iters", type=int, default=1500)
    p.add_argument("--catboost-depth", type=int, default=8)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    # ---- 1. Load + split + feature build (full pipeline) -------------
    train, val, test = load_and_split(
        args.joined, TRAIN_START_DEFAULT, VAL_START_DEFAULT,
        TEST_START_DEFAULT, quick=False)

    train = add_sampled_leads(train, rng)
    val = add_sampled_leads(val, rng)
    lookback_pool = pd.concat([train, val, test], ignore_index=True)
    train = add_long_term_trend(
        add_trailing_disruption_features(train, lookback_pool))
    val = add_long_term_trend(
        add_trailing_disruption_features(val, lookback_pool))

    noise = WeatherNoiseInjector.load(args.sigmas, seed=args.seed
                                       ).with_scale(1.0)
    train_n = inject_noise(train, noise)
    val_n = inject_noise(val, noise)

    X_train, cats = prepare_features(train_n)
    X_val, _ = prepare_features(val_n, cat_categories=cats)
    y_train = train_n["disrupted"].astype("int8").to_numpy()
    y_val = val_n["disrupted"].astype("int8").to_numpy()

    test_end = pd.to_datetime(test["scheduled_dep_utc_hour"]).max().ceil("D")
    occ = build_test_occasions(test, TEST_START_DEFAULT, test_end)
    occ = add_long_term_trend(
        add_trailing_disruption_features(occ, lookback_pool))
    occ_n = inject_noise(occ, noise)
    X_test, _ = prepare_features(occ_n, cat_categories=cats)
    y_test = occ_n["disrupted"].astype("int8").to_numpy()
    logger.info("test occasions: %d rows, %d occasions",
                len(y_test), occ_n["occasion_id"].nunique())

    cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()
    X_train_cb = _for_catboost(X_train, cat_cols)
    X_val_cb = _for_catboost(X_val, cat_cols)
    X_test_cb = _for_catboost(X_test, cat_cols)

    results: dict[str, dict] = {}
    occ_rows: list[pd.DataFrame] = []
    # Per-flight predictions for downstream calibration plots.
    # Only persist the four base prediction vectors; damped blends are
    # linear interpolations the notebook computes on the fly.
    per_flight: dict[str, np.ndarray] = {"y_test": None}

    # ---- 2. LightGBM raw (full train) --------------------------------
    logger.info("=" * 64)
    logger.info("[1/4] LightGBM raw on FULL train")
    t0 = time.perf_counter()
    lgbm_full = _fit_lgbm(X_train, y_train, X_val, y_val, seed=args.seed)
    p_test_lgbm = lgbm_full.predict_proba(X_test)[:, 1]
    results["LightGBM raw (full train)"] = _occ_metrics(
        y_test, p_test_lgbm, occ_n)
    occ_rows.append(_per_occasion_rows(
        "LightGBM raw (full train)", 0.0, y_test, p_test_lgbm, occ_n))
    per_flight["y_test"] = y_test.astype("int8")
    per_flight["p_lgbm_full"] = p_test_lgbm.astype("float32")
    logger.info("  done in %.1fs   MAE=%.4f   bias=%+.4f",
                time.perf_counter() - t0,
                results["LightGBM raw (full train)"]["occ_MAE"],
                results["LightGBM raw (full train)"]["occ_bias"])

    # ---- 3. CatBoost raw (full train) --------------------------------
    logger.info("=" * 64)
    logger.info("[2/4] CatBoost raw on FULL train (iters=%d, depth=%d)",
                args.catboost_iters, args.catboost_depth)
    t0 = time.perf_counter()
    catb_full = _fit_catboost(X_train_cb, y_train, X_val_cb, y_val, cat_cols,
                              seed=args.seed, iters=args.catboost_iters,
                              depth=args.catboost_depth)
    p_test_catb = catb_full.predict_proba(X_test_cb)[:, 1]
    results["CatBoost raw (full train)"] = _occ_metrics(
        y_test, p_test_catb, occ_n)
    occ_rows.append(_per_occasion_rows(
        "CatBoost raw (full train)", 0.0, y_test, p_test_catb, occ_n))
    per_flight["p_catb_full"] = p_test_catb.astype("float32")
    logger.info("  done in %.1fs   MAE=%.4f   bias=%+.4f",
                time.perf_counter() - t0,
                results["CatBoost raw (full train)"]["occ_MAE"],
                results["CatBoost raw (full train)"]["occ_bias"])

    # ---- 4. Carve last 14 days off train for calibration -------------
    train_sorted = train_n.sort_values("scheduled_dep_utc_hour").reset_index(drop=True)
    cut = train_sorted["scheduled_dep_utc_hour"].max() - pd.Timedelta(
        days=args.calib_days)
    calib_mask = train_sorted["scheduled_dep_utc_hour"] >= cut
    train_core = train_sorted.loc[~calib_mask].reset_index(drop=True)
    train_calib = train_sorted.loc[calib_mask].reset_index(drop=True)
    logger.info("train_core=%d  train_calib=%d (%s -> %s)",
                len(train_core), len(train_calib),
                train_calib["scheduled_dep_utc_hour"].min(),
                train_calib["scheduled_dep_utc_hour"].max())

    X_core, cats_core = prepare_features(train_core)
    X_calib, _ = prepare_features(train_calib, cat_categories=cats_core)
    X_val_c, _ = prepare_features(val_n, cat_categories=cats_core)
    X_test_c, _ = prepare_features(occ_n, cat_categories=cats_core)
    y_core = train_core["disrupted"].astype("int8").to_numpy()
    y_calib = train_calib["disrupted"].astype("int8").to_numpy()

    X_core_cb = _for_catboost(X_core, cat_cols)
    X_calib_cb = _for_catboost(X_calib, cat_cols)
    X_val_c_cb = _for_catboost(X_val_c, cat_cols)
    X_test_c_cb = _for_catboost(X_test_c, cat_cols)

    # ---- 5. LightGBM core + damped iso sweep --------------------------
    logger.info("=" * 64)
    logger.info("[3/4] LightGBM on train_core + damped iso sweep")
    t0 = time.perf_counter()
    lgbm_core = _fit_lgbm(X_core, y_core, X_val_c, y_val, seed=args.seed)
    p_calib_lgbm = lgbm_core.predict_proba(X_calib)[:, 1]
    p_test_core_lgbm = lgbm_core.predict_proba(X_test_c)[:, 1]
    iso_lgbm = IsotonicRegression(out_of_bounds="clip").fit(
        p_calib_lgbm, y_calib)
    p_test_iso_lgbm = iso_lgbm.transform(p_test_core_lgbm)
    per_flight["p_lgbm_core_raw"] = p_test_core_lgbm.astype("float32")
    per_flight["p_lgbm_core_iso"] = p_test_iso_lgbm.astype("float32")
    for a in ALPHAS:
        p_corr = a * p_test_iso_lgbm + (1.0 - a) * p_test_core_lgbm
        label = f"LightGBM core + iso α={a:.2f}"
        results[label] = _occ_metrics(y_test, p_corr, occ_n)
        occ_rows.append(_per_occasion_rows(
            "LightGBM core + iso", a, y_test, p_corr, occ_n))
    logger.info("  done in %.1fs", time.perf_counter() - t0)

    # ---- 6. CatBoost core + damped iso sweep --------------------------
    logger.info("=" * 64)
    logger.info("[4/4] CatBoost on train_core + damped iso sweep")
    t0 = time.perf_counter()
    catb_core = _fit_catboost(X_core_cb, y_core, X_val_c_cb, y_val, cat_cols,
                              seed=args.seed, iters=args.catboost_iters,
                              depth=args.catboost_depth)
    p_calib_catb = catb_core.predict_proba(X_calib_cb)[:, 1]
    p_test_core_catb = catb_core.predict_proba(X_test_c_cb)[:, 1]
    iso_catb = IsotonicRegression(out_of_bounds="clip").fit(
        p_calib_catb, y_calib)
    p_test_iso_catb = iso_catb.transform(p_test_core_catb)
    per_flight["p_catb_core_raw"] = p_test_core_catb.astype("float32")
    per_flight["p_catb_core_iso"] = p_test_iso_catb.astype("float32")
    for a in ALPHAS:
        p_corr = a * p_test_iso_catb + (1.0 - a) * p_test_core_catb
        label = f"CatBoost core + iso α={a:.2f}"
        results[label] = _occ_metrics(y_test, p_corr, occ_n)
        occ_rows.append(_per_occasion_rows(
            "CatBoost core + iso", a, y_test, p_corr, occ_n))
    logger.info("  done in %.1fs", time.perf_counter() - t0)

    # ---- 7. Persist artifacts -----------------------------------------
    json_path = args.artifacts_dir / "full_calibration_validation.json"
    parquet_path = args.artifacts_dir / "full_calibration_validation.parquet"
    json_path.write_text(json.dumps({
        "config": {
            "calib_days": args.calib_days,
            "catboost_iters": args.catboost_iters,
            "catboost_depth": args.catboost_depth,
            "seed": args.seed,
        },
        "results": results,
        "test_base_rate": float(y_test.mean()),
        "elapsed_s": time.perf_counter() - t_start,
    }, indent=2))
    pd.concat(occ_rows, ignore_index=True).to_parquet(parquet_path, index=False)

    # Per-flight predictions for downstream calibration curves.
    flight_path = args.artifacts_dir / "full_calibration_validation_per_flight.parquet"
    pd.DataFrame(per_flight).to_parquet(flight_path, index=False)
    logger.info("wrote %s", flight_path)

    logger.info("=" * 64)
    logger.info("FULL-PIPELINE VALIDATION SUMMARY")
    logger.info("=" * 64)
    summary = pd.DataFrame(results).T[[
        "AUC", "Brier", "occ_MAE", "occ_RMSE", "occ_bias",
        "occ_pearson", "mean_p"]]
    logger.info("\n%s", summary.round(4).to_string())
    logger.info("wrote %s", json_path)
    logger.info("wrote %s", parquet_path)
    logger.info("total: %.1f min", (time.perf_counter() - t_start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
