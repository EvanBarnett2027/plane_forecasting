"""Deployment-realism backtest: production model on the most recent week.

Answers "how does the production LightGBM (no ``n_reports``) perform when
fed *forecast-condition* weather, compared to ground truth, over the most
recent labelled week?"

Why a noise proxy and not live forecasts
----------------------------------------
A true "real forecast vs realised label" test needs the two to overlap in
time. They cannot here: NWS only serves forecasts from *now* forward,
while the labelled BTS data ends months earlier -- and archived NWS
forecasts are not available from the public API. So we recreate the
forecast condition the way the whole project does: inject lead-time-aware
noise (``src.weather_noise``; sigma from persistence residuals, a
conservative upper bound on real forecast error) onto the observed
weather at each flight's real lead. For a genuine live test, run the
forward snapshot harness instead (see README).

What it does
------------
1.  ``week_start = floor(max_dep) - 7d``; the test week is
    ``[week_start, max_dep]``.
2.  Train a LightGBM on every flight scheduled **before** ``week_start``
    (last ``--val-days`` carved for early stopping) -- production-realistic,
    the week is fully held out.
3.  Train two variants on identical data: **no ``n_reports``** (the
    production model) and **with ``n_reports``** (to confirm dropping it
    doesn't hurt).
4.  Build per-occasion 48 h forecasts at ``--step-h`` cadence for issue
    times whose full window fits before ``max_dep``; inject noise at each
    flight's real lead.
5.  Report per-flight (AUC / AUC-PR / Brier) and per-occasion (MAE / RMSE /
    bias / Pearson) metrics vs realised labels, for both variants. Save a
    JSON report, a per-occasion parquet, and a 3-panel PNG.

Usage
-----
    python -m src.evaluate_recent_week
    python -m src.evaluate_recent_week --week-days 7 --noise-scale 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

from src.ingest_bts_ord import PROJECT_ROOT
from src.train_production_model import (
    DEFAULT_ARTIFACTS, LGBM_PARAMS, PRODUCTION_DROP, drop_production_features,
)
from src.training_pipeline import (
    DEFAULT_JOINED, DEFAULT_SIGMAS, TRAIN_START_DEFAULT, WINDOW_H,
    OCCASION_STEP_H,
    add_long_term_trend, add_sampled_leads, add_trailing_disruption_features,
    build_test_occasions, inject_noise, per_occasion_table, prepare_features,
)
from src.weather_noise import WeatherNoiseInjector

logger = logging.getLogger("evaluate_recent_week")


def _fit(X_tr, y_tr, X_va, y_va, seed):
    m = lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(20, verbose=False)])
    return m


def _flight_metrics(y, p):
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
    }


def _occasion_metrics(occ_n, p):
    t = per_occasion_table(occ_n, p)
    return t, {
        "occ_mae": float(t["abs_error"].mean()),
        "occ_rmse": float(np.sqrt((t["signed_error"] ** 2).mean())),
        "occ_bias": float(t["signed_error"].mean()),
        "occ_pearson": float(
            t[["actual_rate", "predicted_rate"]].corr().iloc[0, 1]),
        "n_occasions": int(t["occasion_id"].nunique()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    p.add_argument("--sigmas", type=Path, default=DEFAULT_SIGMAS)
    p.add_argument("--train-start", type=lambda s: pd.Timestamp(s, tz="UTC"),
                   default=TRAIN_START_DEFAULT)
    p.add_argument("--week-days", type=int, default=7,
                   help="Length of the recent test window (days).")
    p.add_argument("--val-days", type=int, default=30)
    p.add_argument("--step-h", type=int, default=OCCASION_STEP_H)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    rng = np.random.default_rng(args.seed)

    # ---- windows ----------------------------------------------------------
    df = pd.read_parquet(args.joined)
    dep = pd.to_datetime(df["scheduled_dep_utc_hour"], utc=True)
    df["scheduled_dep_utc_hour"] = dep
    df = df.loc[(dep >= args.train_start) & df["disrupted"].notna()].copy()
    max_dep = df["scheduled_dep_utc_hour"].max()
    week_start = max_dep.floor("D") - pd.Timedelta(days=args.week_days)
    val_cut = week_start - pd.Timedelta(days=args.val_days)
    # issue times whose full 48h window fits in the labelled data
    last_issue = max_dep - pd.Timedelta(hours=WINDOW_H)
    logger.info("test week [%s, %s]; issues [%s, %s] @ %dh; train < %s",
                week_start.date(), max_dep.date(), week_start.date(),
                last_issue, args.step_h, val_cut.date())

    train = df.loc[df["scheduled_dep_utc_hour"] < val_cut].reset_index(drop=True)
    val = df.loc[(df["scheduled_dep_utc_hour"] >= val_cut)
                 & (df["scheduled_dep_utc_hour"] < week_start)].reset_index(drop=True)
    test = df.loc[df["scheduled_dep_utc_hour"] >= week_start].reset_index(drop=True)
    logger.info("train=%d  val=%d  test-pool=%d", len(train), len(val), len(test))

    # ---- train/val features ----------------------------------------------
    train = add_sampled_leads(train, rng)
    val = add_sampled_leads(val, rng)
    train = add_long_term_trend(add_trailing_disruption_features(train, df))
    val = add_long_term_trend(add_trailing_disruption_features(val, df))

    noise = WeatherNoiseInjector.load(args.sigmas, seed=args.seed
                                       ).with_scale(args.noise_scale)
    train_n = inject_noise(train, noise)
    val_n = inject_noise(val, noise)
    X_train_full, cats = prepare_features(train_n)
    X_val_full, _ = prepare_features(val_n, cat_categories=cats)
    y_train = train_n["disrupted"].astype("int8").to_numpy()
    y_val = val_n["disrupted"].astype("int8").to_numpy()

    # ---- test occasions (forecast condition = noise at real leads) --------
    occ = build_test_occasions(test, week_start, last_issue, step_h=args.step_h)
    if not len(occ):
        raise SystemExit("no test occasions in the recent week")
    occ = add_long_term_trend(add_trailing_disruption_features(occ, df))
    occ_n = inject_noise(occ, noise)
    X_test_full, _ = prepare_features(occ_n, cat_categories=cats)
    y_test = occ_n["disrupted"].astype("int8").to_numpy()

    # ---- two variants: production (no n_reports) vs with n_reports --------
    results, preds, occ_tables = {}, {}, {}
    variants = {
        "production (no n_reports)": True,
        "with n_reports": False,
    }
    for name, drop in variants.items():
        Xtr = drop_production_features(X_train_full) if drop else X_train_full
        Xva = drop_production_features(X_val_full) if drop else X_val_full
        Xte = drop_production_features(X_test_full) if drop else X_test_full
        model = _fit(Xtr, y_train, Xva, y_val, args.seed)
        p_test = model.predict_proba(Xte)[:, 1]
        preds[name] = p_test
        t, occ_m = _occasion_metrics(occ_n, p_test)
        occ_tables[name] = t
        results[name] = {**_flight_metrics(y_test, p_test), **occ_m,
                         "n_features": Xtr.shape[1]}
        logger.info("[%s] AUC=%.4f Brier=%.4f | occ MAE=%.4f bias=%+.4f r=%.3f",
                    name, results[name]["auc_roc"], results[name]["brier"],
                    results[name]["occ_mae"], results[name]["occ_bias"],
                    results[name]["occ_pearson"])

    # ---- report -----------------------------------------------------------
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "test_week": {"start": str(week_start), "end": str(max_dep),
                      "n_occasions": results["production (no n_reports)"]["n_occasions"],
                      "n_test_flight_occasions": int(len(y_test))},
        "forecast_condition": f"weather-noise proxy, scale={args.noise_scale}",
        "results": results,
    }
    (args.artifacts_dir / "recent_week_eval.json").write_text(
        json.dumps(report, indent=2))

    prod = occ_tables["production (no n_reports)"].copy()
    prod.to_parquet(args.artifacts_dir / "recent_week_per_occasion.parquet",
                    index=False)

    # ---- 3-panel plot -----------------------------------------------------
    prod_p = preds["production (no n_reports)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # (a) time series of predicted vs actual per-occasion rate
    t = prod.sort_values("t_issue")
    axes[0].plot(t["t_issue"], t["actual_rate"], "o-", color="k",
                 lw=1.5, label="actual")
    axes[0].plot(t["t_issue"], t["predicted_rate"], "s-", color="C0",
                 lw=1.5, label="predicted (production)")
    axes[0].set_title("48h disruption rate: predicted vs actual")
    axes[0].set_ylabel("rate"); axes[0].set_xlabel("issue time")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for lbl in axes[0].get_xticklabels():
        lbl.set_rotation(25)

    # (b) calibration
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for name, p, c in [("production (no n_reports)", prod_p, "C0"),
                       ("with n_reports", preds["with n_reports"], "C3")]:
        fp, mp = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        axes[1].plot(mp, fp, "o-", color=c, label=name)
    axes[1].set_title("Per-flight calibration")
    axes[1].set_xlabel("mean predicted P"); axes[1].set_ylabel("observed freq")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

    # (c) per-occasion scatter
    axes[2].scatter(t["actual_rate"], t["predicted_rate"], s=28, alpha=0.7,
                    color="C0")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    mae = results["production (no n_reports)"]["occ_mae"]
    axes[2].set_title(f"Per-occasion (MAE={mae:.3f})")
    axes[2].set_xlabel("actual rate"); axes[2].set_ylabel("predicted rate")
    axes[2].grid(alpha=0.3)
    lim = max(0.05, t[["actual_rate", "predicted_rate"]].max().max() * 1.1)
    axes[2].set_xlim(0, lim); axes[2].set_ylim(0, lim)

    fig.suptitle(f"Production model on recent week "
                 f"[{week_start.date()} -> {max_dep.date()}]  "
                 f"(forecast-noise proxy, scale={args.noise_scale})", y=1.02)
    fig.tight_layout()
    png = args.artifacts_dir / "recent_week_eval.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    logger.info("wrote %s, recent_week_eval.json, recent_week_per_occasion.parquet",
                png.name)

    # ---- console summary --------------------------------------------------
    logger.info("=" * 70)
    logger.info("RECENT-WEEK SUMMARY  (test week %s -> %s, %d occasions)",
                week_start.date(), max_dep.date(),
                report["test_week"]["n_occasions"])
    summary = pd.DataFrame(results).T[
        ["n_features", "auc_roc", "auc_pr", "brier", "base_rate",
         "occ_mae", "occ_rmse", "occ_bias", "occ_pearson"]]
    logger.info("\n%s", summary.round(4).to_string())
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
