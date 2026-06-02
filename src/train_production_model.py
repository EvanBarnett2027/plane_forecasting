"""Train the production LightGBM disruption model (no ``n_reports``).

This is the model the real-time dashboard serves. Two deliberate choices
distinguish it from the experimental ``training_pipeline``:

1.  **No ``*_wx_n_reports`` feature.** ``n_reports`` is the count of
    sub-hourly METARs that fed each historical hourly weather row. It
    ranked high in importance, but it is a *diagnostic of the
    observation stream*, not a weather variable -- and a **forecast has
    no equivalent** (see :mod:`src.live_weather`). Keeping it would mean
    training on a feature that is always NaN at serve time: a guaranteed
    train/serve skew. We drop it from both endpoints and retrain so the
    model only relies on features the live pipeline can actually supply.

2.  **No isotonic calibrator.** Shipped raw. (The full-pipeline study in
    ``notebooks/lightgbm_investigation.ipynb`` showed damped isotonic
    helps per-occasion MAE but *worsens* per-flight ECE; whether to add
    it is a downstream-consumer decision, kept out of the core artifact.)

The model trains on **all** labelled data from ``--train-start`` through
the latest available, holding out only the last ``--val-days`` for early
stopping -- a production model should use every row it can.

Artifacts (under ``--artifacts-dir``, default ``artifacts/production_model``):

* ``model.joblib``    -- {model, categories, feature_names, drop_features}
* ``model.txt``       -- LightGBM native booster dump (portable; survives
                         library-version drift better than a pickle)
* ``metadata.json``   -- feature list, category vocabularies, library
                         versions, train window, base rate, sanity metrics

Usage
-----
    python -m src.train_production_model
    python -m src.train_production_model --val-days 30
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
import sklearn
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

from src.ingest_bts_ord import PROJECT_ROOT
from src.training_pipeline import (
    DEFAULT_JOINED, DEFAULT_SIGMAS, TRAIN_START_DEFAULT, WINDOW_H,
    OCCASION_STEP_H,
    add_long_term_trend, add_sampled_leads, add_trailing_disruption_features,
    build_test_occasions, inject_noise, per_occasion_table, prepare_features,
)
from src.weather_noise import WeatherNoiseInjector

logger = logging.getLogger("train_production_model")

DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "production_model"

# Features the live pipeline cannot supply (no forecast equivalent).
PRODUCTION_DROP = ("origin_wx_n_reports", "dest_wx_n_reports")

# Production LightGBM hyperparameters (same family as training_pipeline).
LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=200, subsample=0.9, colsample_bytree=0.9,
    objective="binary", n_jobs=-1, verbose=-1,
)


def drop_production_features(X: pd.DataFrame) -> pd.DataFrame:
    """Remove the features the live serving pipeline cannot provide."""
    return X.drop(columns=[c for c in PRODUCTION_DROP if c in X.columns])


def _library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    p.add_argument("--sigmas", type=Path, default=DEFAULT_SIGMAS)
    p.add_argument("--train-start", type=lambda s: pd.Timestamp(s, tz="UTC"),
                   default=TRAIN_START_DEFAULT)
    p.add_argument("--val-days", type=int, default=30,
                   help="Last N days held out (time-ordered) for early stop.")
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    rng = np.random.default_rng(args.seed)

    # ---- load + time-ordered train/val (use ALL labelled data) ------------
    logger.info("loading %s", args.joined)
    df = pd.read_parquet(args.joined)
    dep = pd.to_datetime(df["scheduled_dep_utc_hour"], utc=True)
    df["scheduled_dep_utc_hour"] = dep
    df = df.loc[(dep >= args.train_start) & df["disrupted"].notna()].copy()
    max_dep = df["scheduled_dep_utc_hour"].max()
    val_cut = max_dep - pd.Timedelta(days=args.val_days)
    train = df.loc[df["scheduled_dep_utc_hour"] < val_cut].reset_index(drop=True)
    val = df.loc[df["scheduled_dep_utc_hour"] >= val_cut].reset_index(drop=True)
    logger.info("train=%d (<%s)  val=%d (last %dd)",
                len(train), val_cut.date(), len(val), args.val_days)

    # ---- features: leads + trailing + trend + noise -----------------------
    train = add_sampled_leads(train, rng)
    val = add_sampled_leads(val, rng)
    pool = pd.concat([train, val], ignore_index=True)
    train = add_long_term_trend(add_trailing_disruption_features(train, pool))
    val = add_long_term_trend(add_trailing_disruption_features(val, pool))

    noise = WeatherNoiseInjector.load(args.sigmas, seed=args.seed
                                       ).with_scale(args.noise_scale)
    train_n = inject_noise(train, noise)
    val_n = inject_noise(val, noise)

    X_train, cats = prepare_features(train_n)
    X_val, _ = prepare_features(val_n, cat_categories=cats)
    X_train = drop_production_features(X_train)
    X_val = drop_production_features(X_val)
    y_train = train_n["disrupted"].astype("int8").to_numpy()
    y_val = val_n["disrupted"].astype("int8").to_numpy()
    logger.info("fitting LightGBM on %d rows x %d features (n_reports dropped)",
                len(X_train), X_train.shape[1])

    # ---- fit --------------------------------------------------------------
    model = lgb.LGBMClassifier(random_state=args.seed, **LGBM_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False)])

    p_val = model.predict_proba(X_val)[:, 1]
    val_metrics = {
        "val_auc_roc": float(roc_auc_score(y_val, p_val)),
        "val_auc_pr": float(average_precision_score(y_val, p_val)),
        "val_brier": float(brier_score_loss(y_val, p_val)),
        "val_base_rate": float(y_val.mean()),
    }
    logger.info("validation: AUC=%.4f  AUC-PR=%.4f  Brier=%.4f  base=%.4f",
                val_metrics["val_auc_roc"], val_metrics["val_auc_pr"],
                val_metrics["val_brier"], val_metrics["val_base_rate"])

    # ---- persist a deployable bundle --------------------------------------
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(X_train.columns)
    joblib.dump(
        {"model": model, "categories": cats, "feature_names": feature_names,
         "drop_features": list(PRODUCTION_DROP), "uses_isotonic": False},
        args.artifacts_dir / "model.joblib")
    model.booster_.save_model(str(args.artifacts_dir / "model.txt"))

    metadata = {
        "model_type": "lightgbm.LGBMClassifier",
        "uses_n_reports": False,
        "uses_isotonic": False,
        "drop_features": list(PRODUCTION_DROP),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "categories": {k: list(map(str, v)) for k, v in cats.items()},
        "hyperparameters": {**LGBM_PARAMS, "best_iteration": int(
            getattr(model, "best_iteration_", 0) or model.n_estimators)},
        "library_versions": _library_versions(),
        "train_window": {
            "train_start": str(args.train_start),
            "train_end_exclusive": str(val_cut),
            "val_days": args.val_days,
            "data_max_dep": str(max_dep),
        },
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "train_base_rate": float(y_train.mean()),
        "noise_scale": args.noise_scale,
        "seed": args.seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        **val_metrics,
    }
    (args.artifacts_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2))
    logger.info("wrote production bundle to %s "
                "(model.joblib, model.txt, metadata.json)", args.artifacts_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
