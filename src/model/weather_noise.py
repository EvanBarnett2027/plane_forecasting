"""Inject lead-time-aware synthetic noise into weather features at train time.

Why
---
At inference the model will see weather *forecasts*, not observations. Training
on clean observations and deploying on noisy forecasts is a train/test mismatch:
the model overfits to precise values it won't actually have. Adding noise to
the weather columns during training matches the input distribution to the
deployment distribution. (For a stronger fix, train on archived gridded
forecasts directly; this module is the cheap honest alternative.)

How much noise: per-column persistence residuals
------------------------------------------------
For each numeric weather column ``v`` and lead time ``L``::

    sigma_v(L) = std( v(t) - v(t - L) )   pooled across stations

This is what *changes* between forecast issue and verification time, which is a
conservative upper bound on real forecast error -- good forecasts beat
persistence, especially at short leads. No external data required; everything
is computed from the cleaned weather parquet.

Per-column noise model
----------------------
- Symmetric numerics (tmpf, dwpf, relh, alti, mslp, feel): additive N(0, sigma)
- Non-negative (sknt, gust, peak_wind_gust, skyl1): additive N(0, sigma),
  clipped at 0
- Circular (drct): (d + N(0, sigma_deg)) mod 360, residuals computed circularly
- Bounded (vsby): additive N(0, sigma), clipped to [0, 10]
- Zero-inflated (p01i, ice_accretion_1hr, snowdepth): multiplicative log-normal
  noise on >0 values, zeros preserved (no zero<->non-zero flips in v1)
- Categorical / text / diagnostic: left alone

Lead-time awareness
-------------------
``transform`` takes per-row ``lead_origin_h`` and ``lead_dest_h``. sigma is
linearly interpolated between the precomputed leads. Because uncertainty
grows with lead time, a flight predicted 6 h out gets less noise on its
weather features than one predicted 48 h out.

Reproducibility
---------------
A single ``WeatherNoiseInjector`` carries its sigma table, an RNG with a
seed, and a global ``scale`` multiplier (treat scale as a hyperparameter:
sweep {0.25, 0.5, 1.0, 1.5} on the validation set). The sigma table is
persisted to JSON so training runs are deterministic.

Usage
-----
Build the sigma table once::

    python -m src.weather_noise fit \\
        --weather data/processed/dest_weather_iem_2020_present.parquet \\
        --out data/processed/weather_noise_sigmas.json

Use it in a training pipeline::

    noise = WeatherNoiseInjector.load(SIGMA_PATH).with_scale(1.0).with_seed(42)
    X_train = noise.transform(
        X_train,
        lead_origin_h=df_train["lead_origin_h"],
        lead_dest_h=df_train["lead_dest_h"],
    )
    # X_val / X_test left alone (or pass scale=0 / use a fixed seed)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("weather_noise")

# --------------------------------------------------------------------------- #
# Per-column kind dispatch
# --------------------------------------------------------------------------- #

GAUSSIAN_COLS = ("tmpf", "dwpf", "relh", "alti", "mslp", "feel")
NON_NEG_COLS = ("sknt", "gust", "peak_wind_gust", "skyl1")
CIRCULAR_COLS = ("drct",)
BOUNDED_COLS: dict[str, float] = {"vsby": 10.0}      # upper bound (lower = 0)
ZERO_INFLATED_COLS = ("p01i", "ice_accretion_1hr", "snowdepth")

# Variables with a strong daily cycle. Persistence residuals at non-24h
# leads conflate diurnal swing with real forecast error (e.g. tmpf at
# L=12h compares noon to midnight). For these columns we fit sigma only
# at the diurnal-aligned leads {24, 48} and interpolate/clamp the rest --
# np.interp clamps shorter leads to sigma(24), which is conservative
# (over-noisy at very short leads) but never contaminated by the cycle.
# Other columns (wind, visibility, pressure, precip) lack a strong
# diurnal pattern, so their short-lead residuals are honest forecast-
# error proxies and are kept.
DIURNAL_COLS = ("tmpf", "relh", "feel")
DIURNAL_SAFE_LEADS = (24, 48)

# Anything not in the lists above is left untouched (skyc1, wxcodes, station,
# valid_utc, valid_local, n_reports, and every non-weather feature).
ALL_NOISED = (
    set(GAUSSIAN_COLS) | set(NON_NEG_COLS) | set(CIRCULAR_COLS)
    | set(BOUNDED_COLS) | set(ZERO_INFLATED_COLS)
)

DEFAULT_LEADS = (1, 3, 6, 12, 24, 48)


# --------------------------------------------------------------------------- #
# Injector
# --------------------------------------------------------------------------- #


@dataclass
class WeatherNoiseInjector:
    """Train-time additive-noise transform for forecast-uncertainty modelling."""

    scale: float = 1.0
    seed: int | None = None
    # raw column name -> {lead_h (int) -> sigma (float)}
    sigmas: dict[str, dict[int, float]] = field(default_factory=dict)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ----- builders -------------------------------------------------------- #

    def with_scale(self, scale: float) -> "WeatherNoiseInjector":
        self.scale = float(scale)
        return self

    def with_seed(self, seed: int) -> "WeatherNoiseInjector":
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        return self

    # ----- fit ------------------------------------------------------------- #

    def fit(self, weather_long: pd.DataFrame, *,
            lead_hours: Sequence[int] = DEFAULT_LEADS,
            station_col: str = "station",
            time_col: str = "valid_utc") -> "WeatherNoiseInjector":
        """Estimate per-column persistence-residual sigma at each lead time.

        ``weather_long`` should be the combined per-station hourly weather
        (the output of ``clean_iem_asos --combined-out``).
        """
        if station_col not in weather_long or time_col not in weather_long:
            raise ValueError(
                f"weather_long must contain '{station_col}' and '{time_col}'"
            )
        # Sort within each station so .shift(L) is by-station.
        wx = weather_long.sort_values([station_col, time_col])
        self.sigmas = {}
        for col in wx.columns:
            if col not in ALL_NOISED:
                continue
            # Restrict diurnal-cycle columns to {24, 48} to avoid the
            # noon-vs-midnight contamination in non-24h residuals.
            if col in DIURNAL_COLS:
                col_leads = tuple(L for L in lead_hours
                                  if L in DIURNAL_SAFE_LEADS)
                if not col_leads:
                    logger.warning("  %s is diurnal: needs at least one of "
                                    "leads %s; skipping", col,
                                    DIURNAL_SAFE_LEADS)
                    continue
            else:
                col_leads = tuple(lead_hours)
            self.sigmas[col] = self._fit_one(wx, col, col_leads, station_col)
            sample = {L: round(self.sigmas[col][L], 3) for L in col_leads}
            logger.info("  sigma[%-12s] = %s%s", col, sample,
                        "  [diurnal]" if col in DIURNAL_COLS else "")
        return self

    def _fit_one(self, wx: pd.DataFrame, col: str,
                 lead_hours: Sequence[int],
                 station_col: str) -> dict[int, float]:
        out: dict[int, float] = {}
        v = wx[col]
        for L in lead_hours:
            shifted = wx.groupby(station_col, sort=False)[col].shift(L)
            if col in CIRCULAR_COLS:
                resid = ((v - shifted + 180.0) % 360.0) - 180.0
            elif col in ZERO_INFLATED_COLS:
                # Log-multiplicative residual on the both-positive subset.
                # Zeros are handled deterministically at transform time.
                mask = (v > 0) & (shifted > 0)
                if not mask.any():
                    out[L] = 0.0
                    continue
                resid = np.log(v.where(mask)) - np.log(shifted.where(mask))
            else:
                resid = v - shifted
            out[L] = float(resid.dropna().std())
        return out

    # ----- transform ------------------------------------------------------- #

    def transform(self, df: pd.DataFrame, *,
                  lead_origin_h: int | float | pd.Series,
                  lead_dest_h: int | float | pd.Series,
                  prefixes: Sequence[str] = ("origin_wx_", "dest_wx_"),
                  ) -> pd.DataFrame:
        """Return a copy of ``df`` with noise added to every ``*_wx_*`` column.

        ``lead_origin_h`` / ``lead_dest_h`` may be a scalar (same lead for
        every row) or a per-row pandas Series. Non-weather columns are
        copied through unchanged. NaN inputs are preserved as NaN.
        """
        if not self.sigmas:
            raise RuntimeError(
                "Injector has no sigmas. Call fit(...) or load(path) first."
            )
        if self.scale == 0.0:
            return df.copy()

        out = df.copy()
        leads = {
            prefixes[0]: _broadcast(lead_origin_h, len(df)),
            prefixes[1]: _broadcast(lead_dest_h, len(df)),
        }
        for col in df.columns:
            for prefix in prefixes:
                if col.startswith(prefix):
                    raw = col[len(prefix):]
                    if raw in ALL_NOISED:
                        out[col] = self._noise_one(
                            df[col], raw, leads[prefix])
                    break
        return out

    def _noise_one(self, values: pd.Series, raw: str,
                   leads: np.ndarray) -> pd.Series:
        sigma_per_row = self._sigma_at(raw, leads) * self.scale
        v_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        nan_mask = np.isnan(v_arr)

        if raw in CIRCULAR_COLS:
            noise = self._rng.normal(0.0, sigma_per_row)
            out = (v_arr + noise) % 360.0
        elif raw in ZERO_INFLATED_COLS:
            # log-normal multiplicative noise on positives; zeros preserved.
            out = v_arr.copy()
            pos = (v_arr > 0) & ~nan_mask
            if pos.any():
                ln_noise = self._rng.normal(0.0, sigma_per_row[pos])
                out[pos] = v_arr[pos] * np.exp(ln_noise)
            # zeros stay 0 (no flip in v1)
        elif raw in BOUNDED_COLS:
            noise = self._rng.normal(0.0, sigma_per_row)
            out = np.clip(v_arr + noise, 0.0, BOUNDED_COLS[raw])
        elif raw in NON_NEG_COLS:
            noise = self._rng.normal(0.0, sigma_per_row)
            out = np.clip(v_arr + noise, 0.0, None)
        elif raw in GAUSSIAN_COLS:
            noise = self._rng.normal(0.0, sigma_per_row)
            out = v_arr + noise
        else:                                  # not reached given ALL_NOISED check
            out = v_arr

        out[nan_mask] = np.nan
        return pd.Series(out, index=values.index, name=values.name)

    def _sigma_at(self, raw: str, leads: np.ndarray) -> np.ndarray:
        """Linearly interpolate sigma at arbitrary per-row lead times."""
        table = self.sigmas[raw]
        ks = np.array(sorted(table.keys()), dtype=float)
        vs = np.array([table[int(k)] for k in ks], dtype=float)
        # np.interp clips to endpoints outside [ks.min, ks.max], which is the
        # behaviour we want -- no extrapolation to unseen leads.
        return np.interp(np.asarray(leads, dtype=float), ks, vs)

    # ----- persistence ----------------------------------------------------- #

    def save(self, path: Path) -> None:
        payload = {
            "scale": self.scale,
            "sigmas": {
                col: {str(L): float(s) for L, s in by_lead.items()}
                for col, by_lead in self.sigmas.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, seed: int | None = None) -> "WeatherNoiseInjector":
        payload = json.loads(Path(path).read_text())
        obj = cls(scale=float(payload.get("scale", 1.0)), seed=seed)
        obj.sigmas = {
            col: {int(L): float(s) for L, s in by_lead.items()}
            for col, by_lead in payload.get("sigmas", {}).items()
        }
        return obj


def _broadcast(value: int | float | pd.Series, n: int) -> np.ndarray:
    if isinstance(value, pd.Series):
        if len(value) != n:
            raise ValueError(f"lead series length {len(value)} != df length {n}")
        return value.to_numpy(dtype=float)
    return np.full(n, float(value), dtype=float)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cmd_fit(args: argparse.Namespace) -> int:
    logger.info("loading %s ...", args.weather)
    wx = pd.read_parquet(args.weather)
    logger.info("  %d rows, %d stations",
                len(wx), wx["station"].nunique())
    inj = WeatherNoiseInjector().fit(wx, lead_hours=tuple(args.leads))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inj.save(args.out)
    logger.info("wrote %s (%d cols x %d leads)",
                args.out, len(inj.sigmas), len(args.leads))
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser(
        "fit", help="Estimate sigma table from observed weather and save it.")
    fit_p.add_argument("--weather", type=Path, required=True,
                       help="Path to a combined-long weather parquet (station, "
                            "valid_utc, weather columns).")
    fit_p.add_argument("--out", type=Path, required=True,
                       help="Output JSON path for the sigma table.")
    fit_p.add_argument("--leads", type=int, nargs="+",
                       default=list(DEFAULT_LEADS),
                       help=f"Lead times in hours (default: {DEFAULT_LEADS}).")
    fit_p.set_defaults(func=_cmd_fit)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
