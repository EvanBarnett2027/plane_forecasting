"""Tests for src.weather_noise.WeatherNoiseInjector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.weather_noise import (
    BOUNDED_COLS,
    WeatherNoiseInjector,
    _broadcast,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _synthetic_weather(n_hours: int = 24 * 60, n_stations: int = 2,
                       seed: int = 0) -> pd.DataFrame:
    """Smooth synthetic hourly weather long table -- enough to fit sigmas on."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for s_idx in range(n_stations):
        s = f"S{s_idx:02d}"
        t = base + pd.to_timedelta(np.arange(n_hours), unit="h")
        # AR(1)-ish series so persistence residuals are meaningful (not white).
        def ar1(sd):
            x = np.zeros(n_hours)
            for i in range(1, n_hours):
                x[i] = 0.9 * x[i - 1] + rng.normal(0, sd)
            return x
        tmpf = 40 + ar1(1.0)
        sknt = np.clip(8 + ar1(0.5), 0, None)
        gust = np.clip(sknt + np.abs(ar1(0.3)), 0, None)
        drct = (180 + ar1(20.0)) % 360.0
        p01i = np.where(rng.random(n_hours) < 0.1,
                        np.abs(rng.normal(0, 0.05, n_hours)), 0.0)
        vsby = np.clip(10 - np.abs(ar1(0.5)), 0, 10)
        rows.append(pd.DataFrame({
            "station": s, "valid_utc": t,
            "tmpf": tmpf, "sknt": sknt, "gust": gust,
            "drct": drct, "p01i": p01i, "vsby": vsby,
        }))
    return pd.concat(rows, ignore_index=True)


def _flight_like(n: int = 200, seed: int = 1) -> pd.DataFrame:
    """A minimal flights-with-weather frame for transform() tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "flight_id": np.arange(n),
        "dest": rng.choice(["LAX", "JFK"], n),
        "crs_elapsed_time": rng.integers(60, 360, n),
        # weather columns (origin + dest), some NaN seeded in
        "origin_wx_tmpf": rng.normal(40, 10, n),
        "origin_wx_sknt": np.clip(rng.normal(10, 4, n), 0, None),
        "origin_wx_drct": rng.uniform(0, 360, n),
        "origin_wx_vsby": np.clip(rng.normal(8, 2, n), 0, 10),
        "origin_wx_p01i": np.where(rng.random(n) < 0.2,
                                    np.abs(rng.normal(0, 0.05, n)), 0.0),
        "dest_wx_tmpf": rng.normal(55, 12, n),
        "dest_wx_sknt": np.clip(rng.normal(8, 3, n), 0, None),
        "dest_wx_drct": rng.uniform(0, 360, n),
        "dest_wx_vsby": np.clip(rng.normal(9, 1, n), 0, 10),
        "dest_wx_p01i": np.where(rng.random(n) < 0.15,
                                  np.abs(rng.normal(0, 0.05, n)), 0.0),
    })


@pytest.fixture(scope="module")
def fitted_injector():
    wx = _synthetic_weather()
    return WeatherNoiseInjector(scale=1.0, seed=42).fit(
        wx, lead_hours=(1, 6, 24))


# --------------------------------------------------------------------------- #
# fit()
# --------------------------------------------------------------------------- #


def test_fit_populates_sigmas_for_known_columns(fitted_injector):
    """Non-diurnal cols get every requested lead; diurnal cols only {24, 48}."""
    inj = fitted_injector
    for col in ("sknt", "drct", "vsby", "p01i", "gust"):
        assert col in inj.sigmas, f"missing sigma for {col}"
        assert set(inj.sigmas[col].keys()) == {1, 6, 24}
    # tmpf is in DIURNAL_COLS -> intersection of (1,6,24) with (24,48) = {24}.
    assert set(inj.sigmas["tmpf"].keys()) == {24}


def test_fit_sigma_grows_with_lead(fitted_injector):
    """Persistence residuals should increase (weakly) with lead.

    Use only non-diurnal columns; diurnal ones are fit at {24, 48} only
    so a 1h->24h comparison doesn't exist for them.
    """
    for col in ("drct", "sknt"):
        s = fitted_injector.sigmas[col]
        assert s[1] <= s[24] + 1e-6, f"{col}: sigma should not shrink with lead"


def test_fit_diurnal_cols_only_have_24_and_48():
    """Diurnal columns must avoid the noon-vs-midnight persistence artefact."""
    wx = _synthetic_weather(n_hours=24 * 30)
    # The synthetic data doesn't include diurnal cols by default, so add one.
    wx["tmpf_diurnal"] = wx["tmpf"]
    wx = wx.rename(columns={"tmpf_diurnal": "relh"})  # relh is in DIURNAL_COLS
    inj = WeatherNoiseInjector().fit(
        wx, lead_hours=(1, 3, 6, 12, 24, 48))
    # relh should only be fit at the diurnal-safe leads.
    assert set(inj.sigmas["relh"].keys()) == {24, 48}
    # sknt is not diurnal -> all leads kept.
    assert set(inj.sigmas["sknt"].keys()) == {1, 3, 6, 12, 24, 48}


def test_diurnal_short_lead_clamps_to_24h_sigma(fitted_injector):
    """At L < 24 a diurnal col should get sigma(24) (np.interp clamping)."""
    # fitted_injector's synthetic data has no diurnal cols -> add one manually
    # by constructing an injector with synthetic relh sigmas.
    inj = WeatherNoiseInjector(scale=1.0, seed=0)
    inj.sigmas = {"relh": {24: 10.0, 48: 15.0}}
    df = pd.DataFrame({
        "origin_wx_relh": [50.0] * 1000,
        "dest_wx_relh": [50.0] * 1000,
    })
    out_short = inj.transform(df, lead_origin_h=6, lead_dest_h=6)
    out_long = inj.transform(df, lead_origin_h=24, lead_dest_h=24)
    # both should use sigma(24) -> roughly the same noise scale
    s_short = (out_short["origin_wx_relh"] - 50).std()
    s_long = (out_long["origin_wx_relh"] - 50).std()
    assert abs(s_short - s_long) < 1.5, (
        f"L=6 sigma {s_short:.2f} should ~ L=24 sigma {s_long:.2f}")


def test_fit_ignores_unknown_columns():
    """Non-numeric / categorical / text columns should not get sigmas."""
    wx = _synthetic_weather(n_hours=24 * 10)
    wx["wxcodes"] = "BR"
    wx["skyc1"] = "BKN"
    inj = WeatherNoiseInjector().fit(wx, lead_hours=(1, 6))
    assert "wxcodes" not in inj.sigmas
    assert "skyc1" not in inj.sigmas
    assert "station" not in inj.sigmas


# --------------------------------------------------------------------------- #
# transform()
# --------------------------------------------------------------------------- #


def test_transform_leaves_non_weather_columns_unchanged(fitted_injector):
    df = _flight_like()
    out = fitted_injector.transform(df, lead_origin_h=6, lead_dest_h=12)
    for col in ("flight_id", "dest", "crs_elapsed_time"):
        pd.testing.assert_series_equal(out[col], df[col], check_names=False)


def test_transform_drct_stays_in_range(fitted_injector):
    df = _flight_like()
    out = fitted_injector.transform(df, lead_origin_h=24, lead_dest_h=24)
    for col in ("origin_wx_drct", "dest_wx_drct"):
        assert out[col].between(0, 360, inclusive="left").all(), (
            f"{col} escaped [0, 360)")


def test_transform_non_negative_cols_stay_non_negative(fitted_injector):
    df = _flight_like()
    out = fitted_injector.transform(df, lead_origin_h=24, lead_dest_h=24)
    for col in ("origin_wx_sknt", "dest_wx_sknt"):
        assert (out[col] >= 0).all(), f"{col} went negative"


def test_transform_bounded_cols_stay_in_range(fitted_injector):
    df = _flight_like()
    out = fitted_injector.transform(df, lead_origin_h=24, lead_dest_h=24)
    hi = BOUNDED_COLS["vsby"]
    for col in ("origin_wx_vsby", "dest_wx_vsby"):
        assert out[col].between(0, hi).all(), f"{col} escaped [0, {hi}]"


def test_transform_preserves_nan(fitted_injector):
    df = _flight_like()
    df.loc[:9, "origin_wx_tmpf"] = np.nan
    df.loc[10:19, "dest_wx_sknt"] = np.nan
    out = fitted_injector.transform(df, lead_origin_h=6, lead_dest_h=6)
    assert out.loc[:9, "origin_wx_tmpf"].isna().all()
    assert out.loc[10:19, "dest_wx_sknt"].isna().all()
    # everywhere else: not introduce new NaN
    untouched = ~df["origin_wx_tmpf"].isna()
    assert out.loc[untouched, "origin_wx_tmpf"].notna().all()


def test_transform_preserves_zeros_in_zero_inflated(fitted_injector):
    df = _flight_like()
    df["origin_wx_p01i"] = 0.0
    out = fitted_injector.transform(df, lead_origin_h=24, lead_dest_h=24)
    assert (out["origin_wx_p01i"] == 0.0).all(), (
        "zero-inflated col should keep zeros (no flips in v1)")


def test_scale_zero_is_identity(fitted_injector):
    df = _flight_like()
    inj = WeatherNoiseInjector(scale=0.0)
    inj.sigmas = fitted_injector.sigmas
    out = inj.transform(df, lead_origin_h=24, lead_dest_h=48)
    pd.testing.assert_frame_equal(out, df)


def test_seed_reproducibility(fitted_injector):
    df = _flight_like()
    a = WeatherNoiseInjector(scale=1.0, seed=7)
    a.sigmas = fitted_injector.sigmas
    b = WeatherNoiseInjector(scale=1.0, seed=7)
    b.sigmas = fitted_injector.sigmas
    pd.testing.assert_frame_equal(
        a.transform(df, lead_origin_h=12, lead_dest_h=24),
        b.transform(df, lead_origin_h=12, lead_dest_h=24),
    )


def test_per_row_leads_supported(fitted_injector):
    df = _flight_like(n=50)
    leads_o = pd.Series(np.full(50, 6.0))
    leads_d = pd.Series(np.full(50, 24.0))
    out = fitted_injector.transform(df, lead_origin_h=leads_o, lead_dest_h=leads_d)
    assert len(out) == 50


def test_unfitted_raises():
    df = _flight_like(n=10)
    with pytest.raises(RuntimeError):
        WeatherNoiseInjector().transform(df, lead_origin_h=6, lead_dest_h=12)


def test_lead_series_wrong_length_raises():
    with pytest.raises(ValueError):
        _broadcast(pd.Series([1, 2, 3]), n=10)


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #


def test_save_load_roundtrip(fitted_injector, tmp_path):
    p = tmp_path / "sigmas.json"
    fitted_injector.save(p)
    loaded = WeatherNoiseInjector.load(p, seed=fitted_injector.seed)
    assert loaded.scale == fitted_injector.scale
    assert loaded.sigmas == fitted_injector.sigmas
    # Lead keys should be ints, not strings (JSON would naturally stringify).
    any_col = next(iter(loaded.sigmas))
    assert all(isinstance(k, int) for k in loaded.sigmas[any_col].keys())


def test_load_then_transform_matches_pre_save(fitted_injector, tmp_path):
    p = tmp_path / "sigmas.json"
    fitted_injector.save(p)
    df = _flight_like(n=20)
    pre = WeatherNoiseInjector(scale=1.0, seed=99)
    pre.sigmas = fitted_injector.sigmas
    a = pre.transform(df, lead_origin_h=24, lead_dest_h=24)
    post = WeatherNoiseInjector.load(p, seed=99)
    b = post.transform(df, lead_origin_h=24, lead_dest_h=24)
    pd.testing.assert_frame_equal(a, b)
