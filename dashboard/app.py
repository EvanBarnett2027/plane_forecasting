"""ORD departure-disruption dashboard (Streamlit).

Shows the production model's P(disrupted) for every ORD departure in the
next 48 hours, with a refresh button, a search bar, and rich filtering.

Run
---
    # demo source (`no API key needed -- replays a real window)
    .venv/bin/streamlit run dashboard/app.py

    # live source (real next-48h schedule + NWS forecasts)
    AERODATABOX_KEY=... .venv/bin/streamlit run dashboard/app.py

The heavy work (fetch + score) lives in ``src.serve``; this file is only
presentation. Predictions are cached and only recomputed when the user
hits **Refresh** (or changes the data source / demo time), so typing in
the search box and changing filters stay instant.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

# Make ``src.*`` importable when run via ``streamlit run dashboard/app.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serve import (  # noqa: E402
    DEMO_DEFAULT_NOW, RISK_BANDS, load_model_bundle, predict_next_48h,
    resolve_api_key,
)

RISK_ORDER = [b[2] for b in RISK_BANDS]
RISK_COLORS = {"Low": "#2ca02c", "Moderate": "#ff9e1b",
               "Elevated": "#ff6f3c", "High": "#d62728"}

st.set_page_config(page_title="ORD Disruption Forecast",
                   page_icon="✈️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached prediction (recomputed only on refresh / source / time change)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def _bundle():
    return load_model_bundle()


@st.cache_data(show_spinner="Fetching latest data and scoring…")
def _predict(source: str, api_key: str | None, replay_now: str | None,
             weather: bool, provider: str, _refresh_token: int):
    """``_refresh_token`` busts the cache when Refresh is pressed."""
    if source == "live":
        os.environ["AERODATABOX_PROVIDER"] = provider
    now = (pd.Timestamp(replay_now, tz="UTC")
           if (source == "replay" and replay_now) else None)
    pred = predict_next_48h(source=source, api_key=api_key, now=now,
                            weather=weather, bundle=_bundle())
    # return plain data (cache-friendly); rebuild the dataclass-free view
    return {
        "as_of": pred.as_of, "source": pred.source,
        "flights": pred.flights, "by_hour": pred.by_hour,
        "predicted_rate": pred.predicted_rate, "n_flights": pred.n_flights,
        "actual_rate": pred.actual_rate,
    }


# --------------------------------------------------------------------------- #
# Sidebar: source, refresh, filters
# --------------------------------------------------------------------------- #

st.session_state.setdefault("refresh_token", 0)

with st.sidebar:
    st.header("Data source")
    env_key = resolve_api_key()   # aerodatabox.key file -> $AERODATABOX_KEY
    # source = st.radio(
    #     "Source",
    #     options=["upcoming", "replay", "live"],
    #     format_func=lambda s: {
    #         "upcoming": "Next 48 h (live forecast)",
    #         "replay": "Historical replay (with truth)",
    #         "live": "Live (AeroDataBox + NWS)"}[s],
    #     help="Next 48 h = ORD's recent schedule projected onto the real "
    #          "upcoming dates + live NWS forecasts. Replay = a real labelled "
    #          "window with ground truth. Live = real schedule via AeroDataBox "
    #          "(needs an AeroDataBox key).")
    
    source = st.radio(
        "Source",
        options=["live"],
        format_func=lambda s: {
            "live": "Live (AeroDataBox + NWS)"}[s],
        help="window with ground truth. Live = real schedule via AeroDataBox "
             "(needs an AeroDataBox key).")

    api_key = None
    replay_now = None
    weather = True
    provider = os.environ.get("AERODATABOX_PROVIDER", "apimarket").lower()
    if source == "live":
        provider = st.selectbox(
            "Marketplace",
            options=["apimarket", "rapidapi"],
            index=(1 if provider == "rapidapi" else 0),
            format_func=lambda p: {"apimarket": "API.Market",
                                   "rapidapi": "RapidAPI"}[p],
            help="Which AeroDataBox marketplace issued your key. The key and "
                 "the marketplace must match.")
        if env_key:
            st.success("AeroDataBox key loaded (aerodatabox.key / env).")
        api_key = st.text_input(
            "AeroDataBox key", value=env_key or "", type="password",
            help="Auto-loaded from aerodatabox.key or $AERODATABOX_KEY; "
                 "override here if needed.") or env_key
    elif source == "replay":
        replay_now = st.text_input(
            "Replay 'now' (UTC)", value=str(DEMO_DEFAULT_NOW),
            help="Any timestamp inside the labelled data range.")
    else:  # upcoming
        weather = st.checkbox(
            "Fetch live NWS forecasts", value=True,
            help="Off = skip the ~150-airport forecast fetch (faster, but "
                 "weather features become NaN).")

    if st.button("🔄 Refresh data", width="stretch", type="primary"):
        st.session_state.refresh_token += 1
        _predict.clear()

    st.caption("Refresh refetches from the prediction APIs and re-scores. "
               "Search and filters apply instantly without a refetch.")


# --------------------------------------------------------------------------- #
# Fetch + score (cached)
# --------------------------------------------------------------------------- #

st.title("✈️ ORD Departure-Disruption Forecast — next 48 hours")

try:
    data = _predict(source, api_key, replay_now, weather, provider,
                    st.session_state.refresh_token)
except Exception as err:  # noqa: BLE001 - surface any serving error in the UI
    st.error(f"Could not produce predictions: {err}")
    if source == "live":
        msg = str(err).lower()
        if "429" in msg or "quota" in msg or "rate limit" in msg:
            st.info("AeroDataBox is rate- or quota-limited for this key. "
                    "Switch to **Next 48 h** to keep using live NWS weather "
                    "with the projected ORD schedule until the quota resets.")
        else:
            st.info("Live mode needs a valid AeroDataBox API key. "
                    "Switch to **Next 48 h** or **Historical replay** in the "
                    "sidebar to explore without one.")
    st.stop()

flights: pd.DataFrame = data["flights"]
as_of = data["as_of"]

# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted 48 h disruption rate", f"{data['predicted_rate']:.1%}")
c2.metric("Flights in window", f"{data['n_flights']:,}")
hi = int((flights["risk"] == "High").sum())
c3.metric("High-risk flights", f"{hi:,}",
          help="P(disrupted) ≥ 60%")
if data["actual_rate"] is not None:
    delta = data["predicted_rate"] - data["actual_rate"]
    c4.metric("Actual rate (demo truth)", f"{data['actual_rate']:.1%}",
              delta=f"{delta:+.1%} pred−actual", delta_color="off")
else:
    c4.metric("Source", data["source"].upper())

as_of_ct = pd.Timestamp(as_of).tz_convert("America/Chicago")
st.caption(f"As of **{as_of_ct:%Y-%m-%d %H:%M} CT**")

# --------------------------------------------------------------------------- #
# Trajectory + risk mix
# --------------------------------------------------------------------------- #

left, right = st.columns([3, 2])

risk_order = ["High", "Elevated", "Moderate", "Low"]

with left:
    st.subheader("Predicted disruption rate by departure hour (CT)")
    by_hour = data["by_hour"].set_index("dep_hour")["predicted_rate"]
    st.line_chart(by_hour, height=260, y_label="P(disrupted)")
with right:
    st.subheader("Risk mix")

    risk_order = ["High", "Elevated", "Moderate", "Low"]

    mix = (
        flights["risk"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("risk")
        .reset_index(name="count")
    )

    chart = (
        alt.Chart(mix)
        .mark_bar(color="#d62728")
        .encode(
            x=alt.X("count:Q", title=None),
            y=alt.Y("risk:N", sort=risk_order, title=None),
            tooltip=[
                alt.Tooltip("risk:N", title="Risk level"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(chart, use_container_width=True)

    band_defs = "  ·  ".join(
        f"**{name}** {int(lo * 100)}–{int(min(hi, 1.0) * 100)}%"
        for lo, hi, name in RISK_BANDS
    )
    st.caption("Levels by P(disrupted):  " + band_defs)

# --------------------------------------------------------------------------- #
# Search + filters  (instant; no refetch)
# --------------------------------------------------------------------------- #

st.subheader("Flights")

f1, f2, f3 = st.columns([2, 1, 1])
query = f1.text_input("🔎 Search", placeholder="flight #, airline, dest, city…")
airlines = sorted(flights["airline"].dropna().unique().tolist())
dests = sorted(flights["dest"].dropna().unique().tolist())
sel_airlines = f2.multiselect("Airline", airlines)
sel_risk = f3.multiselect("Risk level", RISK_ORDER)

g1, g2, g3 = st.columns([1, 1, 2])
sel_dests = g1.multiselect("Destination", dests)
min_p = g2.slider("Min P(disrupted)", 0.0, 1.0, 0.0, 0.05)
lead_lo, lead_hi = g3.slider("Lead time (hours to departure)", 0.0, 48.0,
                             (0.0, 48.0), 1.0)

view = flights
if query:
    q = query.strip().lower()
    hay = (view["flight"].fillna("") + " " + view["airline"].fillna("") + " "
           + view["dest"].fillna("") + " " + view["dest_city"].fillna(""))
    view = view[hay.str.lower().str.contains(q, regex=False)]
if sel_airlines:
    view = view[view["airline"].isin(sel_airlines)]
if sel_dests:
    view = view[view["dest"].isin(sel_dests)]
if sel_risk:
    view = view[view["risk"].isin(sel_risk)]
view = view[(view["p_disrupted"] >= min_p)
            & (view["lead_h"] >= lead_lo) & (view["lead_h"] <= lead_hi)]

st.caption(f"Showing **{len(view):,}** of {len(flights):,} flights"
           + (f"  ·  filtered predicted rate **{view['p_disrupted'].mean():.1%}**"
              if len(view) else ""))

display = view.rename(columns={
    "flight": "Flight", "airline": "Airline", "dest": "Dest",
    "dest_city": "City", "dep_ct": "Departs (CT)", "lead_h": "Lead (h)",
    "p_disrupted": "P(disrupted)", "risk": "Risk"})
col_config = {
    "P(disrupted)": st.column_config.ProgressColumn(
        "P(disrupted)", min_value=0.0, max_value=1.0, format="%.2f"),
    "Departs (CT)": st.column_config.DatetimeColumn(
        "Departs (CT)", format="MMM DD, HH:mm"),
    "Lead (h)": st.column_config.NumberColumn("Lead (h)", format="%.1f"),
}
show_cols = ["Flight", "Airline", "Dest", "City", "Departs (CT)",
             "Lead (h)", "P(disrupted)", "Risk"]
if "actual_disrupted" in view.columns:
    display = display.rename(columns={"actual_disrupted": "Actual"})
    show_cols.append("Actual")

st.dataframe(display[show_cols], width="stretch", hide_index=True,
             column_config=col_config, height=460)

st.download_button(
    "Download filtered flights (CSV)",
    view.to_csv(index=False).encode(),
    file_name=f"ord_disruption_{as_of:%Y%m%d_%H%M}.csv", mime="text/csv")
