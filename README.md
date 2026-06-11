# Plane Forecasting

DE300 Final Project — predicts 48-hour flight disruption probability for ORD departures.

A LightGBM classifier trained on 5 years of BTS flight records and IEM ASOS weather
observations, served through a Streamlit dashboard that shows per-flight
`P(disrupted)` for the upcoming 48 h.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

Three helper scripts wrap the full workflow end to end. Run them in order:

```bash
./download_data.sh   # 1. download + build the dataset (~30-60 min, network-heavy)
./train_model.sh     # 2. train the production model + run the recent-week backtest
./start_app.sh       # 3. launch the Streamlit dashboard
```

Each script activates nothing globally — it calls the project's `.venv`
directly, streams per-step logs to `logs/`, and shows a live elapsed-time /
ETA readout. On any failure it prints the tail of that step's log and stops.
`download_data.sh` and `train_model.sh` are one-time setup; re-run
`start_app.sh` whenever you want the dashboard.

The sections below document the individual commands each script runs, in case
you want to run or customise a single step.

---

## API Keys

| Key | Used for | Required? |
|---|---|---|
| `AERODATABOX_KEY` | Live ORD departure schedule via AeroDataBox (API.Market or RapidAPI) | Only for the **Live** dashboard source |

No key is needed for NWS weather forecasts or the default "Next 48 h" projection mode.

Store as an environment variable or place the key (one line, no quotes) in
`aerodatabox.key` at the project root.

Two marketplace gateways are supported:

```bash
# API.Market (default)
AERODATABOX_KEY=<key> python -m src.live.live_flights schedule

# RapidAPI
AERODATABOX_PROVIDER=rapidapi AERODATABOX_KEY=<key> python -m src.live.live_flights schedule
```

---

## Data Pipeline

> **Shortcut:** `./download_data.sh` runs every step in this section in order.
> The individual commands below are for running or customising a single step.

Run these steps in order once to build the full training dataset.

### 1. Historical flight data (BTS On-Time Performance)

```bash
# Download monthly raw files from BTS TranStats (2020–present)
python -m src.data.ingest_bts_ord --start-year 2020 --start-month 1

# Combine, clean, add disruption labels, validate
python -m src.data.clean_bts_ord
# → data/processed/ord_departures_bts_2020_present.parquet
```

### 2. Historical weather data (IEM ASOS / METAR)

```bash
# Download hourly METAR observations for ORD
python -m src.data.ingest_iem_asos --start-year 2020 --start-month 1 --station ORD
python -m src.data.clean_iem_asos --station ORD

# Download and clean weather for every destination airport
python -m src.data.ingest_dest_weather --also-clean
# → data/processed/dest_weather_iem_2020_present.parquet
```

### 3. Join flights with weather

```bash
python -m src.data.join_flights_weather
# → data/processed/ord_flights_with_weather_2020_present.parquet
```

`data/` is git-ignored — all files are reproducible from the scripts above.

---

## Model Training

> **Shortcut:** `./train_model.sh` trains the production model and runs the
> recent-week backtest (it checks the dataset exists first). The commands
> below cover that plus the experimental pipeline.

### Fit the weather-noise sigma table (once)

```bash
python -m src.model.weather_noise fit \
    --weather data/processed/dest_weather_iem_2020_present.parquet \
    --out data/processed/weather_noise_sigmas.json
```

### Train the production model

```bash
python -m src.model.train_production_model
# → artifacts/production_model/model.joblib + metadata.json
```

### Experimental training and evaluation

```bash
python -m src.model.training_pipeline          # full run
python -m src.model.training_pipeline --quick  # 10% subsample for fast iteration

# Walk-forward retraining (expanding window, LightGBM)
python -m src.model.training_pipeline --walk-forward --model lgbm \
    --retrain-every-days 30 --first-test-start 2025-07-01
```

---

## How the Model Works

**Task.** At each 6-hour GFS refresh "occasion", score every ORD departure
scheduled in the next 48 h with `P(disrupted)`, then aggregate to a
48 h disruption-rate forecast.

**Features.**
- Calendar: cyclic month/hour encoding, `days_since_2021` (linear time trend)
- Schedule: airline, destination, route, flight number, `crs_elapsed_time`
- Weather at ORD (departure hour) and destination (arrival hour): wind, gusts,
  visibility, ceiling, temperature, precipitation, pressure — ~25 wx columns per endpoint
- Trailing airport disruption rates over 6 h / 24 h / 72 h / 168 h lookback windows
  (indexed by scheduled arrival so they never look forward from the issue time)
- Lead time from issue moment to scheduled departure

**Weather noise injection.** At training time the model sees clean historical
observations; at inference it sees NWS forecasts. To close this train/serve gap,
lead-time-aware Gaussian noise is added to every `*_wx_*` column during training:

$$
\sigma(L) = \operatorname{std}( v(t) − v(t − L) )   \quad \text{pooled across stations}
$$

Noise kind is matched to the variable type (additive, non-negative clipped,
circular, bounded, log-normal for precipitation). `scale` is a hyperparameter
swept on the validation set.

**Model.** LightGBM (`LGBMClassifier`, `binary` objective). Trained with a
time-based split (no random shuffling — disruption is autocorrelated across
hours):

```
train       2021-01 – 2025-09
validation  2025-10 – 2025-12   (early stopping)
test        2026-01 – latest
```

**Production model differences.** The dashboard serves a separate production
artifact that (1) drops `n_reports` (a METAR diagnostic unavailable in NWS
forecasts) and (2) trains on all labelled data with the last 30 days as the
validation slice.

**Latest results** (walk-forward LightGBM, 9 monthly retrains 2025-07 → 2026-02):

| metric | value |
|---|---|
| Per-occasion MAE (48 h rate) | 0.072 |
| Mean \|bias\| | 0.039 |
| AUC-ROC | 0.694 |
| Pearson r (pred vs actual rate) | — |

---

## Starting the Dashboard

> **Shortcut:** `./start_app.sh` launches the dashboard (pass
> `AERODATABOX_KEY=<key> ./start_app.sh` to enable the Live source).

```bash
# Default mode: projected next-48 h schedule + live NWS weather (no key needed)
.venv/bin/streamlit run dashboard/app.py

# Live mode: real AeroDataBox schedule + live NWS weather
AERODATABOX_KEY=<key> .venv/bin/streamlit run dashboard/app.py
```

Three data sources are available in the dashboard UI:

| Source | Flights | Weather | Ground truth |
|---|---|---|---|
| **Next 48 h** (default) | Recent schedule projected onto upcoming dates (same weekday) | Live NWS forecasts | — |
| **Historical replay** | Real labelled 48 h window | Joined observations | Predicted vs actual |
| **Live** | Real AeroDataBox schedule | Live NWS forecasts | — |

---

## Tests

```bash
python -m pytest tests/
```

---

## Directory Structure

```
plane_forecasting/
├── download_data.sh                   # Build the full dataset (runs the data pipeline)
├── train_model.sh                     # Train the production model + recent-week backtest
├── start_app.sh                       # Launch the Streamlit dashboard
├── src/
│   ├── data/                          # Historical data ingestion and processing
│   │   ├── ingest_bts_ord.py          # Download BTS on-time flight records month-by-month
│   │   ├── clean_bts_ord.py           # Combine, clean, label, and validate flight data
│   │   ├── ingest_iem_asos.py         # Download IEM ASOS weather observations (multi-station)
│   │   ├── clean_iem_asos.py          # Aggregate sub-hourly METARs to one row per UTC hour
│   │   ├── ingest_dest_weather.py     # Orchestrate weather ingestion for all destination airports
│   │   └── join_flights_weather.py    # Join flights with origin + destination weather (UTC-correct)
│   ├── model/                         # Model training, noise injection, and evaluation
│   │   ├── weather_noise.py           # Lead-time-aware noise injection (fit sigma table + transform)
│   │   ├── training_pipeline.py       # Split → features → noise → fit → per-occasion evaluation
│   │   ├── train_production_model.py  # Train the deployable LightGBM (no n_reports)
│   │   ├── evaluate_recent_week.py    # Deployment-realism backtest on the most recent week
│   │   └── validate_calibration_full.py  # CatBoost + isotonic calibration validation
│   ├── live/                          # Real-time data collection for the dashboard
│   │   ├── live_flights.py            # AeroDataBox next-48 h schedule + recent completed flights
│   │   ├── live_weather.py            # NWS hourly forecasts mapped to the model weather schema
│   │   ├── trailing_state.py          # As-of-now trailing disruption rate (operational feature)
│   │   └── backfill_recent_cache.py   # Seed the rolling flight cache (rate-limited)
│   └── serve.py                       # Prediction API: collectors + model → per-flight P(disrupted)
├── dashboard/
│   └── app.py                         # Streamlit dashboard (refresh, search, filter, replay mode)
├── notebooks/
│   ├── disruption_model_eda.ipynb     # EDA + logistic vs LightGBM vs HistGBT comparison
│   ├── production_model.ipynb         # Production model analysis and calibration plots
│   ├── walkforward_eda.ipynb          # Walk-forward metric trajectories over time
│   └── lightgbm_investigation.ipynb   # LightGBM hyperparameter and calibration investigation
├── tests/
│   ├── test_weather_noise.py          # Unit tests for the noise injector
│   └── test_live_collection.py        # Unit tests for the live collectors
├── artifacts/
│   ├── disruption_model/              # Experimental model artifacts (model.joblib, metrics.json)
│   └── production_model/              # Deployable bundle (model.joblib, model.txt, metadata.json)
├── data/                              # Raw + processed data (git-ignored, reproducible from scripts)
└── requirements.txt
```
