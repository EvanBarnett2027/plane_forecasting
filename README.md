# plane_forecasting
DE300 Final Project — predicting flight disruptions for departures from ORD.

## Historical flight-data ingestion (BTS On-Time Performance)

### Data source

Official **BTS / TranStats Airline On-Time Performance Data** (table
`T_ONTIME_REPORTING`). The pipeline downloads the official pre-zipped
monthly files BTS publishes for programmatic access:

```
https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip
```

The TranStats web download form just builds these same pre-zipped files, so
this is the official endpoint and needs no form parameters. Each zip holds
one nationwide CSV per month; the `Origin == ORD` filter is applied
client-side. No Kaggle or third-party mirror is used.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Ingest raw monthly files

```bash
python -m src.ingest_bts_ord --start-year 2020 --start-month 1 --origin ORD
```

- Loops month-by-month from 2020-01 through the latest available BTS month.
- Months BTS has not published yet are detected (HTTP 404 / non-zip) and
  reported as `unavailable`; add `--stop-on-unavailable` to halt at the
  first gap instead of probing every remaining month.
- Writes one file per month to `data/raw/bts_ontime_ord/YYYY_MM.parquet`
  (use `--format csv` for CSV).
- Existing monthly files are skipped unless `--force` is passed.

Useful options: `--end-year`, `--end-month` (default: current month),
`--origin`, `--raw-dir`, `--timeout`. See `--help` for all flags.

### 2. Combine + clean

```bash
python -m src.clean_bts_ord
```

This step:

- concatenates every monthly raw file,
- standardises all column names to `snake_case`,
- parses `flight_date` as datetime,
- builds `scheduled_dep_datetime` / `scheduled_arr_datetime` from
  `crs_dep_time` / `crs_arr_time` in **America/Chicago** (ORD local time),
- **preserves cancelled and diverted flights** and does **not** drop rows
  for missing actual departure/arrival fields (those are blank by
  definition on cancelled flights),
- adds the labels below,
- runs a validation report (see *Validation output*),

and writes the processed dataset to:

```
data/processed/ord_departures_bts_2020_present.parquet
```

Rerun anytime; to force a fresh download of months already on disk:

```bash
python -m src.ingest_bts_ord --force      # re-download every month
python -m src.clean_bts_ord               # rebuild the processed parquet
```

### Labels

| Column            | Meaning                                                              |
|-------------------|----------------------------------------------------------------------|
| `delayed`         | 1 if `dep_del15 == 1` OR `arr_del15 == 1` (15+ min late dep or arr)   |
| `disrupted`       | 1 if delayed OR `cancelled == 1` OR `diverted == 1`                   |
| `canceled`        | = `cancelled` (flight was cancelled)                                  |
| `diverted_flag`   | = `diverted` (flight was diverted)                                    |
| `weather_related` | 1 if `weather_delay > 0` OR `cancellation_code == 'B'` (BTS weather)  |

### Feature-leakage guidance

Only **pre-flight** columns are safe as model features. **Post-flight
outcome** columns are kept for labels/diagnostics but must NOT be used as
features (they are realised only after the flight operates):

- **Safe features:** date/calendar parts, `reporting_airline`,
  `tail_number`, `flight_number_reporting_airline`, `origin*`, `dest*`,
  `crs_dep_time`, `crs_arr_time`, `crs_elapsed_time`, `distance`,
  `scheduled_dep_datetime`, `scheduled_arr_datetime`.
- **Post-flight (do NOT use as features):** `dep_time`, `dep_delay*`,
  `dep_del15`, `taxi_out`, `wheels_off/on`, `taxi_in`, `arr_time`,
  `arr_delay*`, `arr_del15`, `*_delay_groups`, `cancelled`,
  `cancellation_code`, `diverted`, `actual_elapsed_time`, `air_time`,
  `carrier/weather/nas/security/late_aircraft_delay`.

The authoritative lists are `SAFE_FEATURE_COLS` and
`POST_FLIGHT_OUTCOME_COLS` in [src/clean_bts_ord.py](src/clean_bts_ord.py).

### Validation output

`clean_bts_ord` prints a report that: asserts every row has
`origin == ORD`, lists row counts per year-month, reports the disruption /
delay / cancellation / diversion / weather-related rates, summarises
missingness for important columns, and asserts there are no duplicate
flight records on
`(flight_date, reporting_airline, flight_number_reporting_airline, origin,
dest, crs_dep_time)` (exact-duplicate rows are dropped first).

## Historical weather ingestion (IEM ASOS / METAR)

### Data source

Official **Iowa Environmental Mesonet (IEM)** ASOS download service, which
rehosts the authoritative NWS/FAA ASOS observations for O'Hare:

```
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
```

Observations are sub-hourly (routine METAR plus SPECI specials). The
pipeline mirrors the flight one: one raw CSV per month, cached, with
retry/backoff, then a combine/clean/validate step. Default range is
2020-01 through the current month (IEM is near-real-time — no publication
lag, so weather runs all the way to *today*, capping the partial month).

The scripts are station-agnostic: `--station ORD` (default) for the origin
pipeline, `--stations A,B,C` or `--all` for many stations at once (used for
the destination-weather step below).

### 1. Ingest raw monthly files

```bash
python -m src.ingest_iem_asos --start-year 2020 --start-month 1 --station ORD
```

Writes `data/raw/iem_asos/{STATION}/YYYY_MM.csv`; existing months are
skipped unless `--force`. Multi-station mode batches up to `--batch-size`
stations per IEM request via the API's repeated `station=` parameter. See
`--help` for `--stations`, `--stations-file`, `--end-year`, `--end-month`,
`--raw-root`, `--timeout`.

### 2. Combine + clean

```bash
python -m src.clean_iem_asos --station ORD
```

Keeps the core meteorological columns, parses `valid` as UTC, coerces
numerics, and **aggregates the sub-hourly METAR/SPECI reports into one
row per UTC hour with a column-appropriate aggregator** —
temperature/dewpoint/humidity appear only on the routine METAR (~10% of
raw rows) while wind, visibility and gust update every ~5 minutes, so a
single-row pick discards the extremes that matter for disruption
modelling. Per-column rules (full list in `HOURLY_AGG` in
[src/clean_iem_asos.py](src/clean_iem_asos.py)):

- **max:** `gust`, `peak_wind_gust`, `p01i`, `ice_accretion_1hr` — the
  worst-in-hour is the signal.
- **min:** `vsby`, `skyl1` — worst visibility / lowest ceiling.
- **mean:** `tmpf`, `dwpf`, `relh`, `feel`, `alti`, `mslp`, `sknt` —
  stable instant readings.
- **circular mean** (unit-vector via atan2): `drct` — degrees don't
  average linearly.
- **union of distinct tokens:** `wxcodes` — keep every weather code that
  occurred.
- **worst sky-cover code** (`VV > OVC > BKN > SCT > FEW > SKC/CLR`):
  `skyc1`.
- **first / last:** `station`, `snowdepth`.

An `n_reports` diagnostic column records how many sub-hourly reports fed
each hourly row (median 13 ≈ 5-min cadence). Validation asserts the
station, prints hourly coverage vs. expected, and missingness. Output:

```
data/processed/weather/{STATION}_iem_2020_present.parquet
```

Force a full refresh the same way as flights:

```bash
python -m src.ingest_iem_asos --force
python -m src.clean_iem_asos --station ORD
```

## Destination-airport weather and the joined dataset

To predict disruption we want weather at both endpoints: ORD at scheduled
departure, and the destination airport at scheduled arrival. The same IEM
scripts handle both — destinations are just additional stations.

### 1. Backfill weather for every destination

```bash
python -m src.ingest_dest_weather --also-clean
```

This reads the processed flight parquet, extracts every distinct `dest`
(plus ORD), and calls the multi-station batched IEM ingest for the full
2020-present range. `--also-clean` then invokes the cleaner over every
station and produces a combined long table keyed `(station, valid_utc)`:

```
data/processed/dest_weather_iem_2020_present.parquet
```

Useful flags: `--top 50` (just the busiest destinations), `--no-origin`
(skip ORD), `--force`.

### 2. Join flights with origin + destination weather

```bash
python -m src.join_flights_weather
```

Builds the final modelling table:

```
data/processed/ord_flights_with_weather_2020_present.parquet
```

Every flight row gets:
- `origin_wx_*` — ORD weather at the **scheduled-departure UTC hour**.
- `dest_wx_*`   — destination weather at the **scheduled-arrival UTC hour**.

**Timezone-correct join.** `scheduled_arr_datetime` in the flight cleaner
was built from `crs_arr_time` (destination-local clock) and labelled
America/Chicago; for non-CT destinations its UTC instant is off by 1–3
hours. The join instead derives `scheduled_arr_utc` from
`scheduled_dep_utc + crs_elapsed_time`, which is correct regardless of
the destination's timezone (no airport→tz lookup required).

## Train-time weather-forecast noise injection

At inference the model will see weather **forecasts**, not observations.
Training on clean observations and deploying on noisy forecasts is a
train/test mismatch. [src/weather_noise.py](src/weather_noise.py) adds
lead-time-aware synthetic noise to the `origin_wx_*` / `dest_wx_*`
columns during training only, as a step in the training pipeline.

**Methodology.** σ per column is estimated from *persistence residuals*
in the observed weather:

```
σ_v(L) = std( v(t) − v(t − L) )   pooled across stations
```

This is a conservative upper bound on real forecast error (good
forecasts beat persistence). Per-column noise kinds are chosen to keep
samples physically valid:

- symmetric numerics → additive Gaussian
- non-negative (wind, gust) → Gaussian clipped at 0
- circular (`drct`) → Gaussian then mod 360
- bounded (`vsby`) → Gaussian clipped to `[0, 10]`
- zero-inflated (precip-like) → log-normal multiplicative on `>0` values
- categorical / text / diagnostic → left alone

NaN inputs are preserved (missingness is a signal). Non-weather columns
are never touched.

**Fit the σ table once:**

```bash
python -m src.weather_noise fit \
    --weather data/processed/dest_weather_iem_2020_present.parquet \
    --out data/processed/weather_noise_sigmas.json
```

**Use in a training pipeline:**

```python
from src.weather_noise import WeatherNoiseInjector
noise = WeatherNoiseInjector.load(SIGMA_PATH, seed=42).with_scale(1.0)
X_train = noise.transform(
    X_train,
    lead_origin_h=df_train["lead_origin_h"],
    lead_dest_h=df_train["lead_dest_h"],
)
# X_val / X_test left alone, or noise with a fixed seed for deployment-realism eval
```

Sweep `scale ∈ {0.25, 0.5, 1.0, 1.5}` on the validation set; treat as a
hyperparameter. Tests: `python -m pytest tests/`.

**Diurnal handling.** Persistence residuals for temperature-cycle
variables (`tmpf`, `relh`, `feel`) are contaminated by the daily cycle
at non-24h leads — e.g. an L=12h shift compares noon to midnight, not
true forecast skill. These columns (listed in `DIURNAL_COLS`) are
therefore fit only at the diurnal-aligned leads `{24, 48}`; shorter
leads clamp to σ(24) at transform time via `np.interp`. Non-diurnal
columns (wind, visibility, pressure, gust, precip) keep the full lead
grid because their short-lead residuals are honest forecast-error
proxies. Trade-off: σ(24) is a *conservative* (over-noisy) estimate for
very short leads on diurnal variables. For a "2 days out" model where
most flights live in L=12-48, that's fine; the `scale` hyperparameter
tunes the rest.

## Model training + evaluation pipeline

End-to-end script that trains a per-flight disruption classifier focused on
the **2-day-out** use case and reports both per-flight and per-occasion
performance metrics: [src/training_pipeline.py](src/training_pipeline.py).

### What it does

1. **Load + filter**. Reads `data/processed/ord_flights_with_weather_2020_present.parquet`,
   drops the pre-2021 COVID era.
2. **Time-based split** (no random shuffling — disruption is heavily
   autocorrelated across hours, so a random split would leak):

   ```
   train       2021-01 .. 2025-09
   validation  2025-10 .. 2025-12
   test        2026-01 .. (latest)   sealed
   ```

3. **Sample per-flight lead times** `L_origin ~ Uniform(0, 48 h)` on
   train and validation (so the model learns lead-conditional decay of
   weather signal). `L_dest = L_origin + crs_elapsed_h`.
4. **Inject noise** on every `*_wx_*` column via
   `WeatherNoiseInjector.transform(..., lead_origin_h, lead_dest_h)`.
   **Both training AND test use noise** so eval reflects deployment
   conditions, not a clean-data upper bound.
5. **Feature engineering**: calendar (cyclic month / hour), schedule,
   route, lead times, all weather numerics. Categorical columns
   (`reporting_airline`, `dest`, `dest_state`) use pandas Categorical
   dtype so sklearn's `HistGradientBoostingClassifier` picks them up
   natively; unseen levels on val/test become NaN, which the model also
   handles natively.
6. **Fit** `HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.05,
   max_iter=500, early_stopping=True, ...)`.
7. **Build test occasions**: every 6 h (the GFS refresh cadence) within
   the test period, gather all flights scheduled in the next 48 h, use
   each flight's **real** lead at that issue time, inject noise.
8. **Metrics** at two levels:
   - **Per flight** (across all test (occasion, flight) rows): AUC-ROC,
     AUC-PR, Brier, base rate.
   - **Per occasion** (the actual deliverable): MAE / RMSE / signed bias
     on aggregate disruption rate, Pearson correlation, severe-window
     precision/recall (top-decile of actual rates).
9. **Artifacts** under `artifacts/disruption_model/`:
   - `model.joblib` — fitted model + frozen category vocabularies
   - `metrics.json` — every metric above
   - `per_occasion.csv` — one row per test occasion: `t_issue`,
     `actual_rate`, `predicted_rate`, `abs_error`, `signed_error`,
     `n_flights`

### Run

```bash
# fast iteration: 10% train subsample
python -m src.training_pipeline --quick

# full run
python -m src.training_pipeline

# noise-scale sweep (treat as a hyperparameter)
for s in 0.5 1.0 1.5; do
    python -m src.training_pipeline --noise-scale $s \
        --artifacts-dir artifacts/disruption_scale_$s
done
```

### Latest results (full run, 1.28 M train rows)

| | |
|---|---|
| Test base rate | **0.360** (regime shift vs train ≈ 0.25 — 2026 is busier/worse) |
| AUC-ROC | 0.716 |
| AUC-PR | 0.612 (lift ≈ 1.7× over base) |
| Brier | 0.202 |
| **Per-occasion MAE on 48 h rate** | **0.088** |
| Per-occasion RMSE | 0.107 |
| Per-occasion signed bias | **−0.069** (under-predicts) |
| Pearson correlation (pred vs actual rate) | 0.88 |
| Severe-window (top-10%) precision / recall | 0.67 / 0.67 |

### Walk-forward retraining (expanding window)

The pipeline also supports **walk-forward evaluation with an expanding
train window** — the production-realistic setup where the model is
re-trained on a fixed cadence and each retrain sees more data than the
last. Toggle with `--walk-forward`:

```bash
python -m src.training_pipeline --walk-forward --model lgbm \
    --retrain-every-days 30 --first-test-start 2025-07-01
```

At step `i` with boundary `T_i`:

- **train** on every flight scheduled before `T_i` (window *grows* each step),
- carve the last `--val-days` (default 30) of train as a time-ordered
  val slice for early stopping,
- **test** per-occasion 48 h forecasts for issue times in
  `[T_i, T_i + retrain_every_days)`.

Useful flags: `--model {hgb,lgbm}` (LightGBM strongly recommended for
walk-forward — it's faster and the production model class for this
pipeline), `--retrain-every-days N`, `--first-test-start YYYY-MM-DD`,
`--val-days N`.

Three artifacts are written:

```
artifacts/disruption_model/
  walkforward_step_metrics.csv      # one row per retrain step
  walkforward_per_occasion.parquet  # one row per (step, occasion)
  walkforward_per_flight.parquet    # one row per (step, occasion, flight)
```

The dedicated EDA notebook is
[notebooks/walkforward_eda.ipynb](notebooks/walkforward_eda.ipynb) —
metric trajectories, calibration bias over time, per-step calibration
plot, per-occasion error distributions, per-step predicted-vs-actual
scatter grid, and Brier stratified by lead bin per step.

### EDA + multi-model comparison notebook

[notebooks/disruption_model_eda.ipynb](notebooks/disruption_model_eda.ipynb)
walks through the data and compares three models on the same
noise-injected train/test split: **logistic regression** (baseline),
**LightGBM**, and **HistGradientBoosting** (the production model). It
includes time-series of disruption rate, per-carrier and per-destination
breakdowns, weather-vs-disruption distributions, per-flight and
per-occasion metrics, a calibration plot overlay across the three models,
predicted-vs-actual occasion scatterplots, and **LightGBM feature
importance** in three views (split count, gain, and permutation
importance on the test set). Headline:

```
          AUC-ROC  AUC-PR  Brier
Logistic   0.666   0.529  0.220   <- linear; meaningfully weaker
LightGBM   0.720   0.618  0.200   <- best (marginally)
HistGBT    0.716   0.612  0.202   <- production; essentially tied
```

Cell outputs are baked in, so opening the notebook shows results
immediately. To re-execute:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 notebooks/disruption_model_eda.ipynb
```

### Engineered operational + trend features

Two cheap, leak-free features address the systematic under-prediction
bias the model showed before:

- **`trailing_disruption_rate_{6,24,72,168}h`** — disruption rate over
  ORD flights whose *scheduled arrival* falls in
  `[t_now − W, t_now)`, where `t_now = scheduled_dep − lead_origin_h`.
  Captures cascading delays, weather-aftermath state, ATC/crew
  disruption, holiday spikes.

  **On leakage.** The natural-sounding "trailing rate of flights
  scheduled to depart in the last W hours" leaks: it includes flights
  whose scheduled arrival is *after* `t_now` (still in the air at
  prediction time), whose arrival-delay labels aren't yet realised in
  production. Indexing by **scheduled arrival** with a strict-less-than
  bound fixes this — every flight in the window has had time to
  realise its full `disrupted` label before `t_now`. The fix is
  verified by a smoke test that puts the only disrupted flight at the
  window edge and checks it's excluded until its arrival passes.
  Empirically the leak cost ≈ 0 (v3 leak-free is within 0.0001 MAE
  of leaky v2) — most of the trailing-rate signal lives in flights
  that fully landed hours ago, not in-flight ones.
- **`days_since_2021`** — linear time trend. Lets the model lean into
  the post-COVID-recovery upward drift instead of regressing to the
  long-run average.

In production these would be sourced from a real-time ops feed
(ASDI/ASPM/airline ops); during backtest we compute them from BTS
labels available strictly before `t_now`. Implementation:
[src/training_pipeline.py:add_trailing_disruption_features](src/training_pipeline.py).
Vectorised via prefix sums, so all four lookback windows for ~1.4 M
flights complete in well under a second per walk-forward step.

Walk-forward LightGBM, **before vs after** adding these features
(monthly retrains, 9 steps from 2025-07 → 2026-02):

| metric                 | before | after  | change |
|---|---:|---:|---:|
| mean per-occasion MAE  | 0.0910 | 0.0724 | **−20.4 %** |
| mean &#124;bias&#124;  | 0.0740 | 0.0390 | **−47.3 %** |
| mean AUC-ROC           | 0.6878 | 0.6940 | +0.006 |
| Dec-2025 step MAE      | 0.186  | 0.106  | **−43 %** (holiday spike) |

The AUC barely moved, which is the correct interpretation: the model
already *ranked* flights well, it just couldn't tell **what regime**
the airport was in. Several signed-bias values flipped from stubbornly
negative to small and zero-centred — exactly the calibration shape a
post-hoc isotonic recalibrator can finish cleaning up.

**Known issues** worth iterating on:

- **Systematic under-prediction.** The −7 pp bias is almost certainly a
  regime shift between train (2021–2025, post-COVID recovery) and test
  (2026, busier era with higher base disruption). Quick fixes: (a) fit a
  post-hoc isotonic calibrator on validation; (b) include a recency
  weight in training; (c) add a "year" or "trailing-30d-disruption-rate"
  feature. The Pearson r of 0.88 says the *ranking* is good — only the
  level is off.
- **Severe-window recall is only 0.67** at the top decile — the model
  catches two out of three of the worst 48 h windows. Lifting this is
  where calibration + a `trailing_disruption_rate` operational feature
  would help most.

### Layout

```
src/ingest_bts_ord.py        # flights: download + per-month raw save (CLI)
src/clean_bts_ord.py         # flights: combine + clean + label + validate
src/ingest_iem_asos.py       # weather: multi-station, batched IEM ingest
src/clean_iem_asos.py        # weather: per-station clean + combined long table
src/ingest_dest_weather.py   # orchestrator: weather for every dest + ORD
src/join_flights_weather.py  # join flights + origin/dest weather (UTC-correct)
src/weather_noise.py         # train-time noise injection (CLI: fit)
src/training_pipeline.py     # split + features + noise + fit + occasion eval
tests/test_weather_noise.py  # pytest suite for the noise injector
notebooks/disruption_model_eda.ipynb  # EDA + LR vs LightGBM vs HistGBT + calibration
notebooks/walkforward_eda.ipynb       # walk-forward (expanding window) metric trajectories

data/raw/bts_ontime_ord/YYYY_MM.parquet                       # raw flights
data/raw/iem_asos/{STATION}/YYYY_MM.csv                       # raw weather
data/processed/ord_departures_bts_2020_present.parquet        # processed flights
data/processed/weather/{STATION}_iem_2020_present.parquet     # per-station weather
data/processed/dest_weather_iem_2020_present.parquet          # combined long weather
data/processed/ord_flights_with_weather_2020_present.parquet  # joined final table
data/processed/weather_noise_sigmas.json                      # noise σ table
artifacts/disruption_model/
  model.joblib, metrics.json, per_occasion.csv     (single-shot)
  walkforward_step_metrics.csv, walkforward_per_{occasion,flight}.parquet
```

`data/` is git-ignored — it is large and fully reproducible from the
scripts above.
