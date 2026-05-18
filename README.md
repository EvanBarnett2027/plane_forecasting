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

### Layout

```
src/ingest_bts_ord.py   # download + per-month raw save (CLI)
src/clean_bts_ord.py    # combine + clean + label + validate (CLI)
data/raw/bts_ontime_ord/YYYY_MM.parquet              # raw monthly files
data/processed/ord_departures_bts_2020_present.parquet  # processed dataset
```

`data/` is git-ignored — it is large and fully reproducible from the
scripts above.
