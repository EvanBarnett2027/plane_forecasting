#!/usr/bin/env bash
#
# download_data.sh -- build the full training dataset from scratch.
#
# Downloads ~6 years of BTS flight records and IEM ASOS weather for ORD plus
# every destination airport, joins them, and fits the weather-noise table.
#
# WARNING: this pulls a lot of data over the network and can take a while
# (very roughly 30-60 min on a typical connection; longer on a slow one).
# Each step shows a live elapsed-time / ETA readout. The ETAs are rough
# estimates -- actual time depends heavily on your network speed.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "Virtual environment not found at $PYTHON"
    echo "Create it first:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# --- pretty helpers -------------------------------------------------------- #
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
    BOLD=''; GREEN=''; RED=''; DIM=''; RESET=''
fi

fmt_dur() {  # seconds -> "45s" or "3m07s"
    local s=$1
    if (( s < 60 )); then printf '%ds' "$s"
    else printf '%dm%02ds' $(( s / 60 )) $(( s % 60 )); fi
}

# run_step "Label" ESTIMATED_SECONDS  command args...
# Runs the command, streaming its output to a log file while showing a live
# elapsed/ETA line. On failure, prints the tail of the log and exits.
run_step() {
    local label="$1"; local est="$2"; shift 2
    local slug; slug="$(printf '%s' "$label" | tr ' /:' '___')"
    local log="$LOG_DIR/${slug}.log"

    printf '\n%s==> %s%s  %s(est ~%s)%s\n' \
        "$BOLD" "$label" "$RESET" "$DIM" "$(fmt_dur "$est")" "$RESET"

    "$@" >"$log" 2>&1 &
    local pid=$!
    local start; start=$(date +%s)

    if [[ -t 1 ]]; then
        local spin='|/-\' i=0
        while kill -0 "$pid" 2>/dev/null; do
            local elapsed=$(( $(date +%s) - start ))
            local remain=$(( est - elapsed )) eta
            if (( remain >= 0 )); then eta="ETA ~$(fmt_dur "$remain")"
            else eta="$(fmt_dur $(( -remain ))) over est"; fi
            i=$(( (i + 1) % 4 ))
            printf '\r  %s  elapsed %s   %s            ' \
                "${spin:i:1}" "$(fmt_dur "$elapsed")" "$eta"
            sleep 1
        done
    fi

    local rc=0
    wait "$pid" || rc=$?
    local total=$(( $(date +%s) - start ))

    if (( rc == 0 )); then
        printf '\r  %s✓%s done in %s%-24s\n' "$GREEN" "$RESET" "$(fmt_dur "$total")" ''
    else
        printf '\r  %s✗%s failed after %s (exit %d)%-12s\n' \
            "$RED" "$RESET" "$(fmt_dur "$total")" "$rc" ''
        echo "  ---- last 30 lines of $log ----"
        tail -n 30 "$log" | sed 's/^/  /'
        exit "$rc"
    fi
}

# --- banner ---------------------------------------------------------------- #
printf '%s\n' "${BOLD}Plane Forecasting -- data download${RESET}"
echo "This downloads several GB of flight + weather data and can take"
echo "roughly 30-60 min depending on your network. Logs are written to ./$LOG_DIR/."
PIPE_START=$(date +%s)

# --- pipeline -------------------------------------------------------------- #
run_step "1/7 Ingest BTS flight records (2020-present)" 600 \
    $PYTHON -m src.data.ingest_bts_ord --start-year 2020 --start-month 1

run_step "2/7 Clean + label flight data" 60 \
    $PYTHON -m src.data.clean_bts_ord

run_step "3/7 Ingest ORD weather observations" 120 \
    $PYTHON -m src.data.ingest_iem_asos --start-year 2020 --start-month 1 --station ORD

run_step "4/7 Clean ORD weather" 30 \
    $PYTHON -m src.data.clean_iem_asos --station ORD

run_step "5/7 Ingest + clean destination-airport weather" 1200 \
    $PYTHON -m src.data.ingest_dest_weather --also-clean

run_step "6/7 Join flights with weather" 60 \
    $PYTHON -m src.data.join_flights_weather

run_step "7/7 Fit weather-noise sigma table" 30 \
    $PYTHON -m src.model.weather_noise fit \
        --weather data/processed/dest_weather_iem_2020_present.parquet \
        --out data/processed/weather_noise_sigmas.json

# --- done ------------------------------------------------------------------ #
TOTAL=$(( $(date +%s) - PIPE_START ))
printf '\n%sAll data ready%s in %s. Processed files are in data/processed/.\n' \
    "$GREEN" "$RESET" "$(fmt_dur "$TOTAL")"
echo "Next: train the model with  ./train_model.sh"
