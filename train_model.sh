#!/usr/bin/env bash
#
# train_model.sh -- train the deployable production model and run the
# recent-week deployment-realism backtest.
#
# Requires the processed dataset produced by ./download_data.sh. Each step
# shows a live elapsed-time / ETA readout (ETAs are rough estimates).
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

JOINED="data/processed/ord_flights_with_weather_2020_present.parquet"
SIGMAS="data/processed/weather_noise_sigmas.json"
for f in "$JOINED" "$SIGMAS"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing required data file: $f"
        echo "Run ./download_data.sh first to build the dataset."
        exit 1
    fi
done

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
printf '%s\n' "${BOLD}Plane Forecasting -- model training${RESET}"
echo "Trains the production LightGBM on all labelled data, then runs the"
echo "recent-week backtest. Logs are written to ./$LOG_DIR/."
PIPE_START=$(date +%s)

# --- pipeline -------------------------------------------------------------- #
run_step "1/2 Train production model" 300 \
    $PYTHON -m src.model.train_production_model

run_step "2/2 Recent-week deployment backtest" 600 \
    $PYTHON -m src.model.evaluate_recent_week

# --- done ------------------------------------------------------------------ #
TOTAL=$(( $(date +%s) - PIPE_START ))
printf '\n%sTraining complete%s in %s. Artifacts are in artifacts/production_model/.\n' \
    "$GREEN" "$RESET" "$(fmt_dur "$TOTAL")"
echo "Next: launch the dashboard with  ./start_app.sh"
