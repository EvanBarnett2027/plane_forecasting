#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STREAMLIT=".venv/bin/streamlit"

if [[ ! -f "$STREAMLIT" ]]; then
    echo "Virtual environment not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Optional: pass AERODATABOX_KEY=<key> before the script to enable the Live source.
# Example: AERODATABOX_KEY=your_key ./start_app.sh

echo "Starting Streamlit dashboard..."
if [[ -n "${AERODATABOX_KEY:-}" ]]; then
    echo "  AeroDataBox key detected — Live source enabled."
else
    echo "  No AERODATABOX_KEY set — running in Next 48h / Historical replay mode."
    echo "  To enable live flight data: AERODATABOX_KEY=<key> ./start_app.sh"
fi
echo ""

exec $STREAMLIT run dashboard/app.py
