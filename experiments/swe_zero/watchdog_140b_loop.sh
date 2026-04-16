#!/bin/bash
# Persistent watchdog loop for the 140B SWE-ZERO pipeline.
#
# Runs monitor_140b_pipeline.py every INTERVAL seconds. Tolerates transient
# Iris controller RPC failures so a single bad pass doesn't kill the loop.

set -u
cd "$(dirname "$0")/../.." || exit 1

PYTHON=/home/kevin/marin-iris-tpu-cli/.venv/bin/python
SCRIPT=experiments/swe_zero/monitor_140b_pipeline.py
LOG=/tmp/monitor_140b_watchdog.log
INTERVAL=${INTERVAL:-300}
MIN_LIVE=${MIN_LIVE:-13}
MAX_RELAUNCHES=${MAX_RELAUNCHES:-13}
STALE_MIN=${STALE_MIN:-45}

while true; do
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog pass ===" >> "$LOG"
    if ! "$PYTHON" "$SCRIPT" \
        --relaunch \
        --min-live-batches "$MIN_LIVE" \
        --max-relaunches "$MAX_RELAUNCHES" \
        --stale-minutes "$STALE_MIN" \
        >>"$LOG" 2>&1; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) monitor pass exited non-zero (continuing)" >> "$LOG"
    fi
    sleep "$INTERVAL"
done
