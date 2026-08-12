#!/usr/bin/env bash
# Poll an already-submitted ar8062 coordinator job to terminal state, then score it.
# Usage: poll_score.sh <run-id> [num-steps]
set -euo pipefail

REPO=/home/marin/projects/marin
cd "$REPO"
RUN_ID="$1"
NUM_STEPS="${2:-1000}"
DROP_BUDGET="${AR_DROP_BUDGET:-0.02}"
DROP_WINDOW="${AR_DROP_WINDOW:-50}"
SCORE_START="${AR_SCORE_START:-100}"
TIMEOUT_SECONDS="${AR_TIMEOUT:-21600}"
IRIS=(uv run iris --config lib/iris/config/marin.yaml)
JOB="/mwittmann/${RUN_ID}-coord"

deadline=$((SECONDS + TIMEOUT_SECONDS))
state=""
while (( SECONDS < deadline )); do
  state=$("${IRIS[@]}" job list --prefix "$JOB" 2>/dev/null \
    | awk -v j="$JOB" '$1 == j {print tolower($2); exit}' || true)
  case "$state" in
    succeeded|completed) break ;;
    failed|killed|error)
      echo "job ${JOB} ended in state ${state}; recent logs:" >&2
      "${IRIS[@]}" job logs --since-seconds 600 "$JOB" >&2 || true
      exit 1 ;;
  esac
  sleep 120
done
if [[ "$state" != succeeded && "$state" != completed ]]; then
  echo "timeout after ${TIMEOUT_SECONDS}s in state '${state}'; killing job to free the rack" >&2
  "${IRIS[@]}" job kill "$JOB" >&2 || true
  exit 1
fi

uv run python "$(dirname "$0")/score_ep64.py" \
  --run-id "$RUN_ID" \
  --start-step "$SCORE_START" --end-step $((NUM_STEPS - DROP_WINDOW - 1)) \
  --drop-start $((NUM_STEPS - DROP_WINDOW)) --drop-end $((NUM_STEPS - 1)) \
  --drop-budget "$DROP_BUDGET" --gpus 64
