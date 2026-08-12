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

STALL_LIMIT="${AR_STALL_LIMIT:-1200}"  # kill if no new W&B step for this many seconds

deadline=$((SECONDS + TIMEOUT_SECONDS))
state=""
last_step=-1
last_progress=$SECONDS
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
  # Stall watchdog: a wedged gang stays 'running' with every rank alive (the
  # iteration-0 baseline burned 10.5 rack-hours that way). Progress = the run's
  # last logged _step advancing; startup/compile is covered by TIMEOUT, not this.
  # Progress only counts while the W&B run is live: a gang retry inherits the
  # previous attempt's finished run, whose stale summary otherwise reads as
  # "progress then stall" and gets a queued job killed (this happened).
  step=$(uv run python -c "
import wandb
try:
    run = wandb.Api().run('marin-community/marin_moe/${RUN_ID}')
    if run.state == 'running':
        print(int(run.summary.get('_step', -1)))
    else:
        print(-1)
except Exception:
    print(-1)
" 2>/dev/null || echo -1)
  if (( step > last_step )); then
    last_step=$step
    last_progress=$SECONDS
  elif (( last_step >= 0 && step >= 0 && SECONDS - last_progress > STALL_LIMIT )); then
    echo "STALL: no step past ${last_step} for >${STALL_LIMIT}s; capturing summary and killing ${JOB}" >&2
    "${IRIS[@]}" job summary "${JOB}/grug-train-${RUN_ID}" >&2 || true
    "${IRIS[@]}" job kill "$JOB" >&2 || true
    exit 1
  fi
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
