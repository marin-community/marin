#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/iris/run_poll_task_memory.sh TASK_ID [OUTPUT_JSONL]

Environment overrides:
  IRIS_CONFIG   Iris config path (default: lib/iris/examples/marin.yaml)
  INTERVAL      Poll interval in seconds (default: 5)
  TIMEOUT_MS    RPC timeout in ms (default: 30000)
  FLUSH_EVERY   Flush cadence in samples (default: 1)
  SAMPLES       Stop after N samples; 0 means run until terminal (default: 0)
  DETACH        Set to 1 to run under nohup in the background (default: 0)

Examples:
  scripts/iris/run_poll_task_memory.sh \
    /ahmed/irl-e4p-100r2-0324-2103/rl-e4p-20260325-042445-train/0

  DETACH=1 scripts/iris/run_poll_task_memory.sh \
    /ahmed/irl-e4p-100r2-0324-2103/rl-e4p-20260325-042445-train/0 \
    scratch/e4p_train_memory.jsonl
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

TASK_ID=$1
CONFIG=${IRIS_CONFIG:-lib/iris/examples/marin.yaml}
INTERVAL_SECONDS=${INTERVAL:-5}
TIMEOUT_MS_VALUE=${TIMEOUT_MS:-30000}
FLUSH_EVERY_VALUE=${FLUSH_EVERY:-1}
SAMPLES_VALUE=${SAMPLES:-0}
DETACH_VALUE=${DETACH:-0}

mkdir -p scratch

if [[ $# -ge 2 ]]; then
  OUTPUT_JSONL=$2
else
  UTC_STAMP=$(date -u +%Y%m%d-%H%M%SZ)
  TASK_LEAF=$(printf '%s' "$TASK_ID" | awk -F/ '{print $(NF-1)}')
  OUTPUT_JSONL="scratch/${UTC_STAMP}_${TASK_LEAF}_memory.jsonl"
fi

LOG_PATH=${OUTPUT_JSONL%.jsonl}.log

CMD=(
  uv run python scripts/iris/poll_task_memory.py
  --config "$CONFIG"
  --task-id "$TASK_ID"
  --interval "$INTERVAL_SECONDS"
  --timeout-ms "$TIMEOUT_MS_VALUE"
  --flush-every "$FLUSH_EVERY_VALUE"
  --samples "$SAMPLES_VALUE"
  --output "$OUTPUT_JSONL"
)

printf '[%s] task_id=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TASK_ID"
printf '[%s] output=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUTPUT_JSONL"

if [[ "$DETACH_VALUE" == "1" ]]; then
  nohup "${CMD[@]}" > "$LOG_PATH" 2>&1 &
  printf '[%s] detached_pid=%s log=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$!" "$LOG_PATH"
else
  exec "${CMD[@]}"
fi
