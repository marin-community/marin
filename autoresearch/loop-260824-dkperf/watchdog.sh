#!/usr/bin/env bash
# Out-of-process runtime cap and early release for one arm.
#
# Three jobs, all of which end with `iris job cancel` on the coord path (Iris cancels descendants,
# so the 16-node train job goes with it):
#   1. Step budget: cancel STEP_BUDGET seconds after the first step this watch observes. Compile
#      time varies run to run (~8 min so far) and an arm that pays a slow compile should still get
#      its full measurement window; an arm whose change makes *steps* slow should not get extra
#      rack time to compensate.
#   2. Compile ceiling: cancel if no step has appeared COMPILE_CEILING seconds after submission.
#      This bounds the total hold when startup wedges before step 0.
#   3. Early release: cancel as soon as W&B shows the last step the scoring window needs. A scored
#      arm has no reason to keep 64 GB200s.
# Iris's own --timeout on the coord job is the outer backstop; it must cover
# COMPILE_CEILING + STEP_BUDGET. This watchdog runs in a separate process so a wedged CLI or a
# lost session cannot defeat it.
#
# systemd-run and crontab are both unavailable in this sandbox, so this is a plain detached poll
# loop rather than a timer unit.
#
# usage: RID=<run-id> [STEP_BUDGET=900] [COMPILE_CEILING=1200] [SCORE_MAX_STEP=19] watchdog.sh
set -uo pipefail

RID="${RID:?set RID}"
STEP_BUDGET="${STEP_BUDGET:-900}"
COMPILE_CEILING="${COMPILE_CEILING:-1200}"
SCORE_MAX_STEP="${SCORE_MAX_STEP:-19}"
POLL="${POLL:-60}"

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
IRIS=(uv run iris --config lib/iris/config/marin.yaml)
COORD="/mwittmann/${RID}-coord"
start=$(date +%s)
# Until a step appears the deadline is the compile ceiling; the first observed step rebases it to
# STEP_BUDGET from that moment.
deadline=$(( start + COMPILE_CEILING ))
first_step=-1

cancel_arm() {  # cancel_arm <reason>
  echo "$(date -u +%H:%M:%S) cancelling ${COORD}: $1"
  # Cancel the train job by its own path as well as through the coordinator. Iris's --timeout may
  # already have taken the coordinator down, and a cancel routed only through a dead parent is
  # exactly how a 16-node child gets orphaned on the rack.
  ( cd "$REPO" && timeout 300 "${IRIS[@]}" job cancel "${COORD}/grug-train-${RID}" 2>&1 | tail -2 )
  ( cd "$REPO" && timeout 300 "${IRIS[@]}" job cancel "$COORD" 2>&1 | tail -2 )
}

state_of() {  # state_of <job path>
  ( cd "$REPO" && timeout 180 "${IRIS[@]}" job describe "$1" 2>/dev/null \
      | sed -n 's/^State: \([a-z_]*\).*/\1/p' | head -1 )
}

max_step() {
  ( cd "$REPO" && timeout 180 uv run --with wandb python - "$RID" <<'PY' 2>/dev/null
import sys, wandb
try:
    run = wandb.Api().run(f"marin-community/marin_moe/{sys.argv[1]}")
except Exception:
    print(-1); raise SystemExit
steps = [row.get("_step", -1) for row in run.scan_history(keys=["_step", "throughput/mfu"])]
print(max(steps) if steps else -1)
PY
  )
}

while :; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    if [ "${first_step:--1}" -ge 0 ] 2>/dev/null; then
      cancel_arm "step budget ${STEP_BUDGET}s past first step reached"
      echo "WATCHDOG hard-cap step-budget"
    else
      cancel_arm "no step within the ${COMPILE_CEILING}s compile ceiling"
      echo "WATCHDOG hard-cap compile-ceiling"
    fi
    exit 0
  fi
  train_state="$(state_of "${COORD}/grug-train-${RID}")"
  steps="$(max_step)"
  echo "$(date -u +%H:%M:%S) t=$(( now - start ))s train=${train_state:-none} step=${steps}"
  # Iris has several terminal failure states beyond a bare `failed` (worker_failed, cosched_failed
  # among them), so match the suffix rather than enumerating them.
  case "$train_state" in
    succeeded|killed|cancelled|*failed)
      echo "WATCHDOG terminal train=${train_state} step=${steps}"; exit 0;;
  esac
  # A restored arm resumes at the checkpoint's step, so progress is counted from the first step
  # this watch actually observed rather than from zero. Without that, an absolute threshold fires
  # on the very first reading and cancels a run that has trained nothing.
  if [ "${first_step:--1}" -lt 0 ] && [ "${steps:--1}" -ge 0 ] 2>/dev/null; then
    first_step="$steps"
    deadline=$(( now + STEP_BUDGET ))
    echo "$(date -u +%H:%M:%S) first observed step=${first_step}, step budget ${STEP_BUDGET}s starts now"
  fi
  if [ "${first_step:--1}" -ge 0 ] 2>/dev/null && [ $(( steps - first_step )) -ge "$SCORE_MAX_STEP" ]; then
    cancel_arm "scored through step ${steps} (${SCORE_MAX_STEP} past first)"
    echo "WATCHDOG scored step=${steps} first=${first_step}"
    exit 0
  fi
  sleep "$POLL"
done
