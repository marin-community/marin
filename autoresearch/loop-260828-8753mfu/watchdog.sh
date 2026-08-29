#!/usr/bin/env bash
# Out-of-process runtime cap, early release, and anomaly alerting for one arm.
#
# Exits (each prints one WATCHDOG marker line; the harness alerts on process exit, so exiting
# fast IS the alert):
#   1. Step budget: cancel STEP_BUDGET seconds after the first step this watch observes. Compile
#      time varies run to run; an arm that pays a slow compile still gets its full measurement
#      window, but an arm whose change makes *steps* slow gets no extra rack time.
#   2. Compile ceiling: cancel if no step has appeared COMPILE_CEILING seconds after submission.
#   3. Early release: cancel as soon as W&B shows the last step the scoring window needs.
#   4. Preemption or task failure: `job describe` publishes failures= and preemptions= counters.
#      Either going nonzero invalidates the measurement (a rescheduled gang recompiles and the
#      step timings straddle the gap), so cancel immediately instead of letting the retry burn
#      the ceiling.
#   5. Coordinator death before the train job exists (e.g. a config guard raising at dispatch):
#      previously this burned the whole compile ceiling; now it exits within one poll.
# Iris's own --timeout on the coord job is the outer backstop; it must cover
# COMPILE_CEILING + STEP_BUDGET. This watchdog runs in a separate process so a wedged CLI or a
# lost session cannot defeat it.
#
# State polls are cheap (one describe RPC), so they run every POLL=20s; the W&B step query is
# heavier and runs every third poll.
#
# usage: RID=<run-id> [STEP_BUDGET=900] [COMPILE_CEILING=1200] [SCORE_MAX_STEP=19] watchdog.sh
set -uo pipefail

RID="${RID:?set RID}"
STEP_BUDGET="${STEP_BUDGET:-900}"
COMPILE_CEILING="${COMPILE_CEILING:-1200}"
SCORE_MAX_STEP="${SCORE_MAX_STEP:-19}"
POLL="${POLL:-20}"
STEP_POLL_EVERY="${STEP_POLL_EVERY:-3}"

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
IRIS=(uv run iris --config lib/iris/config/marin.yaml)
COORD="/mwittmann/${RID}-coord"
TRAIN="${COORD}/grug-train-${RID}"
start=$(date +%s)
# A production submission may queue behind another job, so the compile ceiling starts when the
# train job is first observed running, not at submission. QUEUE_CEILING bounds the pending phase.
QUEUE_CEILING="${QUEUE_CEILING:-10800}"
deadline=$(( start + QUEUE_CEILING ))
run_seen=-1
first_step=-1
steps=-1
tick=0

cancel_arm() {  # cancel_arm <reason>
  echo "$(date -u +%H:%M:%S) cancelling ${COORD}: $1"
  # Cancel the train job by its own path as well as through the coordinator. Iris's --timeout may
  # already have taken the coordinator down, and a cancel routed only through a dead parent is
  # exactly how a 16-node child gets orphaned on the rack.
  ( cd "$REPO" && timeout 300 "${IRIS[@]}" job cancel "$TRAIN" 2>&1 | tail -2 )
  ( cd "$REPO" && timeout 300 "${IRIS[@]}" job cancel "$COORD" 2>&1 | tail -2 )
}

state_line_of() {  # state_line_of <job path> -> "state failures preemptions" (empty if absent)
  ( cd "$REPO" && timeout 120 "${IRIS[@]}" job describe "$1" 2>/dev/null \
      | sed -n 's/^State: \([a-z_]*\).*failures=\([0-9]*\).*preemptions=\([0-9]*\).*/\1 \2 \3/p' | head -1 )
}

max_step() {
  ( cd "$REPO" && timeout 180 uv run --with wandb python - "$RID" <<'PY' 2>/dev/null
import sys, wandb
try:
    run = wandb.Api().run(f"marin-community/marin_moe/{sys.argv[1]}")
except Exception:
    print(-2); raise SystemExit
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
      if [ "${run_seen:--1}" -ge 0 ]; then
        cancel_arm "no step within the ${COMPILE_CEILING}s compile ceiling"
        echo "WATCHDOG hard-cap compile-ceiling"
      else
        cancel_arm "still queued after ${QUEUE_CEILING}s"
        echo "WATCHDOG hard-cap queue-ceiling"
      fi
    fi
    exit 0
  fi

  read -r train_state train_failures train_preemptions <<<"$(state_line_of "$TRAIN")" || true
  if [ -z "${train_state:-}" ]; then
    # No train job yet: the coordinator is either still dispatching or died before dispatch.
    read -r coord_state coord_failures coord_preemptions <<<"$(state_line_of "$COORD")" || true
    case "${coord_state:-}" in
      succeeded|killed|cancelled|*failed)
        cancel_arm "coordinator ${coord_state} before the train job existed"
        echo "WATCHDOG coord-terminal state=${coord_state} failures=${coord_failures:-?}"
        exit 0;;
    esac
  fi
  if [ "${run_seen:--1}" -lt 0 ] && [ "${train_state:-}" = "running" ]; then
    # Iris can report a gang "running" while another job still holds the rack; the W&B run is
    # only created once distributed init actually completes, so gate the compile clock on it.
    probe="$(max_step)"
    if [ "${probe:--2}" -ge -1 ] 2>/dev/null; then
      run_seen=$now
      deadline=$(( now + COMPILE_CEILING ))
      echo "$(date -u +%H:%M:%S) gang up (wandb run exists), compile ceiling ${COMPILE_CEILING}s starts now"
    fi
  fi
  if [ "${train_preemptions:-0}" -gt 0 ] 2>/dev/null; then
    cancel_arm "train job preempted (preemptions=${train_preemptions})"
    echo "WATCHDOG preempted train=${train_state} preemptions=${train_preemptions} step=${steps}"
    exit 0
  fi
  if [ "${train_failures:-0}" -gt 0 ] 2>/dev/null; then
    cancel_arm "train task failed (failures=${train_failures})"
    echo "WATCHDOG task-failed train=${train_state} failures=${train_failures} step=${steps}"
    exit 0
  fi
  case "${train_state:-}" in
    succeeded|killed|cancelled|*failed)
      echo "WATCHDOG terminal train=${train_state} step=${steps}"; exit 0;;
  esac

  if [ $(( tick % STEP_POLL_EVERY )) -eq 0 ]; then
    steps="$(max_step)"
    echo "$(date -u +%H:%M:%S) t=$(( now - start ))s train=${train_state:-none} step=${steps}"
    # A restored arm resumes at the checkpoint's step, so progress is counted from the first step
    # this watch actually observed rather than from zero.
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
  fi
  tick=$(( tick + 1 ))
  sleep "$POLL"
done
