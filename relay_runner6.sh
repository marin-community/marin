#!/usr/bin/env bash
# EP25 R6-4 relay: MXFP8 ladder — numerics, EP4 pair, rack pair. Serial; stop on
# numerics/parity failure so racks aren't wasted on a broken kernel.
set -uo pipefail
cd /home/marin/projects/marin/.worktrees/ep25-d2-bakeoff
IRIS=/home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris
mkdir -p relay-results
RLOG=relay-results/runner6.log

wait_and_harvest() {
  local job="$1" deadline=$((SECONDS + 6000)) line=""
  while [ $SECONDS -lt $deadline ]; do
    line=$(timeout 90 $IRIS --cluster=marin job list 2>/dev/null | grep -F "$job" | grep -v grug-train | head -1)
    echo "$(date -u +%FT%TZ) $line" >> relay-results/poll.log
    case "$line" in
      *succeeded*|*failed*|*killed*|*stopped*|*cancelled*) break ;;
    esac
    sleep 90
  done
  timeout 180 $IRIS --cluster=marin job summary "/mwittmann/$job" > "relay-results/$job.summary" 2>&1
  timeout 240 $IRIS --cluster=marin job logs "/mwittmann/$job" 2>&1 | tail -600 > "relay-results/$job.log"
  echo "=== $(date -u +%FT%TZ) harvested $job ($line)" >> $RLOG
  case "$line" in *succeeded*) return 0 ;; *) return 1 ;; esac
}

submit_and_harvest() {
  local script="$1" out rc job
  out=$(bash "$script" 2>&1); rc=$?
  job=$(printf '%s' "$out" | grep -o 'Submitting job: [a-z0-9-]*' | awk '{print $3}' | head -1)
  printf '%s\n' "$out" > "relay-results/${job:-unknown-$script}.submit"
  echo "=== $(date -u +%FT%TZ) $script rc=$rc job=$job" >> $RLOG
  [ $rc -ne 0 ] || [ -z "$job" ] && { echo "SUBMIT FAILED $script" >> $RLOG; return 1; }
  wait_and_harvest "$job"
}

submit_and_harvest relay-cmd-0.sh || { echo "NUMERICS FAILED — halting ladder" >> $RLOG; exit 1; }
submit_and_harvest relay-cmd-1.sh || { echo "EP4 bf16 FAILED — halting ladder" >> $RLOG; exit 1; }
submit_and_harvest relay-cmd-2.sh || { echo "EP4 mxfp8 FAILED — halting ladder" >> $RLOG; exit 1; }
submit_and_harvest relay-cmd-3.sh
submit_and_harvest relay-cmd-4.sh
echo "RELAY COMPLETE $(date -u +%FT%TZ)" >> $RLOG
