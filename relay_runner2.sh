#!/usr/bin/env bash
# EP25 D2 relay v2: ragged already submitted; harvest it, then fixed, then ring.
# Job names are parsed from actual submit output, not assumed.
set -uo pipefail
cd /home/marin/projects/marin/.worktrees/ep25-d2-bakeoff
IRIS=/home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris
mkdir -p relay-results
RLOG=relay-results/runner2.log

wait_and_harvest() {
  local job="$1"
  local deadline=$((SECONDS + 2700))
  local line=""
  while [ $SECONDS -lt $deadline ]; do
    line=$($IRIS --cluster=marin job list 2>/dev/null | grep -F "$job" | head -1)
    echo "$(date -u +%FT%TZ) $line" >> relay-results/poll.log
    case "$line" in
      *SUCCEEDED*|*FAILED*|*KILLED*|*STOPPED*|*CANCELLED*) break ;;
    esac
    sleep 60
  done
  timeout 180 $IRIS --cluster=marin job summary "/mwittmann/$job" \
    > "relay-results/$job.summary" 2>&1
  timeout 240 $IRIS --cluster=marin job logs "/mwittmann/$job" 2>&1 | tail -500 \
    > "relay-results/$job.log"
  echo "=== $(date -u +%FT%TZ) harvested $job" >> $RLOG
}

submit_and_harvest() {
  local script="$1"
  local out
  out=$(bash "$script" 2>&1)
  local rc=$?
  local job
  job=$(printf '%s' "$out" | grep -o 'Submitting job: [a-z0-9-]*' | awk '{print $3}' | head -1)
  printf '%s\n' "$out" > "relay-results/${job:-unknown-$script}.submit"
  echo "=== $(date -u +%FT%TZ) $script rc=$rc job=$job" >> $RLOG
  if [ $rc -ne 0 ] || [ -z "$job" ]; then
    echo "SUBMIT FAILED for $script" >> $RLOG
    return 1
  fi
  wait_and_harvest "$job"
}

wait_and_harvest ep25d2-ragged-smoke-20260724   # already in flight from v1 runner
submit_and_harvest relay-cmd-0.sh               # fixed + gather
submit_and_harvest relay-cmd-2.sh               # ring_cute EP4
echo "RELAY COMPLETE $(date -u +%FT%TZ)" >> $RLOG
