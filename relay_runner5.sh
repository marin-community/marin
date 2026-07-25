#!/usr/bin/env bash
# EP25 D2 relay v4: two rack-scale transport arms, serial, 100-min cap each.
set -uo pipefail
cd /home/marin/projects/marin/.worktrees/ep25-d2-bakeoff
IRIS=/home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris
mkdir -p relay-results
RLOG=relay-results/runner5.log

wait_and_harvest() {
  local job="$1"
  local deadline=$((SECONDS + 6000))
  local line=""
  while [ $SECONDS -lt $deadline ]; do
    line=$(timeout 90 $IRIS --cluster=marin job list 2>/dev/null | grep -F "$job" | grep -v grug-train | head -1)
    echo "$(date -u +%FT%TZ) $line" >> relay-results/poll.log
    case "$line" in
      *succeeded*|*failed*|*killed*|*stopped*|*cancelled*) break ;;
    esac
    sleep 120
  done
  timeout 180 $IRIS --cluster=marin job summary "/mwittmann/$job" \
    > "relay-results/$job.summary" 2>&1
  timeout 240 $IRIS --cluster=marin job logs "/mwittmann/$job" 2>&1 | tail -600 \
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



submit_and_harvest relay-cmd-1.sh   # ring_cute EP64, rack
echo "RELAY COMPLETE $(date -u +%FT%TZ)" >> $RLOG
