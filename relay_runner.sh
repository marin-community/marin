#!/usr/bin/env bash
# EP25 D2 relay: submit the three transport smokes serially, harvest results.
set -uo pipefail
cd /home/marin/projects/marin/.worktrees/ep25-d2-bakeoff
IRIS=/home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris
mkdir -p relay-results

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
}

declare -A jobname
jobname[0]=ep25d2-ragged-smoke-20260724
jobname[1]=ep25d2-fixed-gather-smoke-20260724
jobname[2]=ep25d2-ring-cute-ep4-smoke-20260724

for i in 1 0 2; do  # fixed first per the relay doc, then ragged, then ring
  j=${jobname[$i]}
  echo "=== $(date -u +%FT%TZ) submitting $j" >> relay-results/runner.log
  bash relay-cmd-$i.sh > "relay-results/$j.submit" 2>&1
  rc=$?
  echo "submit rc=$rc" >> relay-results/runner.log
  if [ $rc -ne 0 ]; then
    echo "SUBMIT FAILED for $j — see relay-results/$j.submit" >> relay-results/runner.log
    continue
  fi
  wait_and_harvest "$j"
  echo "=== $(date -u +%FT%TZ) harvested $j" >> relay-results/runner.log
done
echo "RELAY COMPLETE $(date -u +%FT%TZ)" >> relay-results/runner.log
