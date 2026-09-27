#!/bin/bash
# Detached progress log: every 5 min, one line per training child with its latest tqdm progress (done/total, rate,
# remaining, loss); exits when the parent is terminal. Log: progress.log beside this script.
cd /Users/calvinxu/Projects/Work/Marin/marin
D=.agents/projects/mariner_optimum_scale_transfer_20260924
P=/calvinxu/dm-mopt-scale-transfer-20260924-retry5
CFG=lib/iris/config/marin.yaml
while true; do
  listing=$(uv run --offline --no-sync iris --config $CFG job list --prefix $P 2>/dev/null | grep -v "^Warning\|^I2026\|scopes\|^JOB ID")
  stamp=$(date '+%H:%M %Z')
  echo "$listing" | awk -v p="$P" '$1!=p{print $1, $2}' | while read -r job state; do
    name=$(echo "$job" | sed -E 's#.*/checkpoints-pinlin_calvin_xu-data_mixture-##; s/_[0-9a-f]{8}-[0-9a-f]{8}$//')
    if [ "$state" = "running" ] || [ "$state" = "succeeded" ] || [ "$state" = "failed" ]; then
      line=$(uv run --offline --no-sync iris --config $CFG job logs "$job" 2>/dev/null | grep "tqdm_log" | tail -1 | sed -E 's/.*Progress on:train //; s/ postfix:/ /' | cut -c1-90)
    else
      line=""
    fi
    echo "PROGRESS $stamp $name $state ${line:-(no progress line)}" >> $D/progress.log
  done
  parent=$(echo "$listing" | awk -v p="$P" '$1==p{print $2}')
  case "$parent" in succeeded|failed|killed) echo "PROGRESS $stamp parent=$parent DONE" >> $D/progress.log; break;; esac
  sleep 300
done
