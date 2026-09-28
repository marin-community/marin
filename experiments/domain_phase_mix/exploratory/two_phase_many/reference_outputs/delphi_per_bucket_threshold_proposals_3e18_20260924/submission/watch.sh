#!/bin/bash
# Detached watch: one STATUS line per 10 min with the state of every job under the parent; exits when the parent is
# terminal. Log: watch.log beside this script.
cd /Users/calvinxu/Projects/Work/Marin/marin
D=experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_per_bucket_threshold_proposals_3e18_20260924/submission
P=/calvinxu/dm-delphi-3e18-per-bucket-threshold-v5p8-20260924
while true; do
  out=$(uv run --offline --no-sync iris --config lib/iris/config/marin.yaml job list --prefix $P 2>/dev/null | grep -v "^Warning\|^I2026\|scopes\|^JOB ID" | awk '{n=split($1,a,"/"); sub(/^(checkpoints-)?pinlin_calvin_xu-data_mixture-delphi_per_bucket_threshold_3e18_20260924-/, "", a[n]); sub(/-pbt_.*$/, "", a[n]); sub(/_[0-9a-f]{8}-[0-9a-f]{8}$/, "", a[n]); printf "%s=%s ", a[n], $2}')
  echo "STATUS $(date '+%H:%M %Z') $out" >> $D/watch.log
  parent=$(uv run --offline --no-sync iris --config lib/iris/config/marin.yaml job list --prefix $P 2>/dev/null | grep -v "^Warning\|^I2026\|scopes\|^JOB ID" | awk -v p="$P" '$1==p{print $2}')
  case "$parent" in succeeded|failed|killed) echo "DONE $(date '+%H:%M %Z') parent=$parent" >> $D/watch.log; break;; esac
  sleep 600
done
