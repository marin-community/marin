#!/bin/bash
# Copy each H100 run's FineStore samples into ROOT/<model>/<eval>/r<run>/samples for analyze.py.
# Usage: fetch_h100.sh ROOT MODEL... (needs CW keys: set -a; . ~/oa/marin/.env; set +a)
set -euo pipefail
root=${1:?root}; shift
fsutil=/home/karan/oa/marin-task-261/.venv/bin/fsutil
evals=s3://marin-us-east-02a/marin/evals
for model in "$@"; do
  for key in math500 gsm8k-0shot humanevalplus ifbench; do
    for run in 1 2; do
      prefix=$(timeout 120 $fsutil ls "$evals/rsi289-$model-h100-r1-$key-r$run-*" 2>/dev/null | grep -v '^I20' | head -1)
      [ -n "$prefix" ] || { echo "missing $model $key r$run"; continue; }
      dest="$root/$model/$key/r$run"
      mkdir -p "$dest"
      timeout 600 $fsutil cp -r "$evals/${prefix%/}/results/data/samples" "$dest/" >/dev/null 2>&1
      timeout 60 $fsutil cp "$evals/${prefix%/}/record.json" "$dest/record.json" >/dev/null 2>&1
      echo "$model $key r$run <- ${prefix%/}"
    done
  done
done
