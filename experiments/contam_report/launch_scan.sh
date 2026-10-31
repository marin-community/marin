#!/usr/bin/env bash
# Launch a contamination scan on cw-us-east-02a, where the hero data lives.
# Usage: launch_scan.sh JOB_NAME PLAN_URL OUT_PREFIX MAX_WORKERS
set -euo pipefail
job_name=$1 plan=$2 out=$3 workers=$4
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
  --target-cluster cw-us-east-02a --job-name "$job_name" \
  --cpu 1 --memory 2GB --timeout 86400 \
  -- python -m experiments.contam_report.scan "$plan" "$out" \
  s3://marin-us-east-02a/marin/analysis/contam_report/bloom_v3 --max-workers "$workers"
