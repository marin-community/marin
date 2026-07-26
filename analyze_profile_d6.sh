#!/usr/bin/env bash
# Run d4's xplane overlap report against a d6 run's uploaded xprof dump.
# The submitting sandbox has no S3 credentials, so this runs as an iris CPU job next to the cluster.
#
# Usage: ./analyze_profile_d6.sh <run-id> <steps-dir> [hosts]
#   e.g. ./analyze_profile_d6.sh ep25d6-d6144-e128-dense-120-0726-1440 steps-20-to-23 1
set -euo pipefail
cd "$(dirname "$0")"

RUN="$1"
STEPS="$2"
HOSTS="${3:-1}"
ROOT="s3://marin-us-east-02a/tmp/ttl=30d/xprof/${RUN}/plugins/profile/${STEPS}"

set -x
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --enable-extra-resources --cpu 8 --memory 64GB --disk 64GB \
  --job-name "d6-overlap-${RUN: -20}-$(date +%H%M%S)" \
  -e XPLANE_TOP_OPS "${TOP_OPS:-40}" \
  -- python -m experiments.grug.moe.xplane_overlap "$ROOT" "$HOSTS"
