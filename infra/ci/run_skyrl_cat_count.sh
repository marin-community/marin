#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MARIN_ROOT="$(git rev-parse --show-toplevel)"
REPOSITORY_ROOT="$MARIN_ROOT/vendor/skyrl"
TARGET_CLUSTER="${TARGET_CLUSTER:-cw-rno2a}"
IRIS_JOB_PREFIX="${IRIS_JOB_PREFIX:-/runner}"
DEADLINE=1200
echo "CAT_COUNT_NIGHTLY marin=$(git rev-parse HEAD)"

for attempt in 1 2; do
  log="${RUNNER_TEMP:-/tmp}/cat-count-nightly-a$attempt.log"
  version="$(date -u +%Y.%m.%d).${GITHUB_RUN_ID:-$(date +%s)}${GITHUB_RUN_ATTEMPT:-1}$attempt"
  start=$(date +%s)
  if (cd "$MARIN_ROOT" && timeout --signal=INT --kill-after=30s "${DEADLINE}s" \
    uv run --frozen --package marin-core --extra cpu --no-default-groups iris --cluster marin job run \
    --target-cluster "$TARGET_CLUSTER" --job-name "${JOB_NAME:?}-a$attempt" \
    --cpu 4 --memory 16GB --disk 8GB --extra cpu --priority interactive --max-retries 0 \
    --timeout "$DEADLINE" --enable-extra-resources -- python -m experiments.post_training.cat_count_canary \
    --preset gate --version "$version" --cluster "$TARGET_CLUSTER" \
    --job-timeout-seconds "$DEADLINE" --set trainer.logger=console --run) 2>&1 | tee "$log"; then
    status=0
  else
    status=$?
  fi
  if ((status == 0)); then
    uv run --project "$MARIN_ROOT" --frozen --package marin-core --extra cpu --no-default-groups \
      iris --cluster marin job logs "$IRIS_JOB_PREFIX/$JOB_NAME-a$attempt" --no-tail --max-lines 60000 \
      2>&1 | tee "$log" || { echo "INFRASTRUCTURE_FAILURE: complete native log unavailable"; exit 2; }
  fi
  wall_clock=$(($(date +%s) - start))
  echo "CAT_COUNT_NIGHTLY attempt=$attempt wall_clock_seconds=$wall_clock exit_status=$status"
  if ! grep -q 'WANDB_MIRROR kind=train ' "$log"; then
    echo "INFRASTRUCTURE_FAILURE: no training step ran"
    uv run --project "$MARIN_ROOT" --frozen --package marin-core --extra cpu --no-default-groups \
      iris --cluster marin job cancel --prefix "$IRIS_JOB_PREFIX/$JOB_NAME-a$attempt"
    continue
  fi
  if ((status != 0)); then
    echo "GATE_FAILURE: training started but the job did not complete"
    exit 1
  fi
  python3 "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/gate.py" --log "$log" \
    --spec "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-async.json" \
    --wall-clock-seconds "$wall_clock" || { echo "GATE_FAILURE: native metrics failed"; exit 1; }
  run_id="$(sed -n '/\[telemetry\].* run_id=/ { s/.* run_id=\([^ ]*\).*/\1/; p; q; }' "$log")"
  (cd "$MARIN_ROOT" && uv run --frozen --package marin-core --extra cpu --no-default-groups \
    python "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/dashboard_readiness.py" --run-id "$run_id") \
    || echo "::: the readiness reporter itself failed; the run and its gate are unaffected"
  exit 0
done
exit 2
