#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY before you start the hero.}"

RUN_ID=hero-12d8b6f0-dee637
HERO_ISSUE=https://github.com/marin-community/marin/issues/8506
short_uuid=$(uuidgen | tr '[:upper:]' '[:lower:]')
short_uuid=${short_uuid:0:8}
launch_commit=$(git rev-parse HEAD)
if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
  launch_tree_dirty=true
  launch_tree_state=dirty
else
  launch_tree_dirty=false
  launch_tree_state=clean
fi
launch_job_name="${RUN_ID}-coord-${launch_commit:0:8}-${launch_tree_state}-${short_uuid}"
launch_record=$(printf 'Hero launch requested.\n\n- Run ID: `%s`\n- Commit: `%s`\n- Tree dirty: `%s`\n- Coordinator job: `%s`\n- Target: `cw-us-east-08a` (11 x NVL72)' \
  "$RUN_ID" "$launch_commit" "$launch_tree_dirty" "$launch_job_name")

echo "Recording hero launch on ${HERO_ISSUE}"
gh issue comment "$HERO_ISSUE" --body "$launch_record"
echo "Launching hero from commit ${launch_commit}; tree_dirty=${launch_tree_dirty}"

uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a \
  --priority system \
  --system-reason "hero run; commit=${launch_commit}; tree_dirty=${launch_tree_dirty}" \
  --cpu 1 \
  --memory 4g \
  --max-retries 1000 \
  --job-name "$launch_job_name" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT marin_moe \
  -e GIT_COMMIT "$launch_commit" \
  -e HERO_LAUNCH_TREE_DIRTY "$launch_tree_dirty" \
  -e IRIS_PORT_JAX 32614 \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.75 \
  -e XLA_FLAGS "--xla_gpu_memory_limit_slop_factor=85" \
  -- python -m experiments.grug.moe_hero_ep.launch_scaling_ladder \
    --run-id "$RUN_ID" \
    --size d6144 \
    --version 2026.08.19.2 \
    --run
