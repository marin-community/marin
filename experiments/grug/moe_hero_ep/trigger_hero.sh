#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY before you start the hero.}"

# This run continues the previous hero's full state from this checkpoint without writing to that run's tree.
RUN_ID=hero-ragged_a2a-ep-step54k
HANDOFF_CHECKPOINT=s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-54000
HERO_ISSUE=https://github.com/marin-community/marin/issues/8506
TARGET_CLUSTER=cw-us-east-08a
TARGET_DESCRIPTION='11 x NVL72'
short_uuid=$(uuidgen | tr '[:upper:]' '[:lower:]')
short_uuid=${short_uuid:0:8}
launch_commit=$(git rev-parse HEAD)
if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
  launch_tree_dirty=true
else
  launch_tree_dirty=false
fi
launch_job_name="${RUN_ID}-coord-${short_uuid}"
launch_record=$(printf 'Hero launch requested.\n\n- Run ID: `%s`\n- Commit: `%s`\n- Tree dirty: `%s`\n- Coordinator job: `%s`\n- Target: `%s` (%s)' \
  "$RUN_ID" "$launch_commit" "$launch_tree_dirty" "$launch_job_name" "$TARGET_CLUSTER" "$TARGET_DESCRIPTION")

echo "Recording hero launch on ${HERO_ISSUE}"
gh issue comment "$HERO_ISSUE" --body "$launch_record"
echo "Launching hero from commit ${launch_commit}; tree_dirty=${launch_tree_dirty}"

uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster "$TARGET_CLUSTER" \
  --priority system \
  --system-reason "hero run" \
  --cpu 1 \
  --memory 4g \
  --max-retries 1000 \
  --job-name "$launch_job_name" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_PROJECT marin_moe \
  -e IRIS_PORT_JAX 32614 \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.75 \
  -e XLA_FLAGS "--xla_gpu_memory_limit_slop_factor=85" \
  -- python -m experiments.grug.moe_hero_ep.launch_scaling_ladder \
    --run-id "$RUN_ID" \
    --initialize-from-checkpoint "$HANDOFF_CHECKPOINT" \
    --size d6144 \
    --version 2026.08.19.2 \
    --run
