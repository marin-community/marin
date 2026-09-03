#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY before you start the hero.}"

# Gate/router weight-decay continuation: forks the hero's full state from its step-58014 checkpoint
# under its own run id and tree, and trains with the decay on by default (0.02, annealed).
RUN_ID=hero-wd-gate-router-p02-step58k
HANDOFF_CHECKPOINT=s3://hero-checkpoints/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-58014
short_uuid=$(uuidgen | tr '[:upper:]' '[:lower:]')
short_uuid=${short_uuid:0:8}

uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a \
  --priority system \
  --system-reason "hero run" \
  --cpu 1 \
  --memory 4g \
  --max-retries 1000 \
  --job-name "${RUN_ID}-coord-${short_uuid}" \
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
