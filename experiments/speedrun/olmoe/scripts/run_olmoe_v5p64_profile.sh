#!/usr/bin/env bash
# Example run (from repo root, keep tmp artifacts local):
#   TMPDIR=/tmp RAY_TMPDIR=/tmp ./experiments/speedrun/olmoe/scripts/run_olmoe_v5p64_profile.sh
#
# Convenience wrapper for launching an OLMoE speedrun with profiling enabled.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

: "${TMPDIR:=/tmp}"
: "${RAY_TMPDIR:=/tmp}"
: "${UV_CACHE_DIR:=/tmp/uv-cache}"

UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python -m marin.run.ray_run \
  --cluster "infra/marin-us-central1.yaml" \
  --extra tpu \
  --env_vars WANDB_MODE online \
  --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" \
  --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" \
  --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" \
  --env_vars HF_TOKEN "${HF_TOKEN:-}" \
  --env_vars PIP_NO_CACHE_DIR 1 \
  --env_vars RAY_TMPDIR "${RAY_TMPDIR}" \
  --env_vars TMPDIR "${TMPDIR}" \
  --env_vars JAX_COMPILATION_CACHE_DIR gs://marin-us-central1/jax-cache/olmoe_1b7b \
  --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 \
  --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 \
  --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 \
  -- \
  python experiments/speedrun/olmoe_1b7b_nemotron_40b.py \
    --model olmoe_1b7b \
    --dataset nemotron_cc \
    --tpu-type v5p-64 \
    --global-batch-size 512 \
    --num-train-steps 40 \
    --profile \
    --profile-start-step 15 \
    --profile-num-steps 20 \
    --force_run_failed true
