#!/usr/bin/env bash
# Submit one hero EP64 arm for the 8753-mfu loop (research/mcwitt/8753-mfu-loop, #8753 head base).
#
# Protocol (fixed for every iteration so arms stay comparable; see DESIGN.md):
#   - one NVL72 rack on cw-us-east-08a, PRODUCTION priority allowed with job runtime <1h
#     (user 2026-08-28): ARM_TIMEOUT 3500s outer backstop, watchdog.sh COMPILE_CEILING=1800 +
#     STEP_BUDGET=1200 inner caps (20-min post-compile cap)
#   - restore from the live hero's step-30000 checkpoint, mixture data (trained-router routing);
#     NUM_STEPS is the ABSOLUTE stop step (checkpoint step + window), never a relative count
#   - no profiler, no checkpoints, no eval
#   - no MARIN_PJRT_WHEEL overlay: the xla-fork base's pinned wheel runs ragged EP64 (validated
#     by the t8684-* arms 2026-08-28); COORD_SKIP_JAX_FLOOR=1 keeps the coordinator's stock jax
#   - JAX_COMPILATION_CACHE_DIR rotated per run id (clique-init deadlock dodge)
#
# usage: RID=<run-id> VERSION=<calver> [REPO=<worktree>] [EXTRA_LAUNCH_ARGS="..."] arm.sh
set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
RID="${RID:?set RID}"
VERSION="${VERSION:?set VERSION -- bump it per arm or the artifact layer reuses the last run}"

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${LOOP_DIR}/../.." && pwd)}"

PRIORITY="${PRIORITY:-production}"
ARM_TIMEOUT="${ARM_TIMEOUT:-3500}"
NUM_STEPS="${NUM_STEPS:-30030}"
TRAINING_DATA="${TRAINING_DATA:-mixture}"
MASTER_PARAMS="${MASTER_PARAMS:-disabled}"
MOE_IMPL="${MOE_IMPL:-ragged_all_to_all}"
RESTORE_FROM="${RESTORE_FROM:-s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-30000}"
RESTORE_ARGS=()
if [ -n "$RESTORE_FROM" ]; then
  RESTORE_ARGS=(--restore-from "$RESTORE_FROM")
fi
# The learning-rate schedule's length; fixed across arms so losses stay comparable.
SCHEDULE_STEPS=4470000

read -r -a EXTRA <<<"${EXTRA_LAUNCH_ARGS:-}"

cd "$REPO"
uv run iris --config lib/iris/config/marin.yaml job run \
  --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority "${PRIORITY}" \
  --cpu 2 --memory 8GB --disk 32GB --timeout "${ARM_TIMEOUT}" \
  --job-name "${RID}-coord" \
  -e IRIS_USER mwittmann \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e WANDB_PROJECT marin_moe \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -e COORD_SKIP_JAX_FLOOR 1 \
  -e IRIS_PORT_JAX 32703 \
  -e AWS_MAX_ATTEMPTS 25 -e AWS_RETRY_MODE adaptive \
  -e JAX_COMPILATION_CACHE_DIR "s3://marin-us-east-02a/marin/tmp/ttl=30d/jaxcache/${RID}" \
  -e XLA_FLAGS "${ARM_XLA_FLAGS:-}" \
  -- python -m experiments.grug.moe_hero_ep.launch_diagnostics \
     --run-id "${RID}" \
     --dp-racks 1 --num-steps "${NUM_STEPS}" --schedule-steps "${SCHEDULE_STEPS}" \
     --capacity-factor 1.15 \
     --moe-implementation "${MOE_IMPL}" \
     --processes-per-task 4 \
     --master-params "${MASTER_PARAMS}" \
     --training-data "${TRAINING_DATA}" \
     --profile-steps "${PROFILE_STEPS:-0}" --profile-start-step "${PROFILE_START_STEP:-5}" \
     --watch-interval 0 --eval-every 0 \
     --no-save-checkpoints \
     "${RESTORE_ARGS[@]}" \
     "${EXTRA[@]}" \
     --version "${VERSION}" --run >/dev/null 2>&1

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${RID}" "$(git -C "$REPO" rev-parse --short HEAD)" "${VERSION}" "${MOE_IMPL}" "${TRAINING_DATA}" \
  "${MASTER_PARAMS}" "${SCHEDULE_STEPS}" "${ARM_TIMEOUT}" "${RESTORE_FROM:-none}" \
  "${EXTRA_LAUNCH_ARGS:-none}" >> "${LOOP_DIR}/arms.tsv"

echo "submitted ${RID} at ${PRIORITY} priority (timeout ${ARM_TIMEOUT}s, commit $(git -C "$REPO" rev-parse --short HEAD), data ${TRAINING_DATA}, extra: ${EXTRA_LAUNCH_ARGS:-none})"
