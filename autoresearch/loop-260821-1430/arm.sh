#!/usr/bin/env bash
# Submit one hero EP64 arm for the ragged-transport tuning loop.
#
# Protocol (fixed for every iteration so arms stay comparable):
#   - one NVL72 rack on cw-us-east-08a, production priority, never more than one at a time
#   - synthetic data: the loader never opens TensorStore, which is what makes a 15-minute arm
#     feasible. The batch is deterministic, so loss stays comparable across arms and the drop
#     counts still reflect the routing the model produces.
#   - no profiler, no checkpoints, no eval: every second inside the limit goes to steps.
#   - the kmax128 PJRT wheel, installed on the train tasks via MARIN_PJRT_WHEEL. Stock XLA
#     writes 64 peer slots into MultiGpuBarrierKernel's 32-slot allocation on this path and
#     every rank dies before step 0 (openxla/xla#47283).
#   - JAX_COMPILATION_CACHE_DIR is rotated per run id on purpose. Sharing it across restarts is
#     the confirmed trigger for the GB200 clique-init deadlock, and a hung arm costs more than
#     the compile it saves.
#
# usage: RID=<run-id> VERSION=<calver> [EXTRA_LAUNCH_ARGS="..."] arm.sh
set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
RID="${RID:?set RID}"
VERSION="${VERSION:?set VERSION -- bump it per iteration or the artifact layer reuses the last run}"

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"

PRIORITY="${PRIORITY:-production}"
# Hard runtime limit. Iris enforces it on the coord job; watchdog.sh enforces it out of process.
ARM_TIMEOUT="${ARM_TIMEOUT:-900}"
NUM_STEPS="${NUM_STEPS:-30}"
MASTER_PARAMS="${MASTER_PARAMS:-disabled}"
WHEEL="${WHEEL:-s3://marin-us-east-02a/marin/research/mcwitt-ra2a/pjrt-kmax128-devkernel-merge47263-20260817/jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl}"
SCHEDULE_STEPS=19073486

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
  -e MARIN_PJRT_WHEEL "${WHEEL}" \
  -e IRIS_PORT_JAX 32703 \
  -e AWS_MAX_ATTEMPTS 25 -e AWS_RETRY_MODE adaptive \
  -e JAX_COMPILATION_CACHE_DIR "s3://marin-us-east-02a/marin/tmp/ttl=30d/jaxcache/${RID}" \
  -- python -m experiments.grug.moe_hero_ep.launch_mfu_test \
     --run-id "${RID}" \
     --dp-racks 1 --num-steps "${NUM_STEPS}" --schedule-steps "${SCHEDULE_STEPS}" \
     --capacity-factor 1.15 \
     --moe-implementation ragged_all_to_all \
     --processes-per-task 4 \
     --master-params "${MASTER_PARAMS}" \
     --training-data synthetic \
     --profile-steps 0 \
     --watch-interval 0 --eval-every 0 \
     --no-save-checkpoints \
     "${EXTRA[@]}" \
     --version "${VERSION}" --run >/dev/null 2>&1

echo "submitted ${RID} at ${PRIORITY} priority (timeout ${ARM_TIMEOUT}s, extra: ${EXTRA_LAUNCH_ARGS:-none})"
