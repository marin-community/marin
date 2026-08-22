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
# Outer runtime backstop, enforced by Iris on the coord job. The real limit lives in watchdog.sh:
# a 15-minute step budget counted from the first observed step, plus a 20-minute compile ceiling.
# This value must cover their sum, so it only fires if the watchdog itself is dead.
ARM_TIMEOUT="${ARM_TIMEOUT:-2100}"
# `stop_after_steps` is an absolute step, not a count, so a restored arm resuming at the
# checkpoint's step would exit immediately against a small value. Restore arms set this high and
# let the runtime cap end them instead.
NUM_STEPS="${NUM_STEPS:-30}"
# Synthetic data is a single deterministic batch, which is fine for an untrained router but not
# for a restored one: a trained router routes real tokens by content, and repeating one degenerate
# batch would produce a routing distribution -- and a drop rate -- that no real run ever sees.
TRAINING_DATA="${TRAINING_DATA:-synthetic}"
MASTER_PARAMS="${MASTER_PARAMS:-disabled}"
MOE_IMPL="${MOE_IMPL:-ragged_all_to_all}"
# Restore arms initialize from the live hero's checkpoint so the router is trained and capacity
# clipping is real. The hero writes its master in fp32 on pinned host; the run folds that into
# fp32 device parameters, so it trains in the configuration the PR ships.
RESTORE_FROM="${RESTORE_FROM:-}"
RESTORE_ARGS=()
if [ -n "$RESTORE_FROM" ]; then
  RESTORE_ARGS=(--restore-from "$RESTORE_FROM" --restore-master-params fp32_pinned_host)
fi
WHEEL="${WHEEL:-s3://marin-us-east-02a/marin/research/mcwitt-ra2a/pjrt-kmax128-devkernel-merge47263-20260817/jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl}"
# The learning-rate schedule's length. The harrier mix rejects a schedule whose token budget
# exceeds the mix's own (18.75T at 4.19M tokens/step, so 4.47M steps), and the check runs at
# fingerprint time even for a synthetic-data run. Same value for every arm: at step 20 the LR is
# deep in warmup either way, and holding it fixed keeps losses comparable.
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
  -e MARIN_PJRT_WHEEL "${WHEEL}" \
  -e IRIS_PORT_JAX 32703 \
  -e AWS_MAX_ATTEMPTS 25 -e AWS_RETRY_MODE adaptive \
  -e JAX_COMPILATION_CACHE_DIR "s3://marin-us-east-02a/marin/tmp/ttl=30d/jaxcache/${RID}" \
  -- python -m experiments.grug.moe_hero_ep.launch_mfu_test \
     --run-id "${RID}" \
     --dp-racks 1 --num-steps "${NUM_STEPS}" --schedule-steps "${SCHEDULE_STEPS}" \
     --capacity-factor 1.15 \
     --moe-implementation "${MOE_IMPL}" \
     --processes-per-task 4 \
     --master-params "${MASTER_PARAMS}" \
     --training-data "${TRAINING_DATA}" \
     --profile-steps 0 \
     --watch-interval 0 --eval-every 0 \
     --no-save-checkpoints \
     "${RESTORE_ARGS[@]}" \
     "${EXTRA[@]}" \
     --version "${VERSION}" --run >/dev/null 2>&1

# Every knob that can change a number, recorded per arm. An arm is reproducible only if it can be
# reissued from this row: the commit fixes the code, the wheel URL is an immutable object, the
# synthetic batch is a pure function of the seed, and the schedule length fixes the learning rate.
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${RID}" "$(git -C "$REPO" rev-parse --short HEAD)" "${VERSION}" "${MOE_IMPL}" "${TRAINING_DATA}" \
  "${MASTER_PARAMS}" "${SCHEDULE_STEPS}" "${ARM_TIMEOUT}" "${RESTORE_FROM:-none}" \
  "${EXTRA_LAUNCH_ARGS:-none}" >> "${LOOP_DIR}/arms.tsv"

echo "submitted ${RID} at ${PRIORITY} priority (timeout ${ARM_TIMEOUT}s, impl ${MOE_IMPL}, data ${TRAINING_DATA}, restore ${RESTORE_FROM:-none}, extra: ${EXTRA_LAUNCH_ARGS:-none})"
