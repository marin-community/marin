#!/usr/bin/env bash
# ep25-d6 submission helper. Usage:
#   ./submit_d6.sh smoke-dense | smoke-latent | rack-dense | rack-latent [suffix]
#
# Every arm is d5's hero-shape configuration (submit_d5.sh rack-e128 with ALLOC=bfc OFFLOAD=1),
# which produced the 24.594% reference leg. The latent arms change exactly two things: the expert
# count doubles (128 -> 256) and SCALE_MOE_LATENT_DIM halves the dispatch width (6144 -> 3072), so
# routed parameters are preserved (347.892B either way) while the all-to-all payload halves.
#
# Latent moves the analytic FLOPs/token: 48.186 G dense vs 41.014 G latent. Report tok/s beside MFU.
set -euo pipefail
cd "$(dirname "$0")"

MODE="$1"
SUFFIX="${2:-$(date +%m%d-%H%M)}"
VERSION="ep25d6-dev"
# The reference leg ran the DEFAULT BFC allocator at the DEFAULT 0.75 fraction with host offload.
# Do NOT raise the fraction: NCCL's transport buffers live outside the XLA arena, and 0.90 starved
# them into "ncclAlltoAll ... Cuda failure 2 'out of memory'" on this exact shape.
ALLOC="${ALLOC:-bfc}"
MEM_FRACTION="${MEM_FRACTION-}"   # empty = leave the 0.75 default alone
OFFLOAD="${OFFLOAD:-1}"
# >0 captures a jax profile window of that many steps starting at SCALE_PROFILER_START.
PROFILE_STEPS="${PROFILE_STEPS:-0}"

COMMON_ENV=(
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1
  -e SCALE_A2A_GATHER_DISPATCH 1 -e SCALE_A2A_CUSTOM_ADJOINT 1
  -e SCALE_MOE_QB 1 -e SCALE_REPORT_DROPS 1
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200
  -e SCALE_SEQ_LEN 4096 -e SCALE_SLIDING_WINDOW 2048
  -e SCALE_MOE_IMPL ragged_all_to_all -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all
  -e SCALE_DISABLE_CHECKPOINT 1 -e SCALE_TRACKER json_logger
)

MEM_ENV=()
[[ -n "$MEM_FRACTION" ]] && MEM_ENV+=(-e XLA_PYTHON_CLIENT_MEM_FRACTION "$MEM_FRACTION")
[[ "$ALLOC" == cuda_async ]] && MEM_ENV+=(-e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async)
[[ "$OFFLOAD" == 1 ]] && MEM_ENV+=(-e SCALE_OFFLOAD_OPT_STATE 1)
PROFILE_ENV=()
[[ "$PROFILE_STEPS" != 0 ]] && PROFILE_ENV+=(-e SCALE_PROFILER_STEPS "$PROFILE_STEPS"
                                             -e SCALE_PROFILER_START "${PROFILE_START:-20}")

case "$MODE" in
  # 1 node / 4 GPUs, EP4, 4 layers. The smoke pair mirrors the rack pair's routing regime: the
  # latent arm doubles the expert count, so its per-(sender,expert) bucket mean halves exactly as
  # it does at the rack (1024 -> 512 here; 2048 -> 1024 there).
  smoke-dense)
    NAME="ep25d6-smoke-dense-${SUFFIX}"
    SHAPE=(-e SCALE_GPU_REPLICAS 1 -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 64 -e SCALE_TOP_K 4
           -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 4 -e SCALE_BATCH 16 -e SCALE_STEPS 40)
    ;;
  smoke-latent)
    NAME="ep25d6-smoke-latent-${SUFFIX}"
    SHAPE=(-e SCALE_GPU_REPLICAS 1 -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4
           -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 4 -e SCALE_BATCH 16 -e SCALE_STEPS 40
           -e SCALE_MOE_LATENT_DIM 3072)
    ;;
  # 16 nodes / 64 GPUs, EP64. rack-dense reproduces d5's 24.594% reference leg byte-for-byte.
  rack-dense)
    NAME="ep25d6-d6144-e128-dense-120-${SUFFIX}"
    SHAPE=(-e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4
           -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 48 -e SCALE_BATCH 1024 -e SCALE_STEPS 120
           "${MEM_ENV[@]}" "${PROFILE_ENV[@]}")
    ;;
  # The arm that matters: routed params preserved, dispatch width halved.
  rack-latent)
    NAME="ep25d6-d6144-e256-latent3072-120-${SUFFIX}"
    SHAPE=(-e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 4
           -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 48 -e SCALE_BATCH 1024 -e SCALE_STEPS 120
           -e SCALE_MOE_LATENT_DIM 3072
           "${MEM_ENV[@]}" "${PROFILE_ENV[@]}")
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

set -x
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name "$NAME" -e RUN_ID "$NAME" \
  "${COMMON_ENV[@]}" "${SHAPE[@]}" \
  -e SCALE_JSON_LOGGER "$NAME.metrics" \
  -- python -m experiments.grug.moe.launch_cw_scale --version "$VERSION" --run
