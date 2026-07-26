#!/usr/bin/env bash
# ep25-d5 submission helper. Usage:
#   ./submit_d5.sh smoke-bf16 | smoke-mxfp8 | rack-bf16 | rack-mxfp8 | rack-e128 [suffix]
# Everything is the d1 QB-on cf1.0 custom-adjoint control config, moved to the d6144 shape.
set -euo pipefail
cd "$(dirname "$0")"

MODE="$1"
SUFFIX="${2:-$(date +%m%d-%H%M)}"
VERSION="ep25d5-dev"
# Memory ladder for the 707B shape. ALLOC=cuda_async (default) sidesteps the BFC fraction entirely;
# ALLOC=bfc uses the BFC allocator at MEM_FRACTION instead. Record whichever was used with the result.
ALLOC="${ALLOC:-cuda_async}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
OFFLOAD="${OFFLOAD:-0}"

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

# The cuda_async allocator honors XLA_PYTHON_CLIENT_MEM_FRACTION too: the first d6144 leg reported
# "Limit: 138.22GiB" = 0.75 x 184.3 GiB physical. So the two knobs compose and both are always passed.
MEM_ENV=(-e XLA_PYTHON_CLIENT_MEM_FRACTION "$MEM_FRACTION")
[[ "$ALLOC" == cuda_async ]] && MEM_ENV+=(-e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async)
[[ "$OFFLOAD" == 1 ]] && MEM_ENV+=(-e SCALE_OFFLOAD_OPT_STATE 1)

case "$MODE" in
  # 1 node / 4 GPUs, EP4. 64 experts x top-4 keeps the per-(sender,expert) bucket mean at 1024,
  # the same as the rack target, and d6144/i3072 keeps the real expert-GEMM shape.
  smoke-bf16|smoke-mxfp8)
    NAME="ep25d5-${MODE}-${SUFFIX}"
    SHAPE=(-e SCALE_GPU_REPLICAS 1 -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 64 -e SCALE_TOP_K 4
           -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 4 -e SCALE_BATCH 16 -e SCALE_STEPS 40)
    [[ "$MODE" == smoke-mxfp8 ]] && SHAPE+=(-e SCALE_MOE_MXFP8 1)
    ;;
  # 16 nodes / 64 GPUs, EP64. The honest baseline at the hero-run candidate shape.
  rack-bf16|rack-mxfp8)
    NAME="ep25d5-d6144-e256-${MODE#rack-}-120-${SUFFIX}"
    SHAPE=(-e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 4
           -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 48 -e SCALE_BATCH 1024 -e SCALE_STEPS 120
           "${MEM_ENV[@]}")
    [[ "$MODE" == rack-mxfp8 ]] && SHAPE+=(-e SCALE_MOE_MXFP8 1)
    ;;
  # Fallback primary shape if 707B does not fit: 4-of-128, 359.6B total, same 22.6B active.
  rack-e128)
    NAME="ep25d5-d6144-e128-bf16-120-${SUFFIX}"
    SHAPE=(-e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4
           -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 48 -e SCALE_BATCH 1024 -e SCALE_STEPS 120
           "${MEM_ENV[@]}")
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
