#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 RUN_ID COORDINATOR_PORT [STEPS]" >&2
  exit 2
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY before submitting}"

run_id="$1"
coordinator_port="$2"
steps="${3:-20}"
profile_steps="${PROFILE_STEPS:-0}"
profile_start="${PROFILE_START:-8}"

xla_flags=(
  "--xla_gpu_enable_latency_hiding_scheduler=true"
  "--xla_gpu_experimental_parallel_collective_overlap_limit=4"
)
profile_args=()
if (( profile_steps > 0 )); then
  xla_flags+=("--xla_gpu_enable_command_buffer=")
  profile_args=(
    -e SCALE_PROFILER_STEPS "$profile_steps"
    -e SCALE_PROFILER_START "$profile_start"
    -e SCALE_PROFILER_PROCESS_INDEX 0
    -e SCALE_PROFILER_PROCESS_COUNT 4
  )
fi

.venv/bin/iris --cluster=marin job run \
  --no-wait \
  --max-retries 0 \
  --memory 4GB \
  --enable-extra-resources \
  --target-cluster cw-us-east-08a \
  --priority interactive \
  --job-name "$run_id" \
  -e RUN_ID "$run_id" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY marin-community \
  -e WANDB_PROJECT rav_moe \
  -e SCALE_TRACKER wandb \
  -e SCALE_MAX_RETRIES_FAILURE 0 \
  -e SCALE_MAX_RETRIES_PREEMPTION 0 \
  -e SCALE_MAX_TASK_FAILURES 0 \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -e SCALE_CHECKPOINTS local \
  -e SCALE_WATCH_INTERVAL 0 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_A2A_FIXED 1 \
  -e SCALE_A2A_CHUNKS 1 \
  -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_SAME_EXPERT_CLONES 1 \
  -e SCALE_A2A_CLONE_POOLED 1 \
  -e SCALE_A2A_CLONE_SPARSE_WEIGHTS 1 \
  -e SCALE_A2A_ECHO_CLONES 1 \
  -e SCALE_A2A_CLONE_MAX_RECEIVER_EXPERTS 10 \
  -e SCALE_A2A_CLONE_TOKEN_PADDING_EXPERTS 2 \
  -e SCALE_A2A_SONIC_DISPATCH 1 \
  -e SCALE_A2A_CLONE_SONIC_SLOT_GATHER 1 \
  -e SCALE_A2A_SONIC_COMBINE 1 \
  -e SCALE_A2A_CLONE_SONIC_CUTE 1 \
  -e SCALE_A2A_CLONE_PIPELINE_CHUNKS 2 \
  -e SCALE_QUACK_GROUPED_WGRAD 1 \
  -e SCALE_PROCESSES_PER_TASK 4 \
  -e SCALE_GPUS_PER_NODE 4 \
  -e SCALE_GPU_TYPE GB200 \
  -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 \
  -e SCALE_NUM_EXPERTS 256 \
  -e SCALE_TOP_K 4 \
  -e SCALE_HIDDEN_DIM 5120 \
  -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 2048 \
  -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_NUM_SHARED_EXPERTS 1 \
  -e SCALE_SEQ_LEN 4096 \
  -e SCALE_BATCH 1024 \
  -e SCALE_SLIDING_WINDOW 512 \
  -e SCALE_STEPS "$steps" \
  -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_OPTIMIZER muonh \
  -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 \
  -e SCALE_SCAN_UNROLL 1 \
  -e SCALE_REMAT recompute_all \
  -e SCALE_REPORT_CAPACITY_OVERFLOW 1 \
  -e SCALE_MOE_QB 1 \
  -e JAX_PJRT_CLIENT_CREATE_OPTIONS allocator:cuda_async \
  -e JAX_ENABLE_PGLE true \
  -e JAX_PGLE_PROFILING_RUNS 3 \
  -e JAX_PGLE_AGGREGATION_PERCENTILE 85 \
  -e JAX_COORDINATOR_PORT "$coordinator_port" \
  -e XLA_FLAGS "${xla_flags[*]}" \
  "${profile_args[@]}" \
  -- python -m experiments.grug.moe.launch_cw_scale --version dev --run
