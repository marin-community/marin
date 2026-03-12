#!/usr/bin/env bash
set -euo pipefail

args=()

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env var: $name" >&2
    exit 1
  fi
}

add_arg() {
  local flag="$1"
  local env_name="$2"
  if [[ -n "${!env_name:-}" ]]; then
    args+=("$flag" "${!env_name}")
  fi
}

add_flag_if_truthy() {
  local flag="$1"
  local env_name="$2"
  if [[ "${!env_name:-}" == "1" ]]; then
    args+=("$flag")
  fi
}

require_env EXPERT_AXIS_SIZE

args+=("--expert-axis-size" "${EXPERT_AXIS_SIZE}")
add_arg "--batch-size" BATCH_SIZE
add_arg "--capacity-factor" CAPACITY_FACTOR
add_arg "--block-remat" BLOCK_REMAT
add_arg "--num-experts" NUM_EXPERTS
add_arg "--num-experts-per-token" NUM_EXPERTS_PER_TOKEN
add_arg "--intermediate-dim" INTERMEDIATE_DIM
add_arg "--shared-expert-intermediate-dim" SHARED_EXPERT_INTERMEDIATE_DIM
add_arg "--steps" STEPS
add_arg "--profiler-start-step" PROFILER_START_STEP
add_arg "--profiler-num-steps" PROFILER_NUM_STEPS
add_arg "--cross-entropy-implementation" CROSS_ENTROPY_IMPLEMENTATION
add_arg "--cross-entropy-v-block-divisor" CROSS_ENTROPY_V_BLOCK_DIVISOR
add_arg "--hidden-dim" HIDDEN_DIM
add_arg "--num-layers" NUM_LAYERS
add_arg "--num-heads" NUM_HEADS
add_arg "--num-kv-heads" NUM_KV_HEADS
add_arg "--run-suffix" RUN_SUFFIX

add_flag_if_truthy "--match-activated-params" MATCH_ACTIVATED_PARAMS
add_flag_if_truthy "--match-total-active-flops" MATCH_TOTAL_ACTIVE_FLOPS
add_flag_if_truthy "--block-shuffle" BLOCK_SHUFFLE
add_flag_if_truthy "--synthetic-data" SYNTHETIC_DATA
add_flag_if_truthy "--report-capacity-overflow" REPORT_CAPACITY_OVERFLOW

python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py "${args[@]}"
