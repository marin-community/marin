#!/bin/bash
# Gap-closure follow-up (issue #7331, NCCLEP-010b): the ea41e08 tip wheel's
# collective-stream pin (#3231) deterministically kills 64-GPU first
# execution at data8xexpert8 (ncclCommSplit "remote process exited"; 2- and
# 4-node topologies pass). Rerun the TE arms with the pin stripped
# (NCCLEP_DISABLE_COLLECTIVE_STREAM=1) plus an a2a anchor for this
# allocation.
set -ux

export NCCLEP_BENCH_ATTEMPTS=${NCCLEP_BENCH_ATTEMPTS:-2}

COMMON=(--output-dir /tmp/out --num-gpus 64 --steps 20 --num-layers 48
        --hidden-dim 5120 --batch-size 512 --expert-parallelism 8)

SCOPED_CB="FUSION,CUBLAS,CUBLASLT,CUDNN,DYNAMIC_SLICE_FUSION"
MS_FLAG="--xla_gpu_experimental_parallel_collective_overlap_limit=2"

declare -A RESULTS
run_arm() {
  local name=$1; shift
  bash "$(dirname "$0")/run_bench_gang.sh" --run-id "$name" "${COMMON[@]}" "$@"
  RESULTS[$name]=$?
  echo "ARM-RESULT $name rc=${RESULTS[$name]}"
}

run_arm gc2-a2a-anchor --moe-implementation ragged_all_to_all_cute

NCCLEP_DISABLE_COLLECTIVE_STREAM=1 \
  run_arm gc2-te-moe-shim-base --moe-implementation te_moe
NCCLEP_DISABLE_COLLECTIVE_STREAM=1 NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc2-te-moe-shim-cb-ms --moe-implementation te_moe
NCCLEP_DISABLE_COLLECTIVE_STREAM=1 NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc2-te-moe-shim-cb-ms-sms32 --moe-implementation te_moe --ep-max-num-sms 32
NCCLEP_DISABLE_COLLECTIVE_STREAM=1 NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc2-nccl-ep-shim-cb-ms --moe-implementation nccl_ep

echo "ARM-SUMMARY: $(for k in "${!RESULTS[@]}"; do printf '%s=%s ' "$k" "${RESULTS[$k]}"; done)"
for k in "${!RESULTS[@]}"; do [ "${RESULTS[$k]}" = 0 ] && exit 0; done
exit 1
