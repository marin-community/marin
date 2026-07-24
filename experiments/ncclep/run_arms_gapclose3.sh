#!/bin/bash
# Gap-closure knob arms on the known-good 68493d2 wheel (issue #7331,
# NCCLEP-010c). The ea41e08 tip wheel deterministically crashes BOTH TE
# integrations at 64-GPU data8xexpert8 (shim-independent, so the regression
# is in the nccl-extensions swap, not just the #3231 stream pin) — take the
# tuning-knob readouts on the wheel that trains. 68493d2 predates the
# collective-stream pin entirely, so no shim is needed here.
set -ux

export NCCLEP_BENCH_ATTEMPTS=${NCCLEP_BENCH_ATTEMPTS:-2}
export NCCLEP_WHEEL_PATTERN=68493d2

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

run_arm gc3-a2a-anchor --moe-implementation ragged_all_to_all_cute
run_arm gc3-te-moe-base --moe-implementation te_moe
NCCLEP_CMD_BUFFER="$SCOPED_CB" \
  run_arm gc3-te-moe-cb --moe-implementation te_moe
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc3-te-moe-cb-ms --moe-implementation te_moe
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc3-te-moe-cb-ms-sms16 --moe-implementation te_moe --ep-max-num-sms 16
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc3-te-moe-cb-ms-sms32 --moe-implementation te_moe --ep-max-num-sms 32
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc3-nccl-ep-cb-ms --moe-implementation nccl_ep

echo "ARM-SUMMARY: $(for k in "${!RESULTS[@]}"; do printf '%s=%s ' "$k" "${RESULTS[$k]}"; done)"
for k in "${!RESULTS[@]}"; do [ "${RESULTS[$k]}" = 0 ] && exit 0; done
exit 1
