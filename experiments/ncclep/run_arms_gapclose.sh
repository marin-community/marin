#!/bin/bash
# Gap-closure arms in ONE gang allocation (issue #7331, NCCLEP-010).
# Closes the non-quantization gaps against NVIDIA's recommended EP path, all
# bf16: TE wheel rebuilt at main tip (#3231 EP ops on XLA's collective
# stream, #3226), command-buffer capture scoped instead of globally off,
# multi-stream collective overlap, and an ep_bootstrap max_num_sms sweep.
# a2a_cute gets its own tuned arm (default capture + overlap) so the
# incumbent is also measured at its best config, not just the TE-constrained
# one.
set -ux

export NCCLEP_BENCH_ATTEMPTS=${NCCLEP_BENCH_ATTEMPTS:-2}

COMMON=(--output-dir /tmp/out --num-gpus 64 --steps 20 --num-layers 48
        --hidden-dim 5120 --batch-size 512 --expert-parallelism 8)

# XLA default capture set minus CUSTOM_CALL: EP FFI ops (whose host-side
# handle bookkeeping breaks under capture, NCCLEP-005) and cutlass_call ops
# stay eager while fusions/GEMMs are captured.
SCOPED_CB="FUSION,CUBLAS,CUBLASLT,CUDNN,DYNAMIC_SLICE_FUSION"
MS_FLAG="--xla_gpu_experimental_parallel_collective_overlap_limit=2"

declare -A RESULTS
run_arm() {
  local name=$1; shift
  bash "$(dirname "$0")/run_bench_gang.sh" --run-id "$name" "${COMMON[@]}" "$@"
  RESULTS[$name]=$?
  echo "ARM-RESULT $name rc=${RESULTS[$name]}"
}

# Anchors: same flags as NCCLEP-009, isolating the tip-wheel effect (#3231).
run_arm gc-a2a-base --moe-implementation ragged_all_to_all_cute
run_arm gc-te-moe-base --moe-implementation te_moe

# Tuned incumbent: default capture (incl. CUSTOM_CALL) + collective overlap.
NCCLEP_CMD_BUFFER=default NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc-a2a-tuned --moe-implementation ragged_all_to_all_cute

# Tuned TE block: scoped capture, then + overlap, then the SM-budget sweep.
NCCLEP_CMD_BUFFER="$SCOPED_CB" \
  run_arm gc-te-moe-cb --moe-implementation te_moe
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc-te-moe-cb-ms --moe-implementation te_moe
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc-te-moe-cb-ms-sms16 --moe-implementation te_moe --ep-max-num-sms 16
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc-te-moe-cb-ms-sms32 --moe-implementation te_moe --ep-max-num-sms 32

# Best-tuned seam for reference (does the transport-only integration also
# gain from tip + scoping + overlap?).
NCCLEP_CMD_BUFFER="$SCOPED_CB" NCCLEP_EXTRA_XLA_FLAGS="$MS_FLAG" \
  run_arm gc-nccl-ep-cb-ms --moe-implementation nccl_ep

echo "ARM-SUMMARY: $(for k in "${!RESULTS[@]}"; do printf '%s=%s ' "$k" "${RESULTS[$k]}"; done)"
for k in "${!RESULTS[@]}"; do [ "${RESULTS[$k]}" = 0 ] && exit 0; done
exit 1
