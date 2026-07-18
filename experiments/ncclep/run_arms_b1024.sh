#!/bin/bash
# Sequential b1024 arms in ONE gang allocation (issue #7331, NCCLEP-007).
# Discriminates the intermittent CUBIN-load bug (B200MFU-036: allocation-
# correlated) from a chunked-program-specific failure: if the a2a control
# passes and every chunked arm fails, the chunked program is deterministically
# bad; if the control also fails, the allocation is a bad draw.
set -ux

COMMON=(--output-dir /tmp/out --num-gpus 64 --steps 20 --num-layers 48
        --hidden-dim 5120 --batch-size 1024 --expert-parallelism 8)

declare -A RESULTS
run_arm() {
  local name=$1; shift
  bash "$(dirname "$0")/run_bench_gang.sh" --run-id "$name" "${COMMON[@]}" "$@"
  RESULTS[$name]=$?
  echo "ARM-RESULT $name rc=${RESULTS[$name]}"
}

run_arm b1024-a2a-ctl2 --moe-implementation ragged_all_to_all_cute
run_arm b1024-nccl-ck8k --moe-implementation nccl_ep --ep-chunk-tokens 8192
run_arm b1024-nccl-ck16k --moe-implementation nccl_ep --ep-chunk-tokens 16384
run_arm b1024-nccl-ck4k --moe-implementation nccl_ep --ep-chunk-tokens 4096

echo "ARM-SUMMARY: $(for k in "${!RESULTS[@]}"; do printf '%s=%s ' "$k" "${RESULTS[$k]}"; done)"
# Job is "green" if any chunked arm passed OR the discriminator ran cleanly;
# report failure only if nothing at all succeeded.
for k in "${!RESULTS[@]}"; do [ "${RESULTS[$k]}" = 0 ] && exit 0; done
exit 1
