#!/bin/bash
# Sequential b512 arms in ONE gang allocation (issue #7331, NCCLEP-009).
# Same-allocation three-way comparison at the largest config that fits
# NCCL_EP's no-drop capacity unchunked: a2a_cute (incumbent), nccl_ep
# (TE dispatch/combine around the QuACK FFN — the NCCLEP-006 seam), and
# te_moe (NVIDIA's recommended full TE fused MoE block). One allocation
# controls for the placement variance and the B200MFU-036 CUBIN envelope.
set -ux

# In-job sequential retries absorb the intermittent CUBIN-load failure
# (B200MFU-036); the shared compile cache makes warm retries ~10 min cheaper.
export NCCLEP_BENCH_ATTEMPTS=${NCCLEP_BENCH_ATTEMPTS:-3}

COMMON=(--output-dir /tmp/out --num-gpus 64 --steps 20 --num-layers 48
        --hidden-dim 5120 --batch-size 512 --expert-parallelism 8)

declare -A RESULTS
run_arm() {
  local name=$1; shift
  bash "$(dirname "$0")/run_bench_gang.sh" --run-id "$name" "${COMMON[@]}" "$@"
  RESULTS[$name]=$?
  echo "ARM-RESULT $name rc=${RESULTS[$name]}"
}

run_arm b512-te-moe --moe-implementation te_moe
run_arm b512-nccl-ep --moe-implementation nccl_ep
run_arm b512-a2a-ctl --moe-implementation ragged_all_to_all_cute

echo "ARM-SUMMARY: $(for k in "${!RESULTS[@]}"; do printf '%s=%s ' "$k" "${RESULTS[$k]}"; done)"
for k in "${!RESULTS[@]}"; do [ "${RESULTS[$k]}" != 0 ] && exit 1; done
exit 0
