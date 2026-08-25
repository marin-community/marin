#!/usr/bin/env bash
# Updates-per-peer sweep: does the dk per-byte rate depend on update granularity?
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
PFX=s3://marin-us-east-02a/marin/research/mcwitt-ra2a
W1=jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl
DKFLAGS="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"
OSFLAGS="--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=true"

run_cell() {
  local name="$1" wheel="$2" flags="$3" upp="$4"
  NAME="$name" WHEEL="$wheel" FLAGS="$flags" BENCH_UPDATES_PER_PEER="$upp" bash "${LOOP_DIR}/bench_submit.sh" >/dev/null 2>&1
  cd "$REPO"
  for _ in 1 2 3; do
    timeout 900 uv run iris --config lib/iris/config/marin.yaml job wait "/mwittmann/${name}" >/dev/null 2>&1 && break
  done
  local line val
  line=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "per-call" 2>/dev/null | grep -o "per-call.*" | tail -1)
  val=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "VALIDATION" 2>/dev/null | grep -o "VALIDATION.*" | sort | uniq -c | tr '\n' ';')
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%H:%M)" "$name" "${line:-NO-RESULT}" "${val:-NO-VALIDATION}" >> "${LOOP_DIR}/ladder_results.tsv"
}

run_cell "dkp-usw-dk-u1"    "${PFX}/pjrt-mainpatch-g8x128-20260823/${W1}" "$DKFLAGS" 1
run_cell "dkp-usw-dk-u120"  "${PFX}/pjrt-mainpatch-g8x128-20260823/${W1}" "$DKFLAGS" 120
run_cell "dkp-usw-os-u1"    "${PFX}/pjrt-mainpatch-kmax128-20260823/${W1}" "$OSFLAGS" 1
echo USWEEP_DONE
