#!/usr/bin/env bash
# Follow-up cells: cta16 (correct 0.11.2 wheel name), tonight's one-shot reference, ilp4 retest.
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
PFX=s3://marin-us-east-02a/marin/research/mcwitt-ra2a
W1=jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl
W2=jax_cuda13_pjrt-0.11.2.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl
DKFLAGS="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"
OSFLAGS="--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=true"

run_cell() {
  local name="$1" wheel="$2" flags="$3"
  NAME="$name" WHEEL="$wheel" FLAGS="$flags" bash "${LOOP_DIR}/bench_submit.sh" >/dev/null 2>&1
  cd "$REPO"
  for _ in 1 2 3; do
    timeout 900 uv run iris --config lib/iris/config/marin.yaml job wait "/mwittmann/${name}" >/dev/null 2>&1 && break
  done
  local line val
  line=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "per-call" 2>/dev/null | grep -o "per-call.*" | tail -1)
  val=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "VALIDATION" 2>/dev/null | grep -o "VALIDATION.*" | sort | uniq -c | tr '\n' ';')
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%H:%M)" "$name" "${line:-NO-RESULT}" "${val:-NO-VALIDATION}" >> "${LOOP_DIR}/ladder_results.tsv"
}

run_cell "dkp-lad2-cta16a"  "${PFX}/pjrt-mainpatch-cta0bar16-20260824/${W2}" "$DKFLAGS"
run_cell "dkp-lad2-oneshot" "${PFX}/pjrt-mainpatch-kmax128-20260823/${W1}"   "$OSFLAGS"
run_cell "dkp-lad2-cta16b"  "${PFX}/pjrt-mainpatch-cta0bar16-20260824/${W2}" "$DKFLAGS"
run_cell "dkp-lad2-ilp4"    "${PFX}/pjrt-mainpatch-g8x128ilp4-20260824/${W1}" "$DKFLAGS"
echo LADDER2_DONE
