#!/usr/bin/env bash
# Interleaved A/B at the shared-repro config (U=30): stock dk geometry vs full patch d6d3132a90.
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
PFX=s3://marin-us-east-02a/marin/research/mcwitt-ra2a
W1=jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl
STOCK="${PFX}/pjrt-mainpatch-kmax128-20260823/${W1}"
PATCHED="${PFX}/pjrt-mainpatch-dkbal9-20260825/${W1}"
DKFLAGS="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"

run_cell() {
  local name="$1" wheel="$2"
  NAME="$name" WHEEL="$wheel" FLAGS="$DKFLAGS" BENCH_UPDATES_PER_PEER=30 bash "${LOOP_DIR}/bench_submit.sh" >/dev/null 2>&1
  cd "$REPO"
  for _ in 1 2 3; do
    timeout 900 uv run iris --config lib/iris/config/marin.yaml job wait "/mwittmann/${name}" >/dev/null 2>&1 && break
  done
  local line val
  line=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "per-call" 2>/dev/null | grep -o "per-call.*" | tail -1)
  val=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "VALIDATION" 2>/dev/null | grep -o "VALIDATION.*" | sort | uniq -c | tr '\n' ';')
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%H:%M)" "$name" "${line:-NO-RESULT}" "${val:-NO-VALIDATION}" >> "${LOOP_DIR}/ladder_results.tsv"
}

run_cell "dkp-ab1-stock"   "$STOCK"
run_cell "dkp-ab1-patched" "$PATCHED"
run_cell "dkp-ab2-stock"   "$STOCK"
run_cell "dkp-ab2-patched" "$PATCHED"
echo AB_DONE
