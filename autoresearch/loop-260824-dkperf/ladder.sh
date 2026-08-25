#!/usr/bin/env bash
# Interleaved grid-ladder series: baseline g8x128 vs cta0bar at 8/16/32x SM, two rounds.
# Between-job variance (64-of-72 co-tenancy) swamps single cells; paired medians over an
# interleaved series are the unit of evidence.
# usage: ladder.sh <round-tag>
set -uo pipefail
TAG="${1:?round tag, e.g. r1}"
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
PFX=s3://marin-us-east-02a/marin/research/mcwitt-ra2a
WHL=jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl
DKFLAGS="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"

run_cell() {
  local name="$1" wheel="$2"
  NAME="$name" WHEEL="$wheel" FLAGS="$DKFLAGS" bash "${LOOP_DIR}/bench_submit.sh" >/dev/null 2>&1
  cd "$REPO"
  for _ in 1 2 3; do
    timeout 900 uv run iris --config lib/iris/config/marin.yaml job wait "/mwittmann/${name}" >/dev/null 2>&1 && break
  done
  local line val
  line=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "per-call" 2>/dev/null | grep -o "per-call.*" | tail -1)
  val=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "VALIDATION" 2>/dev/null | grep -o "VALIDATION.*" | sort | uniq -c | tr '\n' ';')
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%H:%M)" "$name" "${line:-NO-RESULT}" "${val:-NO-VALIDATION}" >> "${LOOP_DIR}/ladder_results.tsv"
}

for round in 1 2; do
  run_cell "dkp-lad${TAG}${round}-g8x"    "${PFX}/pjrt-mainpatch-g8x128-20260823/${WHL}"
  run_cell "dkp-lad${TAG}${round}-cta8"   "${PFX}/pjrt-mainpatch-cta0bar8-20260824/${WHL}"
  run_cell "dkp-lad${TAG}${round}-cta16"  "${PFX}/pjrt-mainpatch-cta0bar16-20260824/${WHL}"
  run_cell "dkp-lad${TAG}${round}-cta32"  "${PFX}/pjrt-mainpatch-cta0bar32-20260824/${WHL}"
done
echo LADDER_DONE
