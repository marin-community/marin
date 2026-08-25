#!/usr/bin/env bash
# Wait for the dkbal9 (review-cleaned) wheel, then screen U=3 and U=30 against dkbal8.
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${LOOP_DIR}/../.." && pwd)"
PFX=s3://marin-us-east-02a/marin/research/mcwitt-ra2a
DKFLAGS="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"

cd "$REPO"
for _ in 1 2 3; do
  timeout 900 uv run iris --config lib/iris/config/marin.yaml job wait /mwittmann/ra2a-mw-build-dkbal9-20260825 >/dev/null 2>&1 && break
done
WHEEL_URL=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs /mwittmann/ra2a-mw-build-dkbal9-20260825 --substring "uploaded" 2>/dev/null | grep -o "s3://.*whl" | tail -1)
if [ -z "$WHEEL_URL" ]; then echo "REVALIDATE: no wheel"; exit 1; fi
echo "wheel: $WHEEL_URL"

run_cell() {
  local name="$1" upp="$2"
  NAME="$name" WHEEL="$WHEEL_URL" FLAGS="$DKFLAGS" BENCH_UPDATES_PER_PEER="$upp" bash "${LOOP_DIR}/bench_submit.sh" >/dev/null 2>&1
  cd "$REPO"
  for _ in 1 2 3; do
    timeout 900 uv run iris --config lib/iris/config/marin.yaml job wait "/mwittmann/${name}" >/dev/null 2>&1 && break
  done
  local line val
  line=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "per-call" 2>/dev/null | grep -o "per-call.*" | tail -1)
  val=$(timeout 300 uv run iris --config lib/iris/config/marin.yaml job logs "/mwittmann/${name}" --substring "VALIDATION" 2>/dev/null | grep -o "VALIDATION.*" | sort | uniq -c | tr '\n' ';')
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%H:%M)" "$name" "${line:-NO-RESULT}" "${val:-NO-VALIDATION}" >> "${LOOP_DIR}/ladder_results.tsv"
}

run_cell "dkp-bal9-u3"  3
run_cell "dkp-bal9-u30" 30
echo REVALIDATE_DONE
