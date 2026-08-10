#!/usr/bin/env bash

set -uo pipefail

readonly output_root=/tmp/shuttle-rms-warp-finalize-h100
readonly benchmark=/app/lib/tile_lifetime/benchmarks/jax_generated_row_normalization_backward.py
readonly nvcc=/app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
readonly revision="$1"

mkdir -p "${output_root}/measurement" "${output_root}/xla"

{
  git -C /app rev-parse HEAD
  nvidia-smi -L
  nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits --id=0
  "${nvcc}" --version
  /app/.venv/bin/python - <<'PY'
import jax
import jaxlib

print(f"python={__import__('sys').version}")
print(f"jax={jax.__version__}")
print(f"jaxlib={jaxlib.__version__}")
print(f"device={jax.devices()[0]}")
PY
} >"${output_root}/environment.txt" 2>&1

command=(
  /app/.venv/bin/python
  "${benchmark}"
  --rows 2048
  --hidden 4096
  --threads 256
  --column-groups-per-block 32
  --column-outputs-per-group 1
  --column-reduction-strategy warp_finalize
  --pipeline-schedule coalesce_compatible_row_stages
  --compare-pipeline-schedules
  --warmups 10
  --repeats 30
  --iterations 100
  --seed 20260809
  --nvcc "${nvcc}"
  --architecture sm_90a
  --artifact-directory "${output_root}/measurement"
  --xla-dump-directory "${output_root}/xla"
  --json-output "${output_root}/summary.json"
  --shuttle-revision "${revision}"
)

printf 'PYTHONPATH=/app/lib/tile_lifetime/src:/app' >"${output_root}/benchmark-command.txt"
printf ' %q' "${command[@]}" >>"${output_root}/benchmark-command.txt"
printf '\n' >>"${output_root}/benchmark-command.txt"

set +e
PYTHONPATH=/app/lib/tile_lifetime/src:/app "${command[@]}" \
  >"${output_root}/benchmark.stdout" 2>"${output_root}/benchmark.stderr"
status=$?
set -e
printf '%s\n' "${status}" >"${output_root}/benchmark-exit-code.txt"

find "${output_root}/measurement" -type f -name '*.so' -delete
tar -C /tmp -czf /tmp/shuttle-rms-warp-finalize-h100.tar.gz shuttle-rms-warp-finalize-h100
python - <<'PY'
import base64
from pathlib import Path

payload = base64.b64encode(Path("/tmp/shuttle-rms-warp-finalize-h100.tar.gz").read_bytes()).decode()
print("SHUTTLE_ARTIFACT_BASE64_BEGIN")
print(payload)
print("SHUTTLE_ARTIFACT_BASE64_END")
PY

exit "${status}"
