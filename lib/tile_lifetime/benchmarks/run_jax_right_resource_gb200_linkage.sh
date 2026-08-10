#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly MSA_REVISION="80434d7f67877c6570ca19cac444b84bc9855dac"
readonly SHUTTLE_ROOT="$(git rev-parse --show-toplevel)"
readonly PYTHON="${SHUTTLE_ROOT}/.venv/bin/python"
readonly ARTIFACT_DIRECTORY="${1:-/tmp/shuttle-event-right-resource-gb200}"
readonly MSA_ROOT="${ARTIFACT_DIRECTORY}/msa"
readonly BUILD_DIRECTORY="${ARTIFACT_DIRECTORY}/build"
readonly PREFLIGHT_OUTPUT="${ARTIFACT_DIRECTORY}/preflight.json"
readonly RESULT_OUTPUT="${ARTIFACT_DIRECTORY}/result.json"

mkdir -p "${ARTIFACT_DIRECTORY}"

print_results() {
  local path
  for path in "${PREFLIGHT_OUTPUT}" "${RESULT_OUTPUT}"; do
    if [[ -f "${path}" ]]; then
      echo "===== ${path} ====="
      cat "${path}"
    fi
  done
}
trap print_results EXIT

"${PYTHON}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is not None or "torch" in sys.modules:
    raise RuntimeError("the linkage environment must not contain Torch")
PY

uv pip install --python "${PYTHON}" \
  "cuda-python==13.3.1" \
  "nvidia-cuda-cccl==13.3.3.4.1" \
  "nvidia-cuda-nvcc==13.3.73" \
  "nvidia-cutlass-dsl==4.5.3" \
  "quack-kernels==0.2.10"

git clone --filter=blob:none --no-checkout https://github.com/MiniMax-AI/MSA.git "${MSA_ROOT}"
git -C "${MSA_ROOT}" checkout --detach "${MSA_REVISION}"

readonly NVCC="$(${PYTHON} - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

print(Path(distribution("nvidia-cuda-nvcc").locate_file("nvidia/cu13/bin/nvcc")).resolve())
PY
)"

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export PYTHONPATH="${SHUTTLE_ROOT}/lib/tile_lifetime/src:${SHUTTLE_ROOT}/lib/tile_lifetime/backends/sm100"

"${PYTHON}" lib/tile_lifetime/backends/sm100/preflight_jax_right_resource_runtime.py \
  --msa-root "${MSA_ROOT}" \
  --nvcc "${NVCC}" \
  --build-directory "${BUILD_DIRECTORY}/preflight" \
  --architecture sm_100a \
  --output "${PREFLIGHT_OUTPUT}"

"${PYTHON}" lib/tile_lifetime/backends/sm100/smoke_jax_right_resource_runtime.py \
  --msa-root "${MSA_ROOT}" \
  --nvcc "${NVCC}" \
  --build-directory "${BUILD_DIRECTORY}/device" \
  --query-length 128 \
  --key-length 1024 \
  --query-heads 16 \
  --key-value-heads 2 \
  --selected-count 4 \
  --seed 17 \
  --warmups 2 \
  --repeats 10 \
  --architecture sm_100a \
  --output "${RESULT_OUTPUT}"
