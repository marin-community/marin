#!/usr/bin/env bash
set -euo pipefail
set -x

uv pip install --python .venv/bin/python \
  ninja \
  pybind11==2.13.6 \
  nvidia-cuda-runtime==13.2.75 \
  nvidia-cuda-cccl==13.2.75 \
  nvidia-cuda-profiler-api==13.2.75 \
  nvidia-nvtx==13.2.75
patch \
  .venv/lib/python3.12/site-packages/pybind11/include/pybind11/pybind11.h \
  scripts/hybridep_build_probe/pybind11-nvcc13.patch

cuda_home=".venv/lib/python3.12/site-packages/nvidia/cu13"
ln -sf libcudart.so.13 "${cuda_home}/lib/libcudart.so"
ln -sf libnvtx3interop.so.1 "${cuda_home}/lib/libnvtx3interop.so"
export CUDA_HOME="${PWD}/${cuda_home}"
export PATH="${PWD}/.venv/bin:${PATH}"
export PYTHONPATH="${PWD}/scripts/hybridep_build_probe"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

source_root=/tmp/DeepEP
git clone --depth 1 --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git "${source_root}"
git -C "${source_root}" fetch --depth 1 origin 94a9f8f6b146c07d97ec58f67cd6d303296d6098
git -C "${source_root}" checkout 94a9f8f6b146c07d97ec58f67cd6d303296d6098
(
  cd "${source_root}"
  TORCH_CUDA_ARCH_LIST=10.0 USE_MNNVL=1 MAX_JOBS=16 \
    /app/.venv/bin/python /app/scripts/hybridep_build_probe/build_hybrid.py build_ext --inplace
)
export HYBRID_EP_SOURCE="${source_root}"
export USE_MNNVL=1
gpus_per_task="${HYBRID_EP_GPUS_PER_TASK:-1}"
if ((gpus_per_task == 1)); then
  .venv/bin/python scripts/hybridep_build_probe/runtime_smoke.py
else
  .venv/bin/python -m iris.hooks.multigpu_main \
    --nproc "${gpus_per_task}" \
    --devices-per-proc 1 \
    -- \
    .venv/bin/python scripts/hybridep_build_probe/runtime_smoke.py
fi
