#!/usr/bin/env bash
set -euo pipefail
set -x

probe_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_root="${IRIS_APP_ROOT:-/app}"
python="${app_root}/.venv/bin/python"
venv="${app_root}/.venv"
export PYTHONPATH="${probe_root}"

uv pip install --python "${python}" \
  ninja \
  pybind11==2.13.6
if ! "${python}" -c 'import torch; assert torch.version.cuda is not None'; then
  "${python}" -c 'from restore_torch_bundle import restore_torch_runtime; restore_torch_runtime()'
fi
"${python}" -c 'from restore_torch_bundle import restore_cuda13_toolkit; restore_cuda13_toolkit()'
pybind_header="${venv}/lib/python3.12/site-packages/pybind11/include/pybind11/pybind11.h"
if grep -Fq 'def(init([](Scalar i) { return static_cast<Type>(i); }), arg("value"));' "${pybind_header}"; then
  patch "${pybind_header}" "${probe_root}/pybind11-nvcc13.patch"
fi

cuda_home="${venv}/lib/python3.12/site-packages/nvidia/cu13"
if [[ ! -e "${cuda_home}/include/nv/target" ]]; then
  cp -a "${cuda_home}/include/cccl/nv" "${cuda_home}/include/nv"
fi
ln -sf libcudart.so.13 "${cuda_home}/lib/libcudart.so"
ln -sf libnvtx3interop.so.1 "${cuda_home}/lib/libnvtx3interop.so"
export CUDA_HOME="${cuda_home}"
export PATH="${venv}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

source_root=/tmp/DeepEP
shopt -s nullglob
hybridep_extensions=("${source_root}"/hybrid_ep_cpp*.so)
if (( ${#hybridep_extensions[@]} == 0 )); then
  "${python}" -c 'from pathlib import Path; from restore_torch_bundle import restore_hybridep_bundle; restore_hybridep_bundle(Path("/tmp"))'
fi
hybridep_source_changed=0
allgather_source="${source_root}/csrc/hybrid_ep/extension/allgather.cu"
if ! grep -Fq "const int64_t gathered_elets" "${allgather_source}"; then
  patch -d "${source_root}" -p1 < "${probe_root}/wide-allgather-buffer-size.patch"
  hybridep_source_changed=1
fi
backend_source="${source_root}/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh"
if ! grep -Fq "static_cast<int64_t>(current_token_id)" "${backend_source}"; then
  patch -l -d "${source_root}" -p1 < "${probe_root}/wide-routing-map-offsets.patch"
  hybridep_source_changed=1
fi
hybridep_header="${source_root}/csrc/hybrid_ep/hybrid_ep.cuh"
if ! grep -Fq "preallocated_output_token" "${hybridep_header}"; then
  patch -l -d "${source_root}" -p1 < "${probe_root}/preallocated-jax-output.patch"
  hybridep_source_changed=1
fi
hybridep_extensions=("${source_root}"/hybrid_ep_cpp*.so)
if (( ${#hybridep_extensions[@]} == 0 )) ||
  [[ "${HYBRID_EP_FORCE_REBUILD:-0}" == "1" ]] ||
  (( hybridep_source_changed == 1 )); then
  if [[ ! -d "${source_root}/.git" ]]; then
    if [[ ! -f "${source_root}/setup.py" ]]; then
      echo "HybridEP bundle did not contain buildable sources under ${source_root}" >&2
      exit 1
    fi
  else
    git -C "${source_root}" fetch --depth 1 origin 94a9f8f6b146c07d97ec58f67cd6d303296d6098
    git -C "${source_root}" checkout 94a9f8f6b146c07d97ec58f67cd6d303296d6098
  fi
  if ! (
    cd "${source_root}"
    export HYBRID_EP_JAX_SOURCE="${probe_root}/hybridep_jax_ffi.cu"
    TORCH_CUDA_ARCH_LIST=10.0 USE_MNNVL=1 MAX_JOBS=16 \
      "${python}" "${probe_root}/build_hybrid_jax.py" build_ext --inplace --force
  ) > /tmp/hybridep-jax-build.log 2>&1; then
    grep -E -B 8 -A 20 'FAILED:|error:|fatal error:' /tmp/hybridep-jax-build.log >&2 || true
    tail -n 120 /tmp/hybridep-jax-build.log >&2
    exit 1
  fi
fi
export HYBRID_EP_SOURCE="${source_root}"
export USE_MNNVL=1
gpus_per_task="${HYBRID_EP_GPUS_PER_TASK:-4}"
"${python}" -m iris.hooks.multigpu_main \
  --nproc "${gpus_per_task}" \
  --devices-per-proc 1 \
  -- \
  "${python}" "${probe_root}/jax_ffi_smoke.py"
if [[ -n "${HYBRID_EP_BUNDLE_UPLOAD_URI:-}" ]]; then
  "${python}" "${probe_root}/stage_hybridep_bundle.py"
fi
