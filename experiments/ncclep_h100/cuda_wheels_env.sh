#!/bin/bash
# Synthesize a CUDA 13 toolchain from PyPI wheels.
#
# Source this file from run_gate.sh. NCCL_EP JIT-compiles device kernels during
# ep_bootstrap, so the runtime processes need nvcc and headers in addition to
# shared libraries.

if [[ -z "${WORK:-}" ]]; then
  echo "FATAL: WORK must be set before sourcing cuda_wheels_env.sh" >&2
  return 1
fi

NCCL_RUNTIME_VERSION=${NCCL_RUNTIME_VERSION:-2.30.7}

uv pip install --upgrade \
  nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-crt \
  nvidia-cuda-nvrtc nvidia-curand nvidia-cublas \
  nvidia-cuda-profiler-api nvidia-cuda-cupti nvidia-nvml-dev nvidia-nvtx \
  nvidia-cudnn-cu13 "nvidia-nccl-cu13==${NCCL_RUNTIME_VERSION}"
uv pip install nvidia-cuda-cccl || echo "cccl wheel unavailable; continuing"

SITE_PACKAGES=$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')
CUDA_ROOT="$WORK/cuda"
rm -rf "$CUDA_ROOT"
mkdir -p "$CUDA_ROOT/bin" "$CUDA_ROOT/include" "$CUDA_ROOT/lib64/stubs"

NVCC_BIN=$(find "$SITE_PACKAGES" -name nvcc -type f | head -1)
if [[ -z "$NVCC_BIN" ]]; then
  echo "FATAL: nvcc not found under $SITE_PACKAGES" >&2
  return 1
fi
NVCC_ROOT=$(dirname "$(dirname "$NVCC_BIN")")
ln -sf "$NVCC_ROOT"/bin/* "$CUDA_ROOT/bin/"
if [[ -d "$NVCC_ROOT/nvvm" ]]; then
  ln -sfn "$NVCC_ROOT/nvvm" "$CUDA_ROOT/nvvm"
fi

# Merge only CUDA 13, cuDNN, and NCCL. The project environment can also carry
# CUDA 12 wheels through Torch; mixing them produces invalid DT_NEEDED entries.
for root in "$SITE_PACKAGES/cu13" "$SITE_PACKAGES/cudnn" "$SITE_PACKAGES/nccl"; do
  if [[ -d "$root/include" ]]; then
    cp -rsn "$root/include"/. "$CUDA_ROOT/include/" 2>/dev/null || true
  fi
  for lib in "$root/lib" "$root/lib64"; do
    if [[ -d "$lib" ]]; then
      ln -sf "$lib"/*.so* "$CUDA_ROOT/lib64/" 2>/dev/null || true
      ln -sf "$lib"/*.a "$CUDA_ROOT/lib64/" 2>/dev/null || true
    fi
  done
done
if [[ ! -e "$CUDA_ROOT/include/cudnn.h" ]]; then
  echo "FATAL: cudnn.h missing from CUDA 13 merge" >&2
  return 1
fi
rm -f "$CUDA_ROOT"/lib64/*.so.12* 2>/dev/null || true

# nvtx3 must come from one coherent wheel tree.
NVTX3_DIR=$(find "$SITE_PACKAGES/cu13" -type d -name nvtx3 2>/dev/null | head -1)
if [[ -n "$NVTX3_DIR" ]]; then
  rm -rf "$CUDA_ROOT/include/nvtx3"
  ln -sfn "$NVTX3_DIR" "$CUDA_ROOT/include/nvtx3"
fi

# PyPI wheels generally ship only versioned shared objects. Add linker names.
python - "$CUDA_ROOT/lib64" <<'PY'
import os
import sys

library_dir = sys.argv[1]
for filename in sorted(os.listdir(library_dir)):
    if ".so." not in filename:
        continue
    linker_name = filename.split(".so.")[0] + ".so"
    target = os.path.join(library_dir, linker_name)
    if not os.path.exists(target):
        os.symlink(filename, target)
PY

LIBCUDA=$(ldconfig -p | awk '/libcuda\.so\.1/ {print $NF; exit}')
if [[ -z "$LIBCUDA" ]]; then
  LIBCUDA=$(find /usr/lib /usr/local -name libcuda.so.1 2>/dev/null | head -1)
fi
if [[ -z "$LIBCUDA" ]]; then
  echo "FATAL: libcuda.so.1 not found" >&2
  return 1
fi
ln -sf "$LIBCUDA" "$CUDA_ROOT/lib64/stubs/libcuda.so"
ln -sf "$LIBCUDA" "$CUDA_ROOT/lib64/libcuda.so"

export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export NVCC="$CUDA_ROOT/bin/nvcc"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_ROOT/lib64:${LIBRARY_PATH:-}"

CUDNN_HEADER=$(find "$SITE_PACKAGES" -name cudnn.h | head -1)
NCCL_HEADER=$(find "$SITE_PACKAGES" -name nccl.h | head -1)
if [[ -z "$CUDNN_HEADER" || -z "$NCCL_HEADER" ]]; then
  echo "FATAL: cuDNN or NCCL headers missing from installed wheels" >&2
  return 1
fi
export CUDNN_PATH
CUDNN_PATH=$(dirname "$(dirname "$CUDNN_HEADER")")
export NCCL_HOME
NCCL_HOME=$(dirname "$(dirname "$NCCL_HEADER")")

NCCL_SO2=$(find "$NCCL_HOME" -name libnccl.so.2 | head -1)
if [[ -z "$NCCL_SO2" ]]; then
  echo "FATAL: libnccl.so.2 missing under $NCCL_HOME" >&2
  return 1
fi
if [[ ! -e "$(dirname "$NCCL_SO2")/libnccl.so" ]]; then
  ln -s libnccl.so.2 "$(dirname "$NCCL_SO2")/libnccl.so"
fi

export CMAKE_LIBRARY_PATH="$NCCL_HOME/lib:$CUDNN_PATH/lib:$CUDA_ROOT/lib64"
export CMAKE_INCLUDE_PATH="$NCCL_HOME/include:$CUDNN_PATH/include:$CUDA_ROOT/include"

echo "cuda_wheels_env: CUDA_HOME=$CUDA_HOME CUDNN_PATH=$CUDNN_PATH NCCL_HOME=$NCCL_HOME"
