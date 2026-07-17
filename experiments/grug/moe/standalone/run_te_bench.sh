#!/usr/bin/env bash
# MXFP8-004b: install TransformerEngine (jax) on an aarch64 GB200 pod and run
# bench_te_grouped.py. transformer_engine_cu13 has a prebuilt manylinux aarch64
# wheel up to 2.16.0; the jax glue (transformer_engine_jax) is sdist-only and
# compiles a small pybind11 extension against the pod's jax[cuda13] stack
# (--no-build-isolation so it detects CUDA 13, --no-deps so its static
# transformer_engine_cu12 metadata is ignored).
set -euxo pipefail

PY="$(command -v python)"
uv pip install --python "$PY" --no-cache-dir \
  "transformer-engine==2.16.0" "transformer-engine-cu13==2.16.0" \
  "pybind11[global]" flax ninja cmake packaging \
  nvidia-cuda-runtime nvidia-cuda-cccl nvidia-curand nvidia-cuda-nvrtc nvidia-nvtx

SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# The pod venv has no CUDA toolkit and TE's nvidia/*/include scan comes up
# empty (job mxfp8-004b-g1: fatal error cuda_runtime_api.h; g2: nv/target).
# Assemble a synthetic CUDA_HOME whose include/ symlinks every header dir the
# build needs (cudart + CCCL from the nvidia-cuda-* wheels above, cudnn,
# cublas/cublasLt), scoped to the build so the iris CUDA staging is untouched.
CUDAHOME=/tmp/te-cuda-home
mkdir -p "$CUDAHOME/include"
for h in cuda_runtime_api.h cudnn.h cublas_v2.h cublasLt.h nccl.h; do
  src="$(find "$SITE" -name "$h" | head -1)"
  echo "header $h -> ${src:-NOT FOUND}"
  if [ -n "$src" ]; then
    cp -rsn "$(dirname "$src")/." "$CUDAHOME/include/" || true
  fi
done
ls "$CUDAHOME/include" | head -20

# The pybind11 extension links -lnccl; the pip nccl wheel only ships libnccl.so.2.
NCCLLIB="$(dirname "$(find "$SITE" -name 'libnccl.so*' | head -1)")"
ln -sf "$NCCLLIB/libnccl.so.2" "$NCCLLIB/libnccl.so"

if ! command -v g++ >/dev/null; then
  apt-get update -qq && apt-get install -yqq g++
fi
g++ --version | head -1

# One source-build attempt, 25-minute ceiling.
timeout 1500 env CUDA_HOME="$CUDAHOME" LIBRARY_PATH="$NCCLLIB:${LIBRARY_PATH:-}" \
  uv pip install --python "$PY" --no-cache-dir --no-build-isolation --no-deps \
  "transformer-engine-jax==2.16.0"

# The pybind extension links -lnccl (SONAME libnccl.so.2), which the dynamic
# loader can only resolve from the pip nccl wheel via LD_LIBRARY_PATH. TE core
# also dlopens curand/nvrtc, satisfied by the nvidia-* wheels installed above.
export LD_LIBRARY_PATH="$NCCLLIB:${LD_LIBRARY_PATH:-}"

python -c "import transformer_engine.jax, transformer_engine_jax as t; \
print('TE_IMPORT_OK cublasLt', t.get_cublasLt_version(), 'cuda', t.get_cuda_version(), 'cudnn', t.get_cudnn_version())"

python experiments/grug/moe/standalone/bench_te_grouped.py --out /tmp/bench_te_grouped.json
