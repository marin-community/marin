#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

readonly NCCL_REVISION="db0c814185a0415cc2e23dca387fecb9282de551"
readonly SOURCE_ROOT="/tmp/nccl-ubx-db0c814"

source "$IRIS_VENV/bin/activate"
cd "$IRIS_WORKDIR"

uv pip install --link-mode copy --reinstall \
  nvidia-cuda-nvcc==13.2.78 \
  nvidia-nvvm==13.2.78 \
  nvidia-cuda-cccl==13.3.3.4.1 \
  nvidia-cuda-runtime==13.2.75 \
  nvidia-nccl-cu13==2.30.7

cuda_bin="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu*/bin -name nvcc -print -quit)"
test -n "$cuda_bin"
CUDA_HOME="$(dirname "$(dirname "$cuda_bin")")"
export CUDA_HOME

rm -rf "$SOURCE_ROOT"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/nccl.git "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" fetch --depth 1 origin "$NCCL_REVISION"
git -C "$SOURCE_ROOT" checkout --detach FETCH_HEAD
make -C "$SOURCE_ROOT" -j32 src.build \
  CUDA_HOME="$CUDA_HOME" \
  CUDA_LIB="$CUDA_HOME/lib" \
  CUDARTLIB=cudart \
  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

export LD_LIBRARY_PATH="$SOURCE_ROOT/build/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
wheel_nccl="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/nccl/lib -name libnccl.so.2 -print -quit)"
test -n "$wheel_nccl"
rm -f "$wheel_nccl"
ln -s "$SOURCE_ROOT/build/lib/libnccl.so.2" "$wheel_nccl"
export LD_PRELOAD="$SOURCE_ROOT/build/lib/libnccl.so.2"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"

python -c 'import jax, jaxlib; print("JAX runtime", jax.__version__, jaxlib.__version__, jax.local_devices())'

for routing in balanced learned_skew; do
  python experiments/grug/moe/benchmark_jax_ubx_moe.py \
    --source-root "$SOURCE_ROOT" \
    --cuda-home "$CUDA_HOME" \
    --routing "$routing" \
    --tokens-per-rank "${UBX_MOE_TOKENS_PER_RANK:-256}" \
    --hidden-dim "${UBX_MOE_HIDDEN_DIM:-256}" \
    --intermediate-dim "${UBX_MOE_INTERMEDIATE_DIM:-384}" \
    --warmup "${UBX_MOE_WARMUP:-2}" \
    --iterations "${UBX_MOE_ITERATIONS:-5}"
done
