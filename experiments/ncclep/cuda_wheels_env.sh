# Source this to synthesize a CUDA_HOME from pip cu13 wheels on an iris GPU pod
# (issue #7331). Used by the TE build job AND the NCCL_EP runtime launchers —
# NCCL EP JIT-compiles its device kernels at bootstrap, so runtime needs nvcc +
# CUDA headers too, not just libs.
#
# Requires: $WORK set. Exports CUDA_HOME, PATH, LD_LIBRARY_PATH, LIBRARY_PATH,
# CMAKE_LIBRARY_PATH, CMAKE_INCLUDE_PATH, CUDNN_PATH, NCCL_HOME, NVCC.

NCCL_RUNTIME_VERSION=${NCCL_RUNTIME_VERSION:-2.30.7}

uv pip install --upgrade \
  nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-crt \
  nvidia-cuda-nvrtc nvidia-curand nvidia-cublas \
  nvidia-cuda-profiler-api nvidia-cuda-cupti nvidia-nvml-dev nvidia-nvtx \
  nvidia-cudnn-cu13 "nvidia-nccl-cu13==${NCCL_RUNTIME_VERSION}"
uv pip install nvidia-cuda-cccl || echo "cccl wheel unavailable; continuing"

SP=$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')
CUDA="$WORK/cuda"
rm -rf "$CUDA"
mkdir -p "$CUDA/bin" "$CUDA/include" "$CUDA/lib64/stubs"

NVCC_BIN=$(find "$SP" -name nvcc -type f | head -1)
[ -n "$NVCC_BIN" ] || { echo "FATAL: nvcc not found in $SP"; exit 1; }
NVCC_ROOT=$(dirname "$(dirname "$NVCC_BIN")")
ln -sf "$NVCC_ROOT"/bin/* "$CUDA/bin/"
[ -d "$NVCC_ROOT/nvvm" ] && ln -sfn "$NVCC_ROOT/nvvm" "$CUDA/nvvm"

# Merge ONLY the cu13 tree plus cudnn/nccl. The venv also carries cu12 wheels
# (torch+cu128 deps) under nvidia/<pkg>/ — merging those links against
# libcublas.so.12 (soname mismatch vs the cu13 runtime).
for root in "$SP/cu13" "$SP/cudnn" "$SP/nccl"; do
  [ -d "$root/include" ] && cp -rsn "$root/include"/. "$CUDA/include/" 2>/dev/null || true
  for lib in "$root"/lib "$root"/lib64; do
    if [ -d "$lib" ]; then
      ln -sf "$lib"/*.so* "$CUDA/lib64/" 2>/dev/null || true
      ln -sf "$lib"/*.a "$CUDA/lib64/" 2>/dev/null || true
    fi
  done
done
[ -e "$CUDA/include/cudnn.h" ] || { echo "FATAL: cudnn.h missing from merge"; ls "$SP"; exit 1; }
rm -f "$CUDA"/lib64/*.so.12* 2>/dev/null || true

# nvtx3 must come from ONE wheel — per-file merges can interleave versions.
NVTX3_DIR=$(find "$SP/cu13" -type d -name nvtx3 2>/dev/null | head -1)
if [ -n "$NVTX3_DIR" ]; then
  rm -rf "$CUDA/include/nvtx3"
  ln -sfn "$NVTX3_DIR" "$CUDA/include/nvtx3"
fi

# Unversioned .so symlinks for the linker (wheels ship only .so.N).
python - "$CUDA/lib64" <<'PYEOF'
import os, sys
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    if ".so." in f:
        base = f.split(".so.")[0] + ".so"
        tgt = os.path.join(d, base)
        if not os.path.exists(tgt):
            os.symlink(f, tgt)
PYEOF

# Driver stub: link the node's real libcuda.
LIBCUDA=$(ldconfig -p | awk '/libcuda\.so\.1/ {print $NF; exit}')
[ -n "$LIBCUDA" ] || LIBCUDA=$(find /usr/lib /usr/local -name 'libcuda.so.1' 2>/dev/null | head -1)
[ -n "$LIBCUDA" ] || { echo "FATAL: libcuda.so.1 not found"; exit 1; }
ln -sf "$LIBCUDA" "$CUDA/lib64/stubs/libcuda.so"
ln -sf "$LIBCUDA" "$CUDA/lib64/libcuda.so"

export CUDA_HOME="$CUDA" CUDA_PATH="$CUDA" CUDACXX="$CUDA/bin/nvcc" NVCC="$CUDA/bin/nvcc"
export PATH="$CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA/lib64:${LIBRARY_PATH:-}"
CUDNN_H=$(find "$SP" -name cudnn.h | head -1)
export CUDNN_PATH=$(dirname "$(dirname "$CUDNN_H")")
NCCL_H=$(find "$SP" -name nccl.h | head -1)
export NCCL_HOME=$(dirname "$(dirname "$NCCL_H")")
NCCL_SO2=$(find "$NCCL_HOME" -name 'libnccl.so.2' | head -1)
[ -e "$(dirname "$NCCL_SO2")/libnccl.so" ] || ln -s libnccl.so.2 "$(dirname "$NCCL_SO2")/libnccl.so"
export CMAKE_LIBRARY_PATH="$NCCL_HOME/lib:$CUDNN_PATH/lib:$CUDA/lib64"
export CMAKE_INCLUDE_PATH="$NCCL_HOME/include:$CUDNN_PATH/include:$CUDA/include"
echo "cuda_wheels_env: CUDA_HOME=$CUDA_HOME CUDNN_PATH=$CUDNN_PATH NCCL_HOME=$NCCL_HOME"
