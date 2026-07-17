#!/bin/bash
# Build TransformerEngine (pinned main SHA) with NCCL_EP for JAX on an arm64
# GB200 iris pod (issue #7331, NCCLEP-002).
#
# The iris task image (python:3.12-slim) has gcc/make/git but no CUDA toolkit;
# the entire toolchain is assembled from pip cu13 wheels into a synthetic
# CUDA_HOME symlink tree. libnccl_ep.a builds from TE's in-tree 3rdparty/nccl
# submodule (public NVIDIA/nccl, v2.30u1 line) and links statically into
# libtransformer_engine.so. Runtime NCCL is upgraded to >= 2.30.4 (EP gate).
#
# Run as an iris job on a GB200 node (arch pinned to sm_100a; no GPU needed for
# the compile itself, but the import probe and the libcuda link want one):
#   iris --cluster=marin job run --user mwittmann --target-cluster cw-us-east-08a \
#     --gpu GB200x4 --enable-extra-resources --cpu 64 --memory 200g \
#     --extra gpu --job-name ncclep-te-build -- bash experiments/ncclep/build_te_wheel.sh
set -euxo pipefail

TE_SHA=${TE_SHA:-68493d2d55ac37e540301467b278bdb1c2019e81}  # TE main 2026-07-17
NCCL_RUNTIME_VERSION=${NCCL_RUNTIME_VERSION:-2.30.7}
WHEEL_DEST=${WHEEL_DEST:-s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep/wheels/}
WORK=${WORK:-/tmp/ncclep-build}
export MAX_JOBS=${MAX_JOBS:-64}

mkdir -p "$WORK"
cd "$WORK"

echo "=== PHASE 1: python-level build tools + cu13 toolchain wheels ==="
uv pip install pip cmake ninja "pybind11[global]" wheel setuptools packaging \
  "nvidia-cudnn-frontend>=1.25.0" flax
# CUDA 13 wheels dropped the -cu13 suffix (the suffixed names are deprecated
# 0.0.1 stubs) — except cudnn and nccl, which keep it.
uv pip install --upgrade \
  nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-crt \
  nvidia-cuda-nvrtc nvidia-curand nvidia-cublas \
  nvidia-cuda-profiler-api nvidia-cuda-cupti nvidia-nvml-dev nvidia-nvtx \
  nvidia-cudnn-cu13 "nvidia-nccl-cu13==${NCCL_RUNTIME_VERSION}"
uv pip install nvidia-cuda-cccl || echo "cccl wheel unavailable; continuing"

SP=$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')
echo "nvidia wheel root: $SP"

echo "=== PHASE 2: synthesize CUDA_HOME from wheels ==="
CUDA="$WORK/cuda"
rm -rf "$CUDA"
mkdir -p "$CUDA/bin" "$CUDA/include" "$CUDA/lib64/stubs"

NVCC=$(find "$SP" -name nvcc -type f | head -1)
[ -n "$NVCC" ] || { echo "FATAL: nvcc not found in $SP"; exit 1; }
NVCC_ROOT=$(dirname "$(dirname "$NVCC")")
ln -sf "$NVCC_ROOT"/bin/* "$CUDA/bin/"
[ -d "$NVCC_ROOT/nvvm" ] && ln -sfn "$NVCC_ROOT/nvvm" "$CUDA/nvvm"

# Merge every wheel's include/ and lib/ (first writer wins on collisions);
# find-based so nested wheel layouts (e.g. nvidia/cu13/<pkg>) still resolve.
for inc in $(find "$SP" -maxdepth 3 -type d -name include); do
  cp -rsn "$inc"/. "$CUDA/include/" 2>/dev/null || true
done
for lib in $(find "$SP" -maxdepth 3 -type d -name lib -o -maxdepth 3 -type d -name lib64); do
  ln -sf "$lib"/*.so* "$CUDA/lib64/" 2>/dev/null || true
  ln -sf "$lib"/*.a "$CUDA/lib64/" 2>/dev/null || true
done
# Unversioned .so symlinks for the linker (wheels often ship only .so.N).
python - "$CUDA/lib64" <<'EOF'
import os, sys
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    if ".so." in f:
        base = f.split(".so.")[0] + ".so"
        tgt = os.path.join(d, base)
        if not os.path.exists(tgt):
            os.symlink(f, tgt)
EOF
# nvtx3 must come from ONE wheel — the per-file merge can interleave two nvtx
# versions (e.g. cupti's copy + nvidia-nvtx's), which breaks the macro layering
# (NVTX_NULLPTR undefined). Prefer the nvidia-nvtx wheel's tree wholesale.
NVTX3_DIR=$(find "$SP" -type d -name nvtx3 -path "*nvtx*" | head -1)
[ -z "$NVTX3_DIR" ] && NVTX3_DIR=$(find "$SP" -type d -name nvtx3 | head -1)
if [ -n "$NVTX3_DIR" ]; then
  rm -rf "$CUDA/include/nvtx3"
  ln -sfn "$NVTX3_DIR" "$CUDA/include/nvtx3"
fi

# Driver stub: link the node's real libcuda.
LIBCUDA=$(ldconfig -p | awk '/libcuda\.so\.1/ {print $NF; exit}')
[ -n "$LIBCUDA" ] || LIBCUDA=$(find /usr/lib /usr/local -name 'libcuda.so.1' 2>/dev/null | head -1)
[ -n "$LIBCUDA" ] || { echo "FATAL: libcuda.so.1 not found"; exit 1; }
ln -sf "$LIBCUDA" "$CUDA/lib64/stubs/libcuda.so"
ln -sf "$LIBCUDA" "$CUDA/lib64/libcuda.so"

export CUDA_HOME="$CUDA" CUDA_PATH="$CUDA" CUDACXX="$CUDA/bin/nvcc"
export PATH="$CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA/lib64:${LD_LIBRARY_PATH:-}"
# nvcc resolves its implicit -L from its own (wheel) location, which has no
# lib64 — LIBRARY_PATH covers the host-linker search for cudadevrt/cudart_static.
export LIBRARY_PATH="$CUDA/lib64:${LIBRARY_PATH:-}"
"$CUDA/bin/nvcc" --version
ls -la "$CUDA/lib64/" | grep -E "\.a$|libnccl|libcudart" || true

CUDNN_H=$(find "$SP" -name cudnn.h | head -1)
export CUDNN_PATH=$(dirname "$(dirname "$CUDNN_H")")
NCCL_H=$(find "$SP" -name nccl.h | head -1)
export NCCL_HOME=$(dirname "$(dirname "$NCCL_H")")
# The wheel ships only libnccl.so.2; the EP shared-lib link needs -lnccl.
NCCL_SO2=$(find "$NCCL_HOME" -name 'libnccl.so.2' | head -1)
[ -e "$(dirname "$NCCL_SO2")/libnccl.so" ] || ln -s libnccl.so.2 "$(dirname "$NCCL_SO2")/libnccl.so"
echo "CUDNN_PATH=$CUDNN_PATH NCCL_HOME=$NCCL_HOME"
# cmake find_library/find_path consult these env vars; wheel dirs aren't in
# any default search prefix.
export CMAKE_LIBRARY_PATH="$NCCL_HOME/lib:$CUDNN_PATH/lib:$CUDA/lib64"
export CMAKE_INCLUDE_PATH="$NCCL_HOME/include:$CUDNN_PATH/include:$CUDA/include"

echo "=== PHASE 3: clone TE @ $TE_SHA with submodules ==="
if [ ! -d te/.git ]; then
  git init te
  git -C te remote add origin https://github.com/NVIDIA/TransformerEngine
fi
git -C te fetch --depth 1 origin "$TE_SHA"
git -C te checkout --force FETCH_HEAD
git -C te submodule update --init --recursive --depth 1 \
  || git -C te submodule update --init --recursive
git -C te ls-tree HEAD 3rdparty/

echo "=== PHASE 4: build wheel (jax framework, sm_100a) ==="
export NVTE_FRAMEWORK=jax
export NVTE_CUDA_ARCHS=100a
cd te
# No -v: verbose nvcc command echoes drown the actual errors in iris log
# retention. Tee to a file and surface only error context on failure.
rc=0
python -m pip wheel --no-build-isolation --no-deps -w "$WORK/wheelhouse" . \
  > "$WORK/tebuild.log" 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  echo "=== BUILD FAILED (rc=$rc); error context: ==="
  grep -nE "FAILED:|fatal error|error:|Error limit|ptxas.* error|Killed" "$WORK/tebuild.log" \
    | grep -vE "nvcc -forward|/usr/bin/c\+\+ " | head -60
  echo "=== last 60 lines: ==="
  tail -60 "$WORK/tebuild.log"
  exit "$rc"
fi
tail -5 "$WORK/tebuild.log"
cd "$WORK"
ls -la wheelhouse/
sha256sum wheelhouse/*.whl

echo "=== PHASE 5: install + import probe ==="
uv pip install wheelhouse/transformer_engine*.whl
uv pip install flax || true
python - <<'EOF'
import jax
print("jax", jax.__version__, "devices", jax.devices())
import transformer_engine.jax as tejax
print("te.jax OK:", tejax.__file__)
from transformer_engine.jax import ep
surface = [s for s in dir(ep) if not s.startswith("_")]
print("te.jax.ep surface:", surface)
assert "ep_bootstrap" in surface and "ep_dispatch" in surface and "ep_combine" in surface
import transformer_engine_jax as ext
ep_syms = [s for s in dir(ext) if "ep" in s.lower()]
print("pybind ext EP symbols:", ep_syms)
import ctypes, transformer_engine
print("IMPORT PROBE PASSED")
EOF

echo "=== PHASE 6: stash wheel to object storage ==="
uv pip install s3fs
python - "$WHEEL_DEST" <<'EOF'
import glob, os, sys
import s3fs
dest = sys.argv[1].rstrip("/")
fs = s3fs.S3FileSystem(endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
for whl in glob.glob("wheelhouse/*.whl"):
    tgt = f"{dest}/{os.path.basename(whl)}"
    fs.put(whl, tgt)
    print("uploaded", tgt, fs.info(tgt)["size"])
EOF
echo "=== BUILD JOB DONE ==="
