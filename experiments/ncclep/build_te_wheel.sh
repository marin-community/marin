#!/bin/bash
# Build TransformerEngine (pinned main SHA) with NCCL_EP for JAX on an arm64
# GB200 iris pod (issue #7331, NCCLEP-002).
#
# The iris task image (python:3.12-slim) has gcc/make/git but no CUDA toolkit;
# the entire toolchain is assembled from pip cu13 wheels into a synthetic
# CUDA_HOME symlink tree (cuda_wheels_env.sh). libnccl_ep.a builds from TE's
# in-tree 3rdparty/nccl submodule (public NVIDIA/nccl, v2.30u1 line) and links
# statically into libtransformer_engine.so. Runtime NCCL >= 2.30.4 (EP gate).
#
# Stashes to object storage: the wheel AND a tarball of the generated NCCL_EP
# JIT headers — NCCL EP JIT-compiles device kernels at bootstrap and the wheel
# does not package the headers (baked build paths don't exist on other pods).
#
#   iris --cluster=marin job run --user mwittmann --target-cluster cw-us-east-08a \
#     --gpu GB200x4 --enable-extra-resources --cpu 64 --memory 200g \
#     --extra gpu --job-name ncclep-te-build -- bash experiments/ncclep/build_te_wheel.sh
set -euxo pipefail

TE_SHA=${TE_SHA:-68493d2d55ac37e540301467b278bdb1c2019e81}  # TE main 2026-07-17
STASH=${STASH:-s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep}
WORK=${WORK:-/tmp/ncclep-build}
export MAX_JOBS=${MAX_JOBS:-64}
REPO_ROOT=$(pwd)

mkdir -p "$WORK"
cd "$WORK"

echo "=== PHASE 1: build tools + toolchain env ==="
uv pip install pip cmake ninja "pybind11[global]" wheel setuptools packaging \
  "nvidia-cudnn-frontend>=1.25.0" flax
source "$REPO_ROOT/experiments/ncclep/cuda_wheels_env.sh"
"$CUDA_HOME/bin/nvcc" --version

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

echo "=== PHASE 4b: bundle NCCL_EP JIT headers ==="
JIT_INC="$WORK/te/3rdparty/nccl/build/include"
[ -d "$JIT_INC/nccl_ep" ] || { echo "FATAL: JIT headers missing at $JIT_INC"; exit 1; }
tar -C "$JIT_INC" -czf "$WORK/nccl-ep-jit-headers.tgz" .
ls -la "$WORK/nccl-ep-jit-headers.tgz"

echo "=== PHASE 5: install + import probe ==="
uv pip install wheelhouse/transformer_engine*.whl
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
print("IMPORT PROBE PASSED")
EOF

echo "=== PHASE 6: stash wheel + JIT headers to object storage ==="
uv pip install s3fs
python - "$STASH" <<'EOF'
import glob, os, sys
import s3fs
dest = sys.argv[1].rstrip("/")
fs = s3fs.S3FileSystem(endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
for src, sub in [(w, "wheels") for w in glob.glob("wheelhouse/*.whl")] + [
    ("nccl-ep-jit-headers.tgz", "jit"),
]:
    tgt = f"{dest}/{sub}/{os.path.basename(src)}"
    fs.put(src, tgt)
    print("uploaded", tgt, fs.info(tgt)["size"])
EOF
echo "=== BUILD JOB DONE ==="
