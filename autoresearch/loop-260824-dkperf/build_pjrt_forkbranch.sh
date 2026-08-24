#!/bin/bash
# Build jax-cuda13-pjrt from a commit on the mcwitt/xla fork (leg-4 kernel-patch wheels).
# Optionally override the dk grid SM multiplier (the fork branches carry kGridSmMultiplier = 8).
#
# usage: XLA_REV=<sha> JAX_REV=... STASH=s3://... [GRID_MULT=16] build_pjrt_forkbranch.sh
set -euo pipefail
XLA_REV="${XLA_REV:?set XLA_REV (sha on mcwitt/xla)}"
JAX_REV="${JAX_REV:?set JAX_REV}"
STASH="${STASH:?set STASH (s3://... prefix to upload the wheel)}"

mkdir -p /tmp/build
cd /tmp/build
if [ ! -d jax ]; then git clone --filter=blob:none https://github.com/jax-ml/jax.git jax; fi
cd /tmp/build/jax && git fetch origin "$JAX_REV" && git checkout "$JAX_REV"

if [ ! -d /tmp/build/xla ]; then git clone --filter=blob:none https://github.com/openxla/xla.git /tmp/build/xla; fi
cd /tmp/build/xla
git remote add fork https://github.com/mcwitt/xla.git 2>/dev/null || true
git fetch fork "$XLA_REV"
git checkout "$XLA_REV" && git reset --hard "$XLA_REV"
echo "xla: $(git rev-parse HEAD)"

if [ -n "${GRID_MULT:-}" ]; then
  sed -i "s/static constexpr int32_t kGridSmMultiplier = 8;/static constexpr int32_t kGridSmMultiplier = ${GRID_MULT};/" \
    xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
  grep -q "kGridSmMultiplier = ${GRID_MULT};" xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
  echo "grid multiplier set to ${GRID_MULT}"
fi

cd /tmp/build/jax
python3 build/build.py build \
  --wheels=jax-cuda-pjrt \
  --cuda_major_version=13 \
  --python_version=3.12 \
  --local_xla_path=/tmp/build/xla \
  --bazel_options=--jobs=48 \
  --bazel_options=--define=ynn_enable_arm64_neonfp8=false \
  --verbose
ls -la dist/

python3 - <<EOF
import glob
import fsspec
wheel = glob.glob("dist/jax_cuda13_pjrt-*.whl")[0]
dest = "${STASH}".rstrip("/") + "/" + wheel.split("/")[-1]
fs, _, _ = fsspec.get_fs_token_paths(dest)
fs.put(wheel, dest)
print("uploaded", dest)
EOF
echo BUILD_OK
