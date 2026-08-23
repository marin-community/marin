#!/bin/bash
# Build jax-cuda13-pjrt from XLA main (cleanmain pins) with ONLY the kMaxPeers 32 -> 128 bump
# (openxla/xla#47283), no device-kernel changes.
#
# Purpose: run the one-shot transport at 64 ranks on the main-vintage compiler. This
# discriminates whether the 2x backward-transport op count seen in the dk arms follows the
# compiler vintage (would reproduce here) or the device-kernel flag (would not). It is also
# the minimal-patch one-shot-on-main cell: stock main faults at 64 ranks on both one-shot
# variants because the barrier kernels' 32-slot arrays are indexed per rank without a count
# check.
#
# usage: XLA_REV=... JAX_REV=... STASH=s3://... build_pjrt_mainpatch_kmax128.sh
set -euo pipefail
XLA_REV="${XLA_REV:?set XLA_REV}"
JAX_REV="${JAX_REV:?set JAX_REV}"
STASH="${STASH:?set STASH (s3://... prefix to upload the wheel)}"

mkdir -p /tmp/build
cd /tmp/build
if [ ! -d jax ]; then git clone --filter=blob:none https://github.com/jax-ml/jax.git jax; fi
cd /tmp/build/jax && git fetch origin "$JAX_REV" && git checkout "$JAX_REV"

if [ ! -d /tmp/build/xla ]; then git clone --filter=blob:none https://github.com/openxla/xla.git /tmp/build/xla; fi
cd /tmp/build/xla && git fetch origin "$XLA_REV" && git checkout "$XLA_REV" && git reset --hard "$XLA_REV"
echo "xla: $(git rev-parse HEAD)"

sed -i 's/kMaxPeers = 32;/kMaxPeers = 128;/g' xla/stream_executor/gpu/multi_gpu_barrier_kernel.h
test "$(grep -c 'kMaxPeers = 128' xla/stream_executor/gpu/multi_gpu_barrier_kernel.h)" = "2"
echo "kmax128 patch applied"

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
