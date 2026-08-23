#!/bin/bash
# Build jax-cuda13-pjrt from XLA main (same revisions as the cleanmain wheel) with ONE
# change: the ragged all-to-all device kernel launches 1024 threads per CTA instead of 512.
#
# Motivation (leg 2b): the genuine device kernel measures 19.2-19.4 MFU vs the one-shot's
# 22.8 on every runtime. The one-shot copy kernel fills SMs with many 128-thread blocks;
# the device kernel's cooperative barrier structure pins it to one CTA per SM, so at 512
# threads it holds ~4x fewer in-flight NVLink stores. Doubling threads per CTA doubles the
# in-flight window without touching the cooperative barrier layout (grid stays <= SM count,
# per-CTA barrier indexing unchanged). __launch_bounds__(1024) caps registers at 64/thread,
# ample for a copy loop.
#
# Holding JAX_REV/XLA_REV to the cleanmain wheel's pins makes the resulting arm a clean A/B
# against i14 (cleanmain dk = 19.16): the only delta is this patch.
#
# usage (iris CPU job on cw-us-east-08a, arm64):
#   XLA_REV=e5d008bb03... JAX_REV=366ff3575f... STASH=s3://... build_pjrt_mainpatch_t1024.sh
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

# The patch: 512 -> 1024 threads per CTA in the device kernel, both the host launch
# constant and the kernel's launch-bounds cap. Grep-verified so a source drift fails the
# build instead of silently producing an unpatched wheel.
sed -i 's/static constexpr size_t kThreadsPerCta = 512;/static constexpr size_t kThreadsPerCta = 1024;/' \
  xla/backends/gpu/runtime/ragged_all_to_all.cc
grep -q "kThreadsPerCta = 1024" xla/backends/gpu/runtime/ragged_all_to_all.cc
sed -i 's/__global__ void __launch_bounds__(512) RaggedAllToAllDeviceKernelImpl/__global__ void __launch_bounds__(1024) RaggedAllToAllDeviceKernelImpl/' \
  xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h
grep -q "__launch_bounds__(1024) RaggedAllToAllDeviceKernelImpl" \
  xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h
echo "t1024 patch applied"

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
