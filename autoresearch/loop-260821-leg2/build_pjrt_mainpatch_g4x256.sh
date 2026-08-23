#!/bin/bash
# Build jax-cuda13-pjrt from XLA main (cleanmain pins) with the ragged a2a device kernel
# launching a 4x-SM grid of 256-thread CTAs (vs stock 1x-SM grid of 512-thread CTAs).
#
# Leg-2b iteration 2. t1024 (1024 threads, same 1x grid) was a null: per-SM in-flight bytes
# are not the limit. The surviving geometric difference vs the line-rate one-shot kernel is
# per-peer CTA parallelism (~2 concurrent CTAs per peer vs dozens). 4x CTAs at 256 threads
# holds threads/SM at 1024 (all-resident: 4 CTAs/SM, well under the 2048-thread and register
# limits) while quadrupling concurrent CTAs per peer to ~8 and cutting each CTA's sequential
# update chain 4x. The NCCL barrier registration must cover the larger grid, so
# device_kernel_barrier_count scales with it.
#
# usage: XLA_REV=... JAX_REV=... STASH=s3://... build_pjrt_mainpatch_g4x256.sh
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

sed -i 's/static constexpr size_t kThreadsPerCta = 512;/static constexpr size_t kThreadsPerCta = 256;/' \
  xla/backends/gpu/runtime/ragged_all_to_all.cc
grep -q "kThreadsPerCta = 256" xla/backends/gpu/runtime/ragged_all_to_all.cc
sed -i 's/__global__ void __launch_bounds__(512) RaggedAllToAllDeviceKernelImpl/__global__ void __launch_bounds__(256) RaggedAllToAllDeviceKernelImpl/' \
  xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h
grep -q "__launch_bounds__(256) RaggedAllToAllDeviceKernelImpl" \
  xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h
sed -i 's/return std::max<int32_t>(core_count, kMinDeviceKernelCtaCount);/return std::max<int32_t>(4 * core_count, kMinDeviceKernelCtaCount);/' \
  xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
grep -q "4 \* core_count, kMinDeviceKernelCtaCount" xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
sed -i 's/const int64_t sm_cap = std::max<int64_t>(1, core_count);/const int64_t sm_cap = std::max<int64_t>(1, 4 * static_cast<int64_t>(core_count));/' \
  xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
grep -q "4 \* static_cast<int64_t>(core_count)" xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
echo "g4x256 patch applied"

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
