#!/bin/bash
# Build jax-cuda13-pjrt: g8x128 dk grid patch + 4-way ILP unroll of the LSA copy loop.
#
# leg-4 cell b1. The dk LSA copy is a plain grid-stride `dst[i] = src[i]` -- one dependent
# 16B load->remote-store chain per iteration. The one-shot kernel hides NVLink store latency
# with ~10x CTA oversubscription; the dk cannot grow its grid freely (per-CTA barrier slots),
# so buy memory-level parallelism inside the thread instead: batch 4 independent loads, then
# 4 independent stores per iteration.
#
# usage: XLA_REV=... JAX_REV=... STASH=s3://... build_pjrt_mainpatch_g8x128ilp4.sh
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

# --- g8x128 grid patch (identical to build_pjrt_mainpatch_g8x128.sh) ---
sed -i 's/static constexpr size_t kThreadsPerCta = 512;/static constexpr size_t kThreadsPerCta = 128;/' \
  xla/backends/gpu/runtime/ragged_all_to_all.cc
grep -q "kThreadsPerCta = 128" xla/backends/gpu/runtime/ragged_all_to_all.cc
sed -i 's/__global__ void __launch_bounds__(512) RaggedAllToAllDeviceKernelImpl/__global__ void __launch_bounds__(128) RaggedAllToAllDeviceKernelImpl/' \
  xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h
grep -q "__launch_bounds__(128) RaggedAllToAllDeviceKernelImpl" \
  xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h
sed -i 's/return std::max<int32_t>(core_count, kMinDeviceKernelCtaCount);/return std::max<int32_t>(8 * core_count, kMinDeviceKernelCtaCount);/' \
  xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
grep -q "8 \* core_count, kMinDeviceKernelCtaCount" xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
sed -i 's/const int64_t sm_cap = std::max<int64_t>(1, core_count);/const int64_t sm_cap = std::max<int64_t>(1, 8 * static_cast<int64_t>(core_count));/' \
  xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
grep -q "8 \* static_cast<int64_t>(core_count)" xla/backends/gpu/runtime/ragged_all_to_all_thunk.h
echo "g8x128 patch applied"

# --- ilp4 patch: unroll the LSA copy loop with independent load/store batches ---
python3 - <<'EOF'
path = "xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h"
src = open(path).read()
old = """        const int64_t num_elements = meta.byte_count / kVectorSize;
        for (int64_t i = unit_tid; i < num_elements; i += unit_nthreads) {
          dst[i] = src[i];
        }"""
new = """        const int64_t num_elements = meta.byte_count / kVectorSize;
        const int64_t stride = unit_nthreads;
        int64_t i = unit_tid;
        for (; i + 3 * stride < num_elements; i += 4 * stride) {
          T v0 = src[i];
          T v1 = src[i + stride];
          T v2 = src[i + 2 * stride];
          T v3 = src[i + 3 * stride];
          dst[i] = v0;
          dst[i + stride] = v1;
          dst[i + 2 * stride] = v2;
          dst[i + 3 * stride] = v3;
        }
        for (; i < num_elements; i += stride) {
          dst[i] = src[i];
        }"""
assert src.count(old) == 1, "ilp4 anchor not found exactly once"
open(path, "w").write(src.replace(old, new))
print("ilp4 patch applied")
EOF

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
