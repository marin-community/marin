#!/bin/bash
# Build jax-cuda13-pjrt from XLA main (cleanmain pins) with the ragged a2a device kernel
# launching an 8x-SM grid of 128-thread CTAs (vs stock 1x-SM grid of 512-thread CTAs).
#
# Leg-2b iteration 3. Trajectory: stock dk 19.37 -> t1024 null -> g4x256 21.02 (+1.65).
# Per-peer CTA parallelism is the confirmed bottleneck; this step doubles concurrent CTAs
# per peer again (~16) at the same 1024 threads/SM residency (8 CTAs/SM x 128 threads),
# mirroring the one-shot copy kernel's 128-thread block geometry. Barrier registration
# scales with the grid.
#
# usage: XLA_REV=... JAX_REV=... STASH=s3://... build_pjrt_mainpatch_g8x128mm.sh
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
python3 - <<'PYEOF'
from pathlib import Path
f = Path("xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h")
src = f.read_text()
old = """    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev_comm,
                                           ncclTeamTagLsa{}, blockIdx.x};"""
new = """    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev_comm,
                                           ncclTeamTagLsa{}, blockIdx.x,
                                           /*multimem=*/true};"""
assert src.count(old) == 1, "barrier ctor anchor not found"
f.write_text(src.replace(old, new))
print("multimem barrier patch applied")
PYEOF
echo "g8x128mm patch applied"

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
