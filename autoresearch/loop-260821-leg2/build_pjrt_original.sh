#!/bin/bash
# Build jax-cuda13-pjrt at the dev20260811 pins (jax 8d1be7d / xla 60f8069)
# with two patches:
#   1. kMaxPeers 32 -> 128 (multi_gpu_barrier_kernel.h), as in the #8077 wheel.
#   2. 4096-byte buffer-assignment alignment for the collective memory space
#      color (compile_module_to_llvm_ir.cc) so NCCL window registration of
#      packed mosaic collective-metadata buffers meets NCCL_WIN_REQUIRED_ALIGNMENT.
set -euo pipefail
mkdir -p /tmp/build
cd /tmp/build

if [ ! -d jax ]; then
  git clone https://github.com/jax-ml/jax.git jax
fi
cd /tmp/build/jax
git checkout 8d1be7d

if [ ! -d /tmp/build/xla ]; then
  git clone https://github.com/openxla/xla.git /tmp/build/xla
fi
cd /tmp/build/xla
git checkout 60f8069
git reset --hard 60f8069

# Patch 1: kMaxPeers (both barrier kernel variants).
sed -i 's/kMaxPeers = 32;/kMaxPeers = 128;/g' xla/stream_executor/gpu/multi_gpu_barrier_kernel.h
grep -n "kMaxPeers = 128" xla/stream_executor/gpu/multi_gpu_barrier_kernel.h

# Patch 2: 4096-byte buffer-assignment alignment for ALL colors. Mosaic
# collective-metadata params can be registered as NCCL windows from the
# default arena (the collective-copy analysis misses some inputs), and
# NCCL_WIN_REQUIRED_ALIGNMENT is 4096 — 256-byte packing offsets fail.
python3 - <<'EOF'
path = "xla/service/gpu/compile_module_to_llvm_ir.cc"
src = open(path).read()
old = "[](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },"
new = ("[](LogicalBuffer::Color) {\n"
       "            // NCCL window registration of mosaic collective-metadata\n"
       "            // params requires NCCL_WIN_REQUIRED_ALIGNMENT (4096).\n"
       "            return int64_t{4096};\n"
       "          },")
assert src.count(old) == 1, src.count(old)
open(path, "w").write(src.replace(old, new))
print("alignment patch applied")
EOF

# Patch 3: round every BFC request to 4096 bytes. Regions are MB-multiples
# and all chunk sizes become 4KiB multiples, so every chunk offset stays
# 4KiB-aligned — the alignment NCCL window registration requires for
# collective-memory-space buffers. Bin sizing (kMinAllocationBits) is left
# untouched; bumping it trips the BinForSize construction invariant.
python3 - <<'EOF'
path = "xla/tsl/framework/bfc_allocator.cc"
src = open(path).read()
old = ("size_t rounded_bytes =\n"
       "      (kMinAllocationSize *\n"
       "       ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));")
new = ("// 4096 = NCCL_WIN_REQUIRED_ALIGNMENT; keeps every chunk offset\n"
       "  // window-registrable.\n"
       "  constexpr size_t kWinAlign = 4096;\n"
       "  size_t rounded_bytes = (kWinAlign * ((bytes + kWinAlign - 1) / kWinAlign));")
assert src.count(old) == 1, src.count(old)
open(path, "w").write(src.replace(old, new))
print("bfc rounding patch applied")
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
echo BUILD_OK
