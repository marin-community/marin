#!/usr/bin/env bash
# PR job B: build jax-cuda-plugin + jax-cuda-pjrt from jax@main+patch (CUDA 13, sm_90 for H100),
# so the H100 numeric job runs a fully-from-source stack matching the patched jaxlib. The plugin
# does NOT carry the wgmma change (that's in jaxlib); we build it from the same SHA only for
# version/ABI consistency with the patched jaxlib. Uploads both wheels to R2.
set -euo pipefail
MARIN_ROOT="$(pwd)"
JAX_SHA="6a19c8b5ae8986e3aba44cb78b4bb024cd1997b2"
PATCH="$MARIN_ROOT/experiments/grug/fp8/jax_pr/mixed_fp8_wgmma.patch"
WORK=/tmp/jaxbuild

echo "===== 0. toolchain ====="
export DEBIAN_FRONTEND=noninteractive
# g++-12: clang's CUDA device compilation chokes on the image's gcc-14 libstdc++
# (__or_fn / __glibcxx_assert_fail host/device errors); point clang at gcc-12's libstdc++.
apt-get update -qq && apt-get install -y -qq clang g++-12
GCC12_DIR="$(ls -d /usr/lib/gcc/x86_64-linux-gnu/12 2>/dev/null || true)"
echo "gcc-12 dir: $GCC12_DIR"
curl -fsSL -o /usr/local/bin/bazel \
  https://github.com/bazelbuild/bazel/releases/download/7.7.0/bazel-7.7.0-linux-x86_64
chmod +x /usr/local/bin/bazel
clang --version | head -1; bazel --version

echo "===== 1. clone jax @ $JAX_SHA + apply patch ====="
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
git clone --filter=blob:none https://github.com/jax-ml/jax jax-src
cd jax-src && git checkout --quiet "$JAX_SHA"
git apply "$PATCH"
echo "applied; tree diffstat:"; git --no-pager diff --no-ext-diff --stat

echo "===== 2. build cuda plugin + pjrt (CUDA 13.0, sm_90) ====="
# --cuda_version must be the 3-part hermetic version (matches .bazelrc cuda_v13 = 13.0.0 and the
# supported CUDA_UMD redist list); a 2-part "13.0" overrides the config's value and fails the
# UMD redist lookup. cuDNN/NCCL/UMD come from the cuda13 config and are left untouched.
# Compile CUDA with clang-19, not nvcc: CUDA's nvcc rejects Debian trixie's glibc headers
# (getlogin_r/gethostname linkage errors). Use CUDA 12.9 (not 13): clang-19 predates CUDA 13 and
# its bundled wrapper references headers (texture_fetch_functions.h) that CUDA 13 removed; CUDA 12
# still has them. jaxlib is CUDA-version-agnostic, so a cuda12 plugin pairs fine on H100.
# ML_WHEEL_TYPE=release: stamp clean 0.11.0 versions, consistent with the release-versioned jaxlib.
# --disable_nccl: NCCL's DOCA/GPUNetIO device headers use the `typeof` GNU extension that clang
# rejects in CUDA mode; NCCL is not needed for the single-GPU wgmma numeric test.
python3 build/build.py build --wheels=jax-cuda-plugin,jax-cuda-pjrt --python_version=3.12 \
  --cuda_version=12.9.1 --cuda_compute_capabilities=sm_90 --build_cuda_with_clang --disable_nccl \
  --clang_path=/usr/bin/clang \
  --bazel_options=--jobs=128 --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
  --bazel_options=--copt=--gcc-install-dir="$GCC12_DIR" \
  --bazel_options=--cxxopt=--gcc-install-dir="$GCC12_DIR"

echo "===== 3. upload plugin + pjrt wheels to R2 ====="
mapfile -t WHEELS < <(find "$WORK/jax-src/dist" -name 'jax_cuda*.whl')
echo "found: ${WHEELS[*]}"; [ "${#WHEELS[@]}" -ge 2 ] || { echo "expected >=2 cuda wheels"; exit 1; }
for w in "${WHEELS[@]}"; do
  WHEEL="$w" uv run --no-project --with boto3 python - <<'PY'
import boto3, os
w = os.environ["WHEEL"]; key = "marin/tmp/mixed-fp8-pr/" + os.path.basename(w)
boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name="auto").upload_file(w, "marin-na", key)
print("UPLOADED s3://marin-na/" + key)
PY
done
echo "JOB_B_OK"
