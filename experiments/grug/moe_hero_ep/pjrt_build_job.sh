#!/usr/bin/env bash
# Build jax-cuda13-pjrt from a marin-community/xla ref on an aarch64 GB200 node, as an Iris job.
#
# Two things the fork's CI build does not do: it runs on the cluster (a 144-vCPU Grace node builds
# in a fraction of the 3.5 h the 4-job GitHub ARM runner takes), and it builds against a pip cuDNN
# instead of jax's hermetic 9.12. cudnn-frontend gates its MoE grouped matmul on CUDNN_VERSION >=
# 9.15 at compile time, so XLA's cuDNN ragged-dot fusion is dead in any wheel built at the default.
#
# usage (inside the job): XLA_REF=<branch|sha> [CUDNN_VERSION=9.25.1.1] [BAZEL_JOBS=64] \
#        [S3_DEST=s3://.../prefix] bash experiments/grug/moe_hero_ep/pjrt_build_job.sh
set -euo pipefail

XLA_REF="${XLA_REF:?set XLA_REF}"
CUDNN_VERSION="${CUDNN_VERSION:-9.25.1.1}"
BAZEL_JOBS="${BAZEL_JOBS:-64}"
S3_DEST="${S3_DEST:-s3://marin-us-east-02a/marin/research/mcwitt-ragged-dot/pjrt}"
WORK="${WORK:-/app/xla-build}"
PY="$(command -v python)"
BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-arm64"

echo "== host"; uname -m; nproc; head -2 /proc/meminfo; df -h /app /tmp | tail -2; date -u

echo "== toolchain"
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends clang lld jq xz-utils patch >/dev/null
clang --version | head -1
curl -fsSL -o /usr/local/bin/bazel "$BAZELISK_URL" && chmod +x /usr/local/bin/bazel

mkdir -p "$WORK" && cd "$WORK"
echo "== sources"
git clone -q --filter=blob:none https://github.com/marin-community/xla.git xla
git -C xla fetch -q origin "$XLA_REF" && git -C xla checkout -q --detach FETCH_HEAD
XLA_SHA="$(git -C xla rev-parse HEAD)"
JAX_COMMIT="$(jq -r .jax_commit xla/marin/release/config.json)"
JAX_VERSION="$(jq -r .jax_version xla/marin/release/config.json)"
git clone -q --filter=blob:none https://github.com/jax-ml/jax.git jax
git -C jax checkout -q --detach "$JAX_COMMIT"
echo "xla $XLA_SHA (ref $XLA_REF)"; echo "jax $JAX_COMMIT ($JAX_VERSION)"

echo "== cuDNN $CUDNN_VERSION headers for the hermetic build"
uv pip install --python "$PY" "nvidia-cudnn-cu13==$CUDNN_VERSION" 2>&1 | tail -1
LOCAL_CUDNN_PATH="$("$PY" -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')"
grep -h "define CUDNN_MAJOR\|define CUDNN_MINOR\|define CUDNN_PATCHLEVEL" "$LOCAL_CUDNN_PATH/include/cudnn_version.h"

SUFFIX="+marin.${XLA_SHA:0:12}"
echo "== build ${JAX_VERSION}${SUFFIX} with $BAZEL_JOBS jobs"; date -u
cd jax
"$PY" build/build.py build \
  --wheels=jax-cuda-pjrt \
  --cuda_major_version=13 \
  --python_version=3.12 \
  --local_xla_path="$WORK/xla" \
  --bazel_startup_options=--output_user_root="$WORK/bazel" \
  --bazel_options=--jobs="$BAZEL_JOBS" \
  --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
  --bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX="$SUFFIX" \
  --bazel_options=--repo_env=LOCAL_CUDNN_PATH="$LOCAL_CUDNN_PATH" \
  --bazel_options=--define=ynn_enable_arm64_neonfp8=false \
  --verbose 2>&1 | grep -v "^\[[0-9,]* / [0-9,]*\]" | tail -400
date -u
ls -la dist/
WHEEL="$(ls dist/jax_cuda13_pjrt-*.whl)"
sha256sum "$WHEEL"
mkdir -p "${IRIS_OUTPUT_DIR:-/tmp/out}" && cp "$WHEEL" "${IRIS_OUTPUT_DIR:-/tmp/out}/"
"$PY" - "$WHEEL" "$S3_DEST/${XLA_SHA:0:12}" <<'PY'
import os, sys
import fsspec
wheel, dest = sys.argv[1], sys.argv[2]
fs = fsspec.filesystem("s3")
target = f"{dest.rstrip('/')}/{os.path.basename(wheel)}"
fs.put(wheel, target)
print("uploaded", target, fs.size(target))
PY
