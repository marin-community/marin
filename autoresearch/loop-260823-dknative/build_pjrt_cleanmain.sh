#!/bin/bash
# Build jax-cuda13-pjrt from UNPATCHED current XLA main, with the same build
# machinery as the campaign's self-built wheels (see the original build_pjrt.sh
# recipe: jax build/build.py with --local_xla_path on an aarch64 GB200 pod).
#
# Purpose: isolate XLA source vintage in the device-kernel comparison. The
# patched wheel's device kernel ties the one-shot (22.7); the official
# dev20260821 nightly measures 19.4 with the flags main requires. This wheel
# holds build config fixed and moves only the source to clean main.
#
# usage (on a GB200 pod): XLA_REV=<sha> JAX_REV=<sha> STASH=<s3 prefix> build_pjrt_cleanmain.sh
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
echo "xla: $(git rev-parse HEAD)  (clean, no patches)"

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
