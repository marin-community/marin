#!/usr/bin/env bash
# PR job A (mixed E4M3/E5M2 wgmma upstream PR): build jaxlib from jax@main+patch, then run the
# MLIR dialect verifier test on CPU -- this exercises the jaxlib C++ WGMMAOp::verify() change
# WITHOUT a GPU (the mosaic-GPU dialect, incl. the verifier, is compiled into jaxlib, not the
# CUDA plugin). Uploads the jaxlib wheel to R2 for the H100 numeric job.
#
# Runs in the Iris iris-task image (python:3.12-slim, root, gcc-14; clang + bazel installed here).
set -euo pipefail
MARIN_ROOT="$(pwd)"                       # iris bundles the marin worktree to /app
JAX_SHA="6a19c8b5ae8986e3aba44cb78b4bb024cd1997b2"
PATCH="$MARIN_ROOT/experiments/grug/fp8/jax_pr/mixed_fp8_wgmma.patch"
WORK=/tmp/jaxbuild

echo "===== 0. toolchain ====="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq clang
curl -fsSL -o /usr/local/bin/bazel \
  https://github.com/bazelbuild/bazel/releases/download/7.7.0/bazel-7.7.0-linux-x86_64
chmod +x /usr/local/bin/bazel
clang --version | head -1; bazel --version

echo "===== 1. clone jax @ $JAX_SHA + apply patch ====="
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
git clone --filter=blob:none https://github.com/jax-ml/jax jax-src   # blobless: any SHA checkoutable
cd jax-src && git checkout --quiet "$JAX_SHA"
git apply --stat "$PATCH"
git apply "$PATCH"
echo "applied; tree diffstat:"; git --no-pager diff --no-ext-diff --stat

echo "===== 2. build jaxlib (CPU, hermetic py3.12) ====="
# ML_WHEEL_TYPE=release stamps a clean version (0.11.0, no dev0+selfbuilt suffix) so that
# jax.version._version == jax.lib.__version__ -- the mosaic dialect tests skip otherwise.
python3 build/build.py build --wheels=jaxlib --python_version=3.12 \
  --clang_path=/usr/bin/clang \
  --bazel_options=--jobs=128 --bazel_options=--repo_env=ML_WHEEL_TYPE=release
JAXLIB_WHL="$(find "$WORK/jax-src/dist" -name 'jaxlib-*.whl' | head -1)"
echo "JAXLIB_WHL=$JAXLIB_WHL"; [ -n "$JAXLIB_WHL" ] || { echo "no jaxlib wheel"; exit 1; }

echo "===== 3. upload jaxlib wheel to R2 ====="
WHEEL="$JAXLIB_WHL" uv run --no-project --with boto3 python - <<'PY'
import boto3, os
w = os.environ["WHEEL"]; key = "marin/tmp/mixed-fp8-pr/" + os.path.basename(w)
boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name="auto").upload_file(w, "marin-na", key)
print("UPLOADED s3://marin-na/" + key)
PY

echo "===== 4. venv: jax(editable) + my jaxlib + test deps ====="
uv venv /tmp/jvenv --python 3.12
source /tmp/jvenv/bin/activate
uv pip install --quiet -e .                                   # jax(main) + deps + stock jaxlib
uv pip install --quiet -r build/test-requirements.txt
uv pip install --quiet --force-reinstall --no-deps "$JAXLIB_WHL"   # override jaxlib with our build
python -c "import jax, jaxlib; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__, jaxlib.__file__)"

echo "===== 5. C++ verifier test (MLIR dialect, CPU) ====="
# Exercises WGMMAOp::verify(): test_wgmma_types_match (non-fp8 mismatch still rejected) +
# test_wgmma_mixed_fp8_operands_are_allowed (the new e4m3/e5m2 acceptance).
JAX_PLATFORMS=cpu python -m pytest tests/mosaic/gpu_dialect_test.py -k "wgmma" -v
echo "DIALECT_VERIFIER_TESTS_PASSED"
echo "JOB_A_OK"
