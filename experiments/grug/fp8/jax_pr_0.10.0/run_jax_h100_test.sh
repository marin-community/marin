#!/usr/bin/env bash
# 0.10.0 BACKPORT job C (H100, HALIAX-MATCHED): run the mixed-FP8 wgmma NUMERIC test on a real
# Hopper GPU using the EXACT stack haliax pins -- jax 0.10.0 + jaxlib 0.10.0 + STOCK jax-cuda13
# plugin/pjrt 0.10.0 (marin/haliax uv.lock resolves jax==0.10.0, jaxlib==0.10.0, jax-cuda13-*==0.10.0).
#
# Our custom jaxlib 0.10.0 (carries the WGMMAOp::verify() relaxation) drops in over the stock
# jaxlib==0.10.0; the mosaic-GPU dialect+verifier lives in jaxlib (NOT the plugin), so the custom
# jaxlib drives the relaxed lowering and the stock cuda13 plugin just executes -- proving the patch
# is a clean drop-in for the haliax env. Exercises tests/mosaic/gpu_test.py::WGMMATest::test_wgmma_mixed_fp8.
set -euo pipefail
MARIN_ROOT="$(pwd)"
JAX_SHA="a33ed614c58ee8a10d0b7536c50c2609c38500c1"   # jax-v0.10.0
PATCH="$MARIN_ROOT/experiments/grug/fp8/jax_pr_0.10.0/mixed_fp8_wgmma_0.10.0.patch"
RUNNER="$MARIN_ROOT/experiments/grug/fp8/jax_pr_0.10.0/run_mixed_fp8_pytest.py"
S3_PREFIX="marin/tmp/mixed-fp8-pr-0.10.0/"
WORK=/tmp/jaxtest

echo "===== 0. clone jax @ $JAX_SHA (jax-v0.10.0) + apply patch ====="
export DEBIAN_FRONTEND=noninteractive
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
git clone --filter=blob:none https://github.com/jax-ml/jax jax-src
cd jax-src && git checkout --quiet "$JAX_SHA"
git apply "$PATCH"
echo "applied; tree diffstat:"; git --no-pager diff --no-ext-diff --stat
echo "jax version:"; grep '_version =' jax/version.py | head -1

echo "===== 1. fetch our custom jaxlib wheel from R2 (0.10.0 prefix) ====="
mkdir -p /tmp/wheels
S3_PREFIX="$S3_PREFIX" uv run --no-project --with boto3 python - <<'PY'
import boto3, os
prefix = os.environ["S3_PREFIX"]
s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name="auto")
objs = s3.list_objects_v2(Bucket="marin-na", Prefix=prefix).get("Contents", [])
cand = [(o["LastModified"], o["Key"]) for o in objs
        if o["Key"].endswith(".whl") and os.path.basename(o["Key"]).startswith("jaxlib-")]
assert cand, "no jaxlib wheel under " + prefix
_, key = sorted(cand)[-1]
dest = "/tmp/wheels/" + os.path.basename(key)
s3.download_file("marin-na", key, dest)
print("fetched", dest)
PY
ls -la /tmp/wheels/

echo "===== 2. venv: jax(editable) + STOCK cuda13 plugin/pjrt 0.10.0 + our jaxlib + test deps ====="
uv venv /tmp/jvenv --python 3.12
source /tmp/jvenv/bin/activate
uv pip install --quiet -e .                                  # jax(0.10.0) + deps (+ stock jaxlib)
uv pip install --quiet -r build/test-requirements.txt
# Stock matched cuda13 plugin + pjrt from PyPI -- the exact versions haliax's uv.lock resolves.
# [with-cuda] pulls the nvidia-*-cu13 runtime libs (libcudart, ptxas, libdevice, ...).
uv pip install --quiet "jax-cuda13-pjrt==0.10.0" "jax-cuda13-plugin[with-cuda]==0.10.0"
# Override jaxlib with our patched build (verifier relaxation); --no-deps keeps the stack pinned.
JAXLIB_WHL="$(ls /tmp/wheels/jaxlib-*.whl)"
uv pip install --quiet --force-reinstall --no-deps "$JAXLIB_WHL"
python -c "import jax, jaxlib; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__)"
python -c "import jax; print('devices:', jax.devices())"

echo "===== 3. numeric test on H100 ====="
python -u "$RUNNER" "$WORK/jax-src"
echo "JOB_C_OK"
