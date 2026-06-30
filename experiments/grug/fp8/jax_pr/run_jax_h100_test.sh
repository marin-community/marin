#!/usr/bin/env bash
# PR job C (H100): run the upstream mixed-FP8 wgmma NUMERIC test on a real Hopper GPU using the
# fully-from-source stack (jax@main+patch editable, plus jaxlib/cuda-plugin/cuda-pjrt wheels built
# by jobs A/B and staged in R2). Exercises tests/mosaic/gpu_test.py::WGMMATest::test_wgmma_mixed_fp8.
set -euo pipefail
MARIN_ROOT="$(pwd)"
JAX_SHA="6a19c8b5ae8986e3aba44cb78b4bb024cd1997b2"
PATCH="$MARIN_ROOT/experiments/grug/fp8/jax_pr/mixed_fp8_wgmma.patch"
RUNNER="$MARIN_ROOT/experiments/grug/fp8/jax_pr/run_mixed_fp8_pytest.py"
S3_PREFIX="marin/tmp/mixed-fp8-pr/"
WORK=/tmp/jaxtest

echo "===== 0. clone jax @ $JAX_SHA + apply patch ====="
export DEBIAN_FRONTEND=noninteractive
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
git clone --filter=blob:none https://github.com/jax-ml/jax jax-src
cd jax-src && git checkout --quiet "$JAX_SHA"
git apply "$PATCH"
echo "applied; tree diffstat:"; git --no-pager diff --no-ext-diff --stat

echo "===== 1. fetch built wheels (jaxlib + cuda-plugin + pjrt) from R2 ====="
mkdir -p /tmp/wheels
uv run --no-project --with boto3 python - "$S3_PREFIX" <<'PY'
import boto3, os, sys
prefix = sys.argv[1]
s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name="auto")
objs = s3.list_objects_v2(Bucket="marin-na", Prefix=prefix).get("Contents", [])
# keep the newest of each wheel family (jaxlib / jax_cuda*_plugin / jax_cuda*_pjrt)
import collections
fam = collections.defaultdict(list)
for o in objs:
    k = o["Key"]
    if not k.endswith(".whl"):
        continue
    base = os.path.basename(k).split("-")[0]
    fam[base].append((o["LastModified"], k))
for base, items in fam.items():
    _, key = sorted(items)[-1]
    dest = "/tmp/wheels/" + os.path.basename(key)
    s3.download_file("marin-na", key, dest)
    print("fetched", dest)
PY
ls -la /tmp/wheels/

echo "===== 2. venv: jax(editable) + built wheels + test deps ====="
uv venv /tmp/jvenv --python 3.12
source /tmp/jvenv/bin/activate
uv pip install --quiet -e .                                  # jax(main) + deps (+ stock jaxlib)
uv pip install --quiet -r build/test-requirements.txt
JAXLIB_WHL="$(ls /tmp/wheels/jaxlib-*.whl)"
PJRT_WHL="$(ls /tmp/wheels/jax_cuda*_pjrt-*.whl)"
PLUGIN_WHL="$(ls /tmp/wheels/jax_cuda*_plugin-*.whl)"
uv pip install --quiet --force-reinstall --no-deps "$JAXLIB_WHL"   # keep our patched jaxlib
# Install pjrt + plugin[with-cuda] together as file refs: the plugin pins
# jax-cuda12-pjrt==0.11.0 (our local wheel, not on PyPI), and the [with-cuda] extra
# pulls the nvidia-*-cu12 runtime libs (libcudart etc.) at the versions the wheel pins.
uv pip install --quiet --force-reinstall \
  "jax-cuda12-pjrt @ file://$PJRT_WHL" \
  "jax-cuda12-plugin[with-cuda] @ file://$PLUGIN_WHL"
python -c "import jax, jaxlib; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__)"
python -c "import jax; print('devices:', jax.devices())"

echo "===== 3. numeric test on H100 ====="
python -u "$RUNNER" "$WORK/jax-src"
echo "JOB_C_OK"
