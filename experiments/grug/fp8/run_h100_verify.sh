#!/usr/bin/env bash
# Phase B (logbook GFP8-036): verify the PRODUCTION Warpgroup mixed-FP8 wgmma path on a real
# H100 using a jaxlib built from source with the relaxed WGMMAOp::verify() (built by
# build_patched_jaxlib.sh, uploaded to R2). Runs in the Iris iris-task image at Python 3.12
# with the marin GPU env already uv-synced (stock jax / jax-cuda13-plugin 0.10.0).
#
# Steps: pull the custom cp312 jaxlib wheel from R2, force-reinstall it over stock jaxlib
# (keeping the stock cuda13 plugin for execution), then run test_mixed_fp8_wgmma.py. The
# test itself applies the Python emitter patch (primitives.py + wgmma.py); the C++ verifier
# relaxation comes from this swapped jaxlib binary.
set -euo pipefail

PREFIX="marin/tmp/mixed-fp8/"
echo "===== locate + fetch patched jaxlib wheel from R2 ====="
# boto3 in an ephemeral uv env (active venv has no pip); creds from AWS_* env.
WHEEL_PATH="$(uv run --no-project --with boto3 python - "$PREFIX" <<'PY'
import boto3, os, sys
prefix = sys.argv[1]
s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name="auto")
keys = [o["Key"] for o in s3.list_objects_v2(Bucket="marin-na", Prefix=prefix).get("Contents", [])
        if o["Key"].endswith(".whl") and "jaxlib-" in o["Key"]]
assert keys, f"no jaxlib wheel under s3://marin-na/{prefix}"
key = sorted(keys)[-1]
dest = "/tmp/" + os.path.basename(key)
s3.download_file("marin-na", key, dest)
print(dest)
PY
)"
WHEEL_PATH="$(echo "$WHEEL_PATH" | tail -1)"
echo "wheel: $WHEEL_PATH"
[ -f "$WHEEL_PATH" ] || { echo "download failed"; exit 1; }

echo "===== before swap ====="
python -c "import jaxlib,jax; print('jaxlib', jaxlib.__version__, jaxlib.__file__); print('jax', jax.__version__)"

echo "===== force-reinstall patched jaxlib (no deps; keep stock plugin) ====="
uv pip install --force-reinstall --no-deps "$WHEEL_PATH"
python -c "import jaxlib; print('jaxlib now at', jaxlib.__version__, jaxlib.__file__)"

echo "===== run production-path verification ====="
python -u experiments/grug/fp8/test_mixed_fp8_wgmma.py
