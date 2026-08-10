#!/usr/bin/env bash

set -euo pipefail

MOK_WHEEL=/app/experiments/grug/moe_hero_ep/mixture_of_kittens-0.1.0-cp312-cp312-linux_aarch64.whl
PROFILE_ROOT=s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/k3sc0re/mok-vs-deepep-20260808

/bin/uv pip install \
  --python "$IRIS_VENV/bin/python" \
  --link-mode symlink \
  --no-deps \
  "$MOK_WHEEL"

run_profile() {
  local output_uri=$1
  shift
  "$IRIS_PYTHON" -m iris.hooks.multigpu_main \
    --nproc 4 \
    --devices-per-proc 1 \
    -- \
    bash -c 'exec "$IRIS_PYTHON" -m iris.hooks.nsys_main "$@"' iris-nsys \
    --tasks first \
    --capture-range \
    --output-uri "$output_uri" \
    -- \
    "$IRIS_PYTHON" -m experiments.grug.moe_hero_ep.dev_run "$@"
}

run_profile "$PROFILE_ROOT/mok" \
  --run-id mark-mok-profile-parity-mok-mb8192-100-s80n5-r8-20260808 \
  --num-steps 100 \
  --stop-after-steps 85 \
  --backend mok \
  --profile-start-step 80 \
  --profile-num-steps 5 \
  --mok-minibatch-size 8192 \
  --mok-fwd-num-comm-sms 40 \
  --mok-bwd-num-comm-sms 28

run_profile "$PROFILE_ROOT/deepep" \
  --run-id mark-mok-profile-parity-deepep-100-s80n5-r8-20260808 \
  --num-steps 100 \
  --stop-after-steps 85 \
  --backend fixed \
  --profile-start-step 80 \
  --profile-num-steps 5
