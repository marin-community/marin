#!/usr/bin/env bash

set -euo pipefail

MOK_WHEEL=/app/experiments/grug/moe_hero_ep/mixture_of_kittens-0.1.0-cp312-cp312-linux_aarch64.whl

/bin/uv pip install \
  --python "$IRIS_VENV/bin/python" \
  --link-mode symlink \
  --no-deps \
  "$MOK_WHEEL"

run_profile() {
  "$IRIS_PYTHON" -m iris.hooks.multigpu_main \
    --nproc 4 \
    --devices-per-proc 1 \
    -- \
    "$IRIS_PYTHON" -m experiments.grug.moe_hero_ep.dev_run "$@"
}

run_profile \
  --run-id mark-mok-profile-parity-mok-mb8192-100-s80n5-r9-xprof-20260808 \
  --num-steps 100 \
  --stop-after-steps 85 \
  --backend mok \
  --profile-start-step 80 \
  --profile-num-steps 5 \
  --mok-minibatch-size 8192 \
  --mok-fwd-num-comm-sms 40 \
  --mok-bwd-num-comm-sms 28

run_profile \
  --run-id mark-mok-profile-parity-deepep-100-s80n5-r9-xprof-20260808 \
  --num-steps 100 \
  --stop-after-steps 85 \
  --backend fixed \
  --profile-start-step 80 \
  --profile-num-steps 5
