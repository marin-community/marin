#!/usr/bin/env bash

set -euo pipefail

MOK_WHEEL=/app/experiments/grug/moe_hero_ep/mixture_of_kittens-0.1.0-cp312-cp312-linux_aarch64.whl
PGLE_PROFILE_SHA256=cbd3d7f0d0d6ca3bdaf2ff12ce88416f8753ecfc282af4b6ebcaf7f8fd757e4b
PGLE_PROFILE_PATH=/tmp/mok-pgle-all64-p90-${PGLE_PROFILE_SHA256}.pb

uv pip install \
  --python "$IRIS_VENV/bin/python" \
  --link-mode symlink \
  --no-deps \
  "$MOK_WHEEL"

if [[ -n "${MOK_PGLE_PROFILE_URI:-}" ]]; then
  uv run --no-sync fsutil cp "$MOK_PGLE_PROFILE_URI" "$PGLE_PROFILE_PATH"
  actual_sha256=$(uv run --no-sync python -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$PGLE_PROFILE_PATH")
  if [[ "$actual_sha256" != "$PGLE_PROFILE_SHA256" ]]; then
    echo "PGLE profile checksum mismatch: expected $PGLE_PROFILE_SHA256, got $actual_sha256" >&2
    exit 1
  fi
  export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_pgle_profile_file_or_directory_path=$PGLE_PROFILE_PATH"
fi

exec uv run --no-sync python -u -m iris.hooks.multigpu_main \
  --nproc 4 \
  --devices-per-proc 1 \
  -- \
  "$IRIS_VENV/bin/python" -u -m experiments.grug.moe_hero_ep.dev_run "$@"
