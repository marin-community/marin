#!/usr/bin/env bash
set -euo pipefail

job_name="${1:?usage: validate_snowball_h100.sh JOB_NAME}"

# Sync both GPU extras once. marin-core selects the cu128 Torch wheel and Iris
# restores JAX's CUDA 13 libraries after sync. A later uv sync undoes that order.
# Limit pytest's automatic worker count to the one assigned GPU.
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --target-cluster cw-us-east-02a \
  --gpu H100x1 \
  --cpu 8 \
  --memory 64GB \
  --disk 64GB \
  --timeout 3600 \
  --priority interactive \
  --sync-package marin-core \
  --sync-package marin-levanter \
  --extra gpu \
  --job-name "$job_name" \
  -e JAX_PLATFORMS cuda \
  -e PYTEST_XDIST_AUTO_NUM_WORKERS 1 \
  -- bash -lc '
    set -e
    uv pip install --python .venv/bin/python --link-mode copy \
      pytest==9.0.3 pytest-asyncio==1.4.0 pytest-forked==1.6.0 \
      pytest-timeout==2.4.0 pytest-xdist==3.8.0
    .venv/bin/python scripts/ci/preflight_h100_torch_jax.py
    exec .venv/bin/python -m pytest \
      tests/test_june_snowball_model.py lib/levanter/tests/test_snowball.py -q
  '
