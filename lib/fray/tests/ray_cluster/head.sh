#!/usr/bin/env bash
# Head node entrypoint: build (or refresh) the marin venv from the read-only checkout,
# add ray to it, then run the Ray head in the foreground.
set -euo pipefail
cd /workspace
uv sync --frozen --inexact --package marin-zephyr --group test --no-install-package marin-iris-native
uv pip install --python /opt/venv/bin/python "ray[default]==2.58.0"
exec /opt/venv/bin/ray start --head --port=6379 --num-cpus=2 --num-gpus=0 \
    --memory=8000000000 --object-store-memory=200000000 \
    --include-dashboard=false --disable-usage-stats --block
