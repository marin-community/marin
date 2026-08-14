#!/usr/bin/env bash
# autoresearch guard: targeted MoE tests, CPU locally + GPU on one GB200 node.
# Must exit 0 for a candidate change to be kept. Uses 1 node (4 GPUs) — within
# the 2-rack quota even while a verify rack is winding down.
set -euo pipefail

REPO="${AR_REPO:-/home/marin/projects/marin}"
cd "$REPO"
IRIS=(uv run iris --config lib/iris/config/marin.yaml)
SHA="$(git rev-parse --short HEAD)"

echo "guard[1/2]: local CPU-safe targeted tests" >&2
uv run pytest -q \
  tests/test_moe_hero_ep.py \
  tests/test_grug_variant_contracts.py \
  lib/levanter/tests/grug/test_grugformer_moe.py

echo "guard[2/2]: GPU tests on one GB200 node (bundles the working tree)" >&2
# GPU tests self-skip without a GPU backend, so they only actually run here.
# Default pod jax is CPU-only: sync --extra gpu first (cuDNN comes with it).
# -n 4: the repo default xdist worker count (~2x CPUs) OOMs the GPUs when every
# worker initializes JAX; preallocate off so the 4 workers share the 4 devices.
# Baseline-relative pass: clean main already fails some of these on GB200
# (see guard_allowlist.txt), so we require NO NEW failures, not zero failures.
gpulog=$(mktemp)
"${IRIS[@]}" job run --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --gpu "GB200x4" --cpu 64 --memory 256GB --disk 128GB \
  --job-name "ar8062-guard-${SHA}" \
  -e IRIS_USER mwittmann \
  -e XLA_PYTHON_CLIENT_PREALLOCATE false \
  -- bash -c "uv sync --all-packages --extra gpu >&2 && uv run pytest -q -p no:randomly -n 4 \
        lib/levanter/tests/grug/test_grugformer_moe.py \
        lib/levanter/tests/grug/test_fa4_cute_attention.py \
        tests/test_moe_hero_ep.py" >"$gpulog" 2>&1 || true
cat "$gpulog" >&2
grep -qE "[0-9]+ passed" "$gpulog" || { echo "guard: GPU pytest never reported results" >&2; exit 1; }
# Task-log lines may carry an "...| " stream prefix; strip it before matching, and
# tolerate zero FAILED lines under set -e.
new_failures=$( { sed 's/.*| //' "$gpulog" | grep "^FAILED" || true; } | sed 's/^FAILED //; s/ - .*//' | sort -u \
  | comm -13 <(grep -v '^#' "$(dirname "$0")/guard_allowlist.txt" | sort -u) -)
if [[ -n "$new_failures" ]]; then
  echo "guard: NEW GPU test failures (not in allowlist):" >&2
  echo "$new_failures" >&2
  exit 1
fi
echo "guard: GPU failures all within allowlist" >&2
