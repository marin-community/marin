#!/usr/bin/env bash
# Run every grouped-GEMM baseline in its own process on one GB200 and collect JSON.
# Usage: run_all.sh <out_dir> [--sizes-json sizes.json]
set -euo pipefail
out=${1:?out dir}; shift || true
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
bench="$here/bench_grouped_gemm_baselines.py"

uv run python "$bench" --impl xla "$@" --json "$out/xla.json" 2>&1 | tee "$out/xla.log"
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_use_ragged_dot_fusion=true" \
  uv run python "$bench" --impl xla-cudnn "$@" --json "$out/xla-cudnn.json" 2>&1 | tee "$out/xla-cudnn.log"
uv run python "$bench" --impl triton --sweep "$@" --json "$out/triton.json" 2>&1 | tee "$out/triton.log"
uv run python "$bench" --impl quack --sweep "$@" --json "$out/quack.json" 2>&1 | tee "$out/quack.log"
uv run python "$here/numerics_expert_mlp.py" --hero --json "$out/numerics.json" 2>&1 | tee "$out/numerics.log"
