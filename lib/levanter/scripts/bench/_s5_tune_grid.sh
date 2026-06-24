#!/usr/bin/env bash
# S5 H100 — skill-style autotune of the Triton ragged MoE GEMM (bf16 vs FP8) at the REAL
# Grug trial-model regime (hidden=1024, intermediate=512, {8,64} experts), replacing the
# earlier mis-sized microbench. Bounded block/tile grid swept per shape bucket; raw results
# saved as a JSON artifact; best-config table + fp8-vs-bf16 verdict printed.
set +e
mkdir -p "$PWD/_s5_artifacts"
python lib/levanter/scripts/tune/tune_ragged_fp8.py \
  --steps 12 --warmup 4 \
  --artifact "$PWD/_s5_artifacts/ragged_fp8_autotune.json"
echo "### DONE"
