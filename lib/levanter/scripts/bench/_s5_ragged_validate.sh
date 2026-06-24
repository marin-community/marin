#!/usr/bin/env bash
# S5 H100 validation: BF16 vs FP8 for the Grug MoE expert GEMMs (ragged_dot), Triton backend.
# Decides A-vs-B — does feeding f8 operands into the existing Triton ragged kernel engage f8
# tensor cores (fp8 TFLOP/s > bf16, f8_reaches_gemm true), or upcast internally (no speedup)?
# Each arm prints a `result_json` line and an f8 scan; the last arm also dumps the full HLO.
set +e
B=lib/levanter/scripts/bench/bench_ragged_fp8.py

echo "### ARM 1: bf16 fwd+bwd (real Grug shapes, baseline)"
python "$B" --path bf16 --implementation triton --no-print-hlo

echo "### ARM 2: fp8 fwd+bwd (real Grug shapes) — KEY timing + f8 scan"
python "$B" --path fp8 --implementation triton --no-print-hlo

echo "### ARM 3: bf16 fwd-only (real Grug shapes)"
python "$B" --path bf16 --implementation triton --forward-only --no-print-hlo

echo "### ARM 4: fp8 fwd-only (real Grug shapes) — KEY fwd timing + f8 scan"
python "$B" --path fp8 --implementation triton --forward-only --no-print-hlo

echo "### ARM 5: fp8 fwd-only (small) WITH full HLO dump — guaranteed f8-operand HLO"
python "$B" --path fp8 --implementation triton --forward-only --tokens 1024 --hidden 512 --intermediate 1024 --experts 8

echo "### DONE"
