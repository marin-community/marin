#!/usr/bin/env bash
# S2: does TN layout remove the XLA-inserted f8 transpose and recover a speedup?
# Runs the manual direct-f8 forward under TN vs NN, plus a bf16 TN baseline, at
# d4096. For each: result_json (TFLOP/s), the $f8 custom-call count, and the
# input_transpose_fusion count (expect 0 for TN, >=1 for NN).
set -euo pipefail

K=4096; N=4096

run() {
  local path=$1
  local layout=$2
  local out="/tmp/${path}_${layout}.txt"
  echo "=== ${path} / ${layout} (d${K}) ==="
  python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
    --k "$K" --n "$N" --path "$path" --layout "$layout" --forward-only --steps 20 --warmup 5 > "$out" 2>&1 \
    || { echo "FAILED:"; tail -30 "$out"; return 1; }
  grep -F 'result_json' "$out" || true
  printf '  $f8 matmul count:        %s\n' "$(grep -cF '__cublas$lt$matmul$f8' "$out" || true)"
  printf '  input_transpose_fusion:  %s\n' "$(grep -cF 'input_transpose_fusion' "$out" || true)"
  printf '  cublas-gemm operands:    %s\n' "$(grep -oE 'custom-call\([^)]*\), custom_call_target="__cublas[^"]*"' "$out" | head -1 || true)"
}

run manual tn
run manual nn
run bf16 tn
echo "=== DONE — compare fwd_tflops_per_s; TN manual should drop the transpose ==="
