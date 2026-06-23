#!/usr/bin/env bash
# S2: does forcing the operand-QDQ forward to precision=HIGHEST make e4m3xe4m3
# fire $f8 (like the backward grad dots), where DEFAULT declines? Reads $f8 and
# f8e4m3 survival from the compiled forward HLO. d4096, forward-only.
#   nn/default  = control (reproduces the qdq forward fallback, no $f8)
#   nn/highest  = the precision test
#   tn/highest  = precision + cuBLASLt-native layout
set -euo pipefail

K=4096; N=4096

run() {
  local prec=$1
  local layout=$2
  local out="/tmp/qdqprec_${prec}_${layout}.txt"
  echo "=== qdq_prec / precision=${prec} / layout=${layout} (d${K}) ==="
  python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
    --k "$K" --n "$N" --path qdq_prec --precision "$prec" --layout "$layout" \
    --forward-only --steps 20 --warmup 5 > "$out" 2>&1 \
    || { echo "FAILED:"; tail -30 "$out"; return 1; }
  grep -F 'result_json' "$out" || true
  printf '  $f8 matmul count:       %s\n' "$(grep -cF '__cublas$lt$matmul$f8' "$out" || true)"
  printf '  f8e4m3 in compiled HLO: %s\n' "$(grep -cF 'f8e4m3' "$out" || true)"
  printf '  cublas matmul line:     %s\n' "$(grep -oE 'custom_call_target="__cublas[^"]*"' "$out" | head -1 || true)"
}

run default nn
run highest nn
run highest tn
echo "=== DONE — $f8 count >0 at highest => precision is the gate ==="
