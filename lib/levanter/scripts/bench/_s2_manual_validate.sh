#!/usr/bin/env bash
# S2 validation: does the manual direct-f8 forward (dq(dot(q(x), q(w)))) fire an
# f8 matmul on H100, where the operand-QDQ forward falls back to bf16? Reads the
# decisive signals straight from the bench's compiled (optimized) forward HLO —
# no per-pass dump needed: (1) does __cublas$lt$matmul$f8 appear; (2) does f8e4m3
# survive into the optimized module and feed the GEMM (vs qdq, whose f8 is
# stripped); (3) forward TFLOP/s vs a bf16 baseline at the same shape.
set -euo pipefail

K=4096; N=4096
M=/tmp/manual_fwd.txt
B=/tmp/bf16_fwd.txt

echo "=== MANUAL forward (d${K}) ==="
python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
  --k "$K" --n "$N" --path manual --forward-only --steps 20 --warmup 5 > "$M" 2>&1 \
  || { echo "MANUAL run failed:"; tail -40 "$M"; exit 1; }

echo "--- result_json (manual) ---";        grep -F 'result_json' "$M" || true
echo "--- \$f8 fired? (match count) ---";    grep -cF '__cublas$lt$matmul$f8' "$M" || true
echo "--- f8e4m3 occurrences in compiled HLO (0 => stripped like qdq) ---"
grep -cF 'f8e4m3' "$M" || true
echo "--- matmul / custom-call / gemm-fusion lines (operand dtype = f8 vs bf16) ---"
grep -nE 'custom-call\(|cublas|gemm_fusion|kind=kCustom|ROOT %.*dot\(|= f[0-9].*dot\(|= bf16.*dot\(' "$M" | head -25 || true

echo "=== BF16 baseline forward (d${K}) ==="
python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
  --k "$K" --n "$N" --path bf16 --forward-only --steps 20 --warmup 5 --no-print-hlo > "$B" 2>&1 \
  || { echo "BF16 run failed:"; tail -20 "$B"; exit 1; }
echo "--- result_json (bf16) ---"; grep -F 'result_json' "$B" || true
echo "=== DONE — compare manual vs bf16 fwd_tflops_per_s ==="
