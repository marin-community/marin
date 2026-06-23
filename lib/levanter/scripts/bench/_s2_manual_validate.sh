#!/usr/bin/env bash
# S2 validation: does the manual direct-f8 forward (dq(dot(q(x), q(w)))) fire an
# f8 matmul on H100, where the operand-QDQ forward falls back to bf16? Decisive
# signals: (1) does __cublas$lt$matmul$f8 appear; (2) does f8e4m3 SURVIVE to the
# optimized module and feed the GEMM (vs qdq, whose f8 is stripped at pass 0029);
# (3) forward TFLOP/s vs a bf16 baseline at the same shape (f8 ~ 2x bf16 peak).
set -euo pipefail

K=4096; N=4096
DM=/tmp/manual_dump
rm -rf "$DM"

echo "=== MANUAL forward (d${K}) — pass dump ==="
python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
  --k "$K" --n "$N" --path manual --forward-only --steps 20 --warmup 5 \
  --no-print-hlo --xla-dump-dir "$DM"

FWD=$(grep -lE 'f8e4m3' "$DM"/*before_optimizations.txt | head -1)
PREFIX=$(basename "$FWD" | sed 's/\.before_optimizations\.txt//')
echo "forward module: $PREFIX ($(ls -1 "$DM/$PREFIX".*.txt | wc -l) pass files)"

echo "=== f8e4m3 count TRANSITIONS across passes (does f8 survive, unlike qdq?) ==="
prev=""
for f in $(ls -v "$DM/$PREFIX".*.txt); do
  c=$(grep -cE 'f8e4m3' "$f" || true)
  if [ "$c" != "$prev" ]; then printf "%3s  %s\n" "$c" "$(basename "$f")"; prev="$c"; fi
done

OPT=$(ls "$DM/$PREFIX"*after_optimizations.txt | head -1)
echo "=== f8 cuBLASLt custom call in optimized module? ==="
grep -F '__cublas$lt$matmul$f8' "$OPT" >/dev/null && echo "YES — \$f8 fired" || echo "NO \$f8"
echo "=== GEMM / matmul lines in optimized module (operand dtype tells f8 vs bf16) ==="
grep -nE 'custom-call|cublas|gemm_fusion|__triton|fusion\(.*kind=kCustom' "$OPT" | head -20 || true
echo "=== f8e4m3 occurrences in optimized module (0 => stripped like qdq) ==="
grep -cE 'f8e4m3' "$OPT" || true

echo "=== BF16 baseline forward (d${K}) for throughput reference ==="
python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
  --k "$K" --n "$N" --path bf16 --forward-only --steps 20 --warmup 5 --no-print-hlo
echo "=== DONE (compare fwd_tflops_per_s: manual vs bf16) ==="
