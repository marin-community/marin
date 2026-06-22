#!/usr/bin/env bash
# S2 diagnostic: dump every XLA pass for the qdq forward and locate the pass
# where the f8e4m3 converts vanish (i.e. why the forward falls back to bf16).
# XLA_FLAGS must carry --xla_dump_hlo_pass_re=.* ; the bench appends --xla_dump_to.
set -euo pipefail

D=/tmp/passdump
rm -rf "$D"

python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
  --k 4096 --n 4096 --path qdq --forward-only --steps 1 --warmup 0 \
  --no-print-hlo --xla-dump-dir "$D"

echo "=== locate forward module (f8 present before optimizations) ==="
FWD=$(grep -lE 'f8e4m3' "$D"/*before_optimizations.txt | head -1)
PREFIX=$(basename "$FWD" | sed 's/\.before_optimizations\.txt//')
echo "forward module: $PREFIX"
NFILES=$(ls -1 "$D/$PREFIX".*.txt | wc -l)
echo "per-pass dump files for this module: $NFILES"

echo "=== f8e4m3 count TRANSITIONS across passes (pass order) ==="
prev=""
for f in $(ls -v "$D/$PREFIX".*.txt); do
  c=$(grep -cE 'f8e4m3' "$f" || true)
  if [ "$c" != "$prev" ]; then
    printf "%2s  %s\n" "$c" "$(basename "$f")"
    prev="$c"
  fi
done

echo "=== passes where the f8 cuBLASLt custom call appears ==="
grep -lF '__cublas$lt$matmul$f8' "$D/$PREFIX".*.txt 2>/dev/null | xargs -n1 basename 2>/dev/null || echo NONE
echo "=== passes where a plain bf16 cuBLASLt matmul appears ==="
grep -lF '__cublas$lt$matmul"' "$D/$PREFIX".*.txt 2>/dev/null | xargs -n1 basename 2>/dev/null || echo NONE
echo "=== DONE ==="
