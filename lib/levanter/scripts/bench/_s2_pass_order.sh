#!/usr/bin/env bash
# S2: pin the XLA pass ordering — does the f8 GemmRewriter run BEFORE or AFTER
# simplify-fp-conversions? Full per-pass HLO dump of the qdq path (fwd+bwd) at
# d4096/NN (the case where the forward is stripped but the backward fires $f8).
# For each f8-bearing module we print the ordered pass schedule with the key
# events (f8e4m3 count change, $f8 appearance) and the gemm/rewriter/cublas/
# float-normalization passes, then a verdict: the pass index where $f8 is created
# vs the index of simplify-fp-conversions. $f8 AFTER the strip => Story B
# (rewriter never sees the forward's f8); BEFORE => Story A (rewriter declined).
set -euo pipefail

K=4096; N=4096
D=/tmp/pass_order
rm -rf "$D"

XLA_FLAGS="--xla_dump_to=$D --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*" \
  python -u lib/levanter/scripts/bench/bench_dense_fp8.py \
    --k "$K" --n "$N" --path qdq --layout nn --steps 1 --warmup 0 --no-print-hlo

ordered_files() { ls -v "$D/$1".[0-9]*.txt; }  # numbered per-pass files, in pass order

analyze() {
  local prefix="$1" label="$2"
  echo "===================== ${label}"
  echo "                      ${prefix} ====================="
  local i=0 prevc="" prevf=""
  for f in $(ordered_files "$prefix"); do
    i=$((i + 1))
    local b c hf mark
    b=$(basename "$f" | sed "s/^${prefix}\.//; s/\.txt\$//")
    c=$(grep -cE 'f8e4m3' "$f" 2>/dev/null || true)
    hf="-"; grep -qF '__cublas$lt$matmul$f8' "$f" 2>/dev/null && hf="F8"
    mark=""; echo "$b" | grep -qiE 'gemm.?rewrit|cublas|simplify-fp|float.?norm' && mark="  <=="
    if [ "$c" != "$prevc" ] || [ "$hf" != "$prevf" ] || [ -n "$mark" ]; then
      printf '  [%3d] f8e4m3=%-3s $f8=%-3s  %s%s\n' "$i" "$c" "$hf" "$b" "$mark"
    fi
    prevc="$c"; prevf="$hf"
  done
}

verdict() {
  local prefix="$1"
  local i=0 sfc_i="" sfc_n="" f8_i="" f8_n=""
  for f in $(ordered_files "$prefix"); do
    i=$((i + 1))
    local b; b=$(basename "$f")
    if [ -z "$sfc_i" ] && echo "$b" | grep -qi 'simplify-fp'; then sfc_i=$i; sfc_n="$b"; fi
    if [ -z "$f8_i" ] && grep -qF '__cublas$lt$matmul$f8' "$f" 2>/dev/null; then f8_i=$i; f8_n="$b"; fi
  done
  echo "  VERDICT:"
  echo "    simplify-fp-conversions first at index: ${sfc_i:-none}  ${sfc_n:-}"
  echo "    \$f8 custom-call first at index:         ${f8_i:-none}  ${f8_n:-}"
  if [ -n "$f8_i" ] && [ -n "$sfc_i" ]; then
    if [ "$f8_i" -gt "$sfc_i" ]; then
      echo "    => \$f8 created AFTER simplify-fp-conversions  (Story B: rewriter never sees forward f8)"
    else
      echo "    => \$f8 created BEFORE simplify-fp-conversions (Story A: rewriter ran before the strip)"
    fi
  fi
}

for pre in $(ls "$D"/*before_optimizations.txt 2>/dev/null | xargs -n1 basename | sed 's/\.before_optimizations\.txt$//' | sort -u); do
  grep -qE 'f8e4m3' "$D/${pre}.before_optimizations.txt" 2>/dev/null || continue
  if ordered_files "$pre" | xargs grep -lF '__cublas$lt$matmul$f8' >/dev/null 2>&1; then
    analyze "$pre" "MODULE WITH \$f8 (backward-bearing)"
    verdict "$pre"
  else
    analyze "$pre" "MODULE WITHOUT \$f8 (forward, stripped)"
    verdict "$pre"
  fi
done
echo "=== DONE ==="
