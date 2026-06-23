#!/usr/bin/env bash
# S2 GFP8-012 H100 batch: validate the forward_precision fix + close the audit's
# open questions in one job. All arms d4096 / TN unless noted. For each arm we
# read result_json (TFLOP/s), the forward $f8 count, f8e4m3 survival, the count of
# bf16->f8 `convert` ops feeding the matmul (partial-fusion tell), and the cuBLAS
# custom-call line.
#
#   A. Fix validation (real Fp8DotGeneralOp): does forward_precision=HIGHEST make
#      the forward fire $f8, and how fast vs the same-job bf16 forward?
#   B. Mechanism A/B (pure bf16, no QDQ): does HIGHEST change the bf16 lowering?
#      changed => H1 (bf16 rewriter preempts at DEFAULT); identical => H2 (f8
#      rewriter independently declines DEFAULT).
#   C. Partial-fusion check: is the qdq/HIGHEST forward one fused $f8 kernel or a
#      convert prologue + gemm? (the 1.38x / 31%-of-peak anomaly). Longer warmup
#      settles cuBLASLt autotune.
#   D. PET de-confound: does materialized-f8 (manual) still fire $f8 at DEFAULT
#      without preferred_element_type=f32?
set -uo pipefail

K=4096; N=4096
BENCH="python -u lib/levanter/scripts/bench/bench_dense_fp8.py --k $K --n $N --layout tn"

run() {  # run <label> <extra-bench-args...>
  local label=$1; shift
  local out="/tmp/s2b_${label}.txt"
  echo "================================================================"
  echo "### ${label}:  $*"
  echo "================================================================"
  if ! $BENCH "$@" > "$out" 2>&1; then
    echo "  FAILED (tail):"; tail -25 "$out"; return
  fi
  grep -F 'result_json' "$out" | head -1
  printf '  forward $f8 matmul count : %s\n' "$(grep -cF '__cublas$lt$matmul$f8' "$out")"
  printf '  f8e4m3 in compiled HLO   : %s\n' "$(grep -cF 'f8e4m3' "$out")"
  printf '  bf16->f8 convert ops     : %s\n' "$(grep -ciE 'convert.*f8e4m3|f8e4m3.*convert' "$out")"
  printf '  cuBLAS custom-call line  : %s\n' "$(grep -oE 'custom_call_target="__cublas[^"]*"' "$out" | head -1)"
}

echo "######## A. FIX VALIDATION (real Fp8DotGeneralOp) ########"
run A1_qdq_default_fwd   --path qdq  --precision default --forward-only
run A2_qdq_highest_fwd   --path qdq  --precision highest --forward-only
run A3_bf16_fwd          --path bf16 --forward-only
run A4_qdq_highest_fwdbwd --path qdq --precision highest
run A5_qdq_default_fwdbwd --path qdq --precision default

echo "######## B. MECHANISM A/B (pure bf16, no QDQ) ########"
run B1_bf16_default_fwd  --path bf16 --precision default --forward-only
run B2_bf16_highest_fwd  --path bf16 --precision highest --forward-only

echo "######## C. PARTIAL-FUSION CHECK (settle autotune) ########"
run C1_qdq_highest_long  --path qdq --precision highest --forward-only --warmup 50 --steps 100
echo "--- C: cuBLAS \$f8 custom-call + its operand defs (convert prologue?) ---"
grep -nE '__cublas\$lt\$matmul\$f8|convert.*f8e4m3' /tmp/s2b_C1_qdq_highest_long.txt | head -20

echo "######## D. PET DE-CONFOUND (materialized f8) ########"
run D1_manual_pet_f32    --path manual --forward-only
run D2_manual_pet_none   --path manual --forward-only --manual-no-output-f32

echo "######## SUMMARY ########"
for f in /tmp/s2b_*.txt; do
  printf '%-26s ' "$(basename "$f" .txt)"
  grep -F 'result_json' "$f" | head -1 | grep -oE '"(fwd|bwd)_tflops_per_s": [0-9.]+' | tr '\n' ' '
  printf '| $f8=%s\n' "$(grep -cF '__cublas$lt$matmul$f8' "$f")"
done
echo "=== DONE ==="
