#!/usr/bin/env bash
# Single-node GEMM microbenchmark driver (H-C1). Runs from the repo root inside an Iris GPU job.
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
C=${D}/hero_gemms.json
OUT=/tmp/gemm_bench
mkdir -p "${OUT}"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>&1 | head -120 || true
MAIN="b1_m65536_n3072_k6144_MK_KN_MN,b1_m65536_n6144_k3072_MK_NK_MN,b1_m65536_n6144_k6144_MK_KN_MN,b1_m3072_n6144_k65536_KM_KN_NM,b1_m65536_n1536_k6144_MK_KN_MN,fused_qkv_fwd,fused_gateup2_fwd,ref_8192cube"
python ${D}/gemm_bench.py --configs ${C} --out ${OUT}/default.json --profile
XLA_FLAGS="--xla_gpu_dump_autotune_logs_to=${OUT}/autotune_logs.txt" python ${D}/gemm_bench.py --configs ${C} --out ${OUT}/default_logs.json --only "${MAIN}" --sustain 2
python ${D}/gemm_bench.py --configs ${C} --out ${OUT}/dev4.json --devices 4 --only "${MAIN}" --sustain 8
python ${D}/gemm_bench.py --configs ${C} --out ${OUT}/long.json --only b1_m65536_n6144_k6144_MK_KN_MN --sustain 30
XLA_FLAGS="--xla_gpu_enable_cublaslt=false" python ${D}/gemm_bench.py --configs ${C} --out ${OUT}/nolt.json --sustain 4
XLA_FLAGS="--xla_gpu_autotune_level=0" python ${D}/gemm_bench.py --configs ${C} --out ${OUT}/at0.json --sustain 4
XLA_FLAGS="--xla_gpu_cublas_fallback=false" python ${D}/gemm_bench.py --configs ${C} --out ${OUT}/triton.json --only "${MAIN}" --sustain 4
for f in "${OUT}"/*.json; do
  echo "=== RESULT $(basename "${f}")"
  python -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "${f}"
done
echo "=== AUTOTUNE LOGS"
head -c 200000 "${OUT}/autotune_logs.txt" 2>/dev/null || echo "(none)"
