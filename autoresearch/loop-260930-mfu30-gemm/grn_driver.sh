#!/usr/bin/env bash
# Fused gated_rms_norm: GPU correctness + kernel timing, then the block benchmark with the fused norm.
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
HERO="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
XLA_FLAGS="${HERO}" python ${D}/grn_check.py --iters 20
for b in "64,128,256,4,3,4,2" "64,64,256,4,4,4,2" "128,64,256,8,3,8,2" "64,128,512,4,3,8,2" "32,128,256,4,3,4,2"; do
  echo "=== block ${b}"
  XLA_FLAGS="${HERO}" python ${D}/grn_check.py --iters 20 --block "${b}" 2>&1 | grep -E "fwd .* ms|Error|error" | head -3
done
XLA_FLAGS="${HERO}" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20 --gated-norm pallas_gpu --check --profile /tmp/bb_grn
for i in 1 2; do
  XLA_FLAGS="${HERO}" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 30 --gated-norm pallas_gpu
  XLA_FLAGS="${HERO} --xla_gpu_enable_triton_gemm=false" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 30 --gated-norm pallas_gpu
done
