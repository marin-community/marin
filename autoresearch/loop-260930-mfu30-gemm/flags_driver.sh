#!/usr/bin/env bash
# Block benchmark: XLA flag combinations on the baseline Block, and the fused norm under the Triton-GEMM flag.
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
H="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
for i in 1 2 3; do
  for f in "" "--xla_gpu_enable_triton_gemm=false" "--xla_gpu_dot_merger_threshold_mb=448" "--xla_gpu_enable_triton_gemm=false --xla_gpu_dot_merger_threshold_mb=448"; do
    echo "=== flags [${f}] rep ${i}"
    XLA_FLAGS="${H} ${f}" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 30 --gated-norm pallas_gpu 2>&1 | grep -E "step |Error"
  done
done
