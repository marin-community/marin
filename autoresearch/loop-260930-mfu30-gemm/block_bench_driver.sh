#!/usr/bin/env bash
# Component benchmark driver: baseline vs candidate Block (and single-fusion variants) with the
# hero's XLA defaults (command buffers off), then without Triton GEMM fusion.
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
HERO="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
V="--variant fused=${D}/model_fused_projections.py --variant qkv=${D}/model_qkv_only.py --variant shared=${D}/model_shared_only.py"
XLA_FLAGS="${HERO}" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20 --check --profile /tmp/bb_default ${V}
XLA_FLAGS="${HERO} --xla_gpu_enable_triton_gemm=false" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20 --profile /tmp/bb_notriton
for i in 1 2; do
  XLA_FLAGS="${HERO}" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 30 ${V}
  XLA_FLAGS="${HERO} --xla_gpu_enable_triton_gemm=false" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 30
done
