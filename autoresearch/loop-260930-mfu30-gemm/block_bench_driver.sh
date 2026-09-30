#!/usr/bin/env bash
# Component benchmark driver: baseline vs candidate Block with the hero's XLA defaults
# (command buffers off), then without Triton GEMM fusion.
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
HERO="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
XLA_FLAGS="${HERO}" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20 --check --profile /tmp/bb_default
XLA_FLAGS="${HERO} --xla_gpu_enable_triton_gemm=false" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20 --profile /tmp/bb_notriton
XLA_FLAGS="${HERO}" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20
