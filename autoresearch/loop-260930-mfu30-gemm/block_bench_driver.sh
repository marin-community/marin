#!/usr/bin/env bash
# Component benchmark driver: baseline vs candidate Block, default flags and without Triton GEMM.
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20 --check --profile /tmp/bb_default
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20 --profile /tmp/bb_notriton
python ${D}/block_bench.py --baseline ${D}/model_baseline.py --iters 20
