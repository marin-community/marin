#!/usr/bin/env bash
# Short-conv block benchmark with the hero's XLA defaults (command buffers off), profiled once.
set -euo pipefail
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
python autoresearch/loop-260930-mfu30/b/sconv_block_bench.py --iters 30 --reps 3 --profile /tmp/sconv_block "$@"
