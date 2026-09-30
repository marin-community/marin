#!/usr/bin/env bash
set -uo pipefail
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
python autoresearch/loop-260930-mfu30-gemm/grn_triton.py
