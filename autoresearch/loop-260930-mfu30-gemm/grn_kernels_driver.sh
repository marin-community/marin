#!/usr/bin/env bash
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
python ${D}/grn_kernels.py --configs "128,64,128,8,3,8,2;128,32,128,8,4,8,2;128,128,128,8,3,8,2;64,64,128,4,3,4,2;64,128,128,4,3,4,2;128,64,64,8,3,4,2;128,64,256,8,3,8,2;64,64,256,4,3,8,2;128,32,128,4,4,8,3;128,64,128,8,4,16,2;64,32,64,4,4,4,2"
