#!/usr/bin/env bash
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
python ${D}/grn_kernels.py --configs "64,128,256,4,3,4,2;128,64,256,8,3,8,2;128,128,256,8,3,8,2;128,64,128,8,3,8,2;128,64,512,8,3,8,2;256,64,256,8,2,8,2;256,32,256,8,3,8,2;128,64,256,4,4,4,2;128,32,256,8,4,8,2;256,64,128,8,3,4,2;128,64,256,8,3,4,2;128,64,256,8,3,8,3;64,64,128,4,3,4,2;256,64,512,8,2,8,2"
