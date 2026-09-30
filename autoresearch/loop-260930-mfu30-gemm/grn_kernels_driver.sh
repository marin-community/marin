#!/usr/bin/env bash
set -uo pipefail
D=autoresearch/loop-260930-mfu30-gemm
H="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
XLA_FLAGS="${H}" python ${D}/grn_kernels.py --configs "128,32,128,4,4,8,3;128,32,128,4,4,8,3,1;64,32,128,4,4,4,3,1;64,32,64,4,4,4,4,1;128,32,64,4,4,8,4,1;64,32,256,4,4,8,2,1;32,32,128,4,4,4,3,1;128,32,256,4,4,8,2,1;128,16,128,4,6,8,3,1;64,32,128,4,4,8,3,1"
XLA_FLAGS="${H} --xla_gpu_enable_triton_gemm=false" python ${D}/grn_kernels.py --configs "128,32,128,4,4,8,3"
