#!/bin/bash
# Slurm wrapper for the June synthetic benchmark on the AMD HPC Fund cluster. Submit from the repository root:
#   sbatch -N1 -p mi3508x -t 45 -o logs/%x-%j.out experiments/june_tpu_67b_a2b/moe/run_gpu.sh \
#       experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --expert-axis 8
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
export RAGGED_DOT_IMPL=${RAGGED_DOT_IMPL:-xla} PYTHONUNBUFFERED=1
# Command buffers (HIP graphs) corrupt memory on this ROCm stack: nan gradients and segfaults after a few steps.
# A later --xla_gpu_enable_command_buffer=... in the caller's XLA_FLAGS overrides this.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= $XLA_FLAGS"
uv run --no-sync python "$@" 2>&1 | grep --line-buffered -v -E "rocm_pcie_bandwidth|rocm_executor"
