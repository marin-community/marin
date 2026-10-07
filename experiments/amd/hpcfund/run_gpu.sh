#!/bin/bash
# Slurm wrapper for Python entry points on the AMD HPC Fund cluster (hpcfund.amd.com), whose GPU
# partitions include mi3508x (8x MI350X) and mi3008x (8x MI300X). Submit from the repository root;
# the job cds to $SLURM_SUBMIT_DIR and runs the given script in the existing venv without syncing it:
#   sbatch -N1 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_gpu.sh \
#       experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --expert-axis 8
# README.md in this directory has the venv recipe and known issues.
set -o pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1

# On GPU, Haliax's ragged_dot otherwise picks its Triton kernel, which targets CUDA.
export RAGGED_DOT_IMPL=${RAGGED_DOT_IMPL:-xla} PYTHONUNBUFFERED=1

# Command buffers (HIP graphs) corrupt memory on this ROCm stack: NaN gradients and segfaults after a few steps.
# A later --xla_gpu_enable_command_buffer=... in the caller's XLA_FLAGS overrides this.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= $XLA_FLAGS"

# JAX stores XLA's per-fusion autotune results inside --compilation-cache-dir. Reusing them across
# configurations produced executables 2-3x slower (save_moe at batch 64 took 4.95 s per step against
# 1.83 s with fresh autotuning) and a false batch-96 regression. `none` still caches compiled
# executables but autotunes each new compilation.
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=${JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES:-none}

# Drop ROCm runtime log lines that repeat on every step.
uv run --no-sync python "$@" 2>&1 | grep --line-buffered -v -E "rocm_pcie_bandwidth|rocm_executor"
