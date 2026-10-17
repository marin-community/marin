#!/bin/bash
# Slurm wrapper for Python entry points on the AMD HPC Fund cluster (hpcfund.amd.com), whose GPU
# partitions include mi3508x (8x MI350X) and mi3008x (8x MI300X). Submit from the repository root;
# the job cds to $SLURM_SUBMIT_DIR and runs the given script in the existing venv without syncing it:
#   sbatch -N1 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_gpu.sh \
#       experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --expert-axis 8
# README.md in this directory has the venv recipe and known issues.
cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONUNBUFFERED=1

# Command buffers (HIP graphs) corrupt memory on this ROCm stack: NaN gradients and segfaults after a few steps.
# A later --xla_gpu_enable_command_buffer=... in the caller's XLA_FLAGS overrides this.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= $XLA_FLAGS"

# JAX stores XLA's per-fusion autotune results inside --compilation-cache-dir. Reusing them across
# configurations produced executables 2-3x slower (save_moe at batch 64 took 4.95 s per step against
# 1.83 s with fresh autotuning) and a false batch-96 regression. `none` still caches compiled
# executables but autotunes each new compilation.
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=${JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES:-none}

# A preallocated BFC pool is one fixed range. XLA frees the step's large temp buffer while the step still
# runs, and a small buffer allocated then can land right after it, leaving the next step's temp no hole
# big enough (OOM around step 5 with 94-97 GiB temps). Without preallocation the pool grows by region, at the
# same step time.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

# The cluster's default module puts /opt/rocm-7.2.0 on LD_LIBRARY_PATH, so jobs load its HIP runtime and RCCL
# rather than the wheel's. With jax 0.11.1, RCCL then aborts at the first collective unless scratch reclaim is
# off. Removing /opt/rocm from the paths avoids that, but makes lax.top_k on rows under 1,024 entries 2.4-4.2x
# slower on MI350X.
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}

# Drop ROCm runtime log lines that repeat on every step. Exit with Python's status: grep exits 1
# when it prints nothing.
uv run --no-sync python "$@" 2>&1 | grep --line-buffered -v -E "rocm_pcie_bandwidth|rocm_executor"
exit "${PIPESTATUS[0]}"
