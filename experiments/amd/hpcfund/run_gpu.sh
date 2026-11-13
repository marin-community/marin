#!/bin/bash
# Slurm wrapper for Python entry points on the AMD HPC Fund cluster (hpcfund.amd.com), whose GPU
# partitions include mi3508x (8x MI350X) and mi3008x (8x MI300X). Submit from the repository root;
# the job cds to $SLURM_SUBMIT_DIR and runs the given script in the existing venv without syncing it:
#   sbatch -N1 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_gpu.sh \
#       experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --expert-axis 8
# Pass --overlap-collectives before the script to add the multi-node collective-overlap XLA flags.
# README.md in this directory has the venv recipe and known issues.
cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONUNBUFFERED=1

# Command buffers (HIP graphs) corrupt memory on this ROCm stack: NaN gradients and segfaults after a few steps.
# A later --xla_gpu_enable_command_buffer=... in the caller's XLA_FLAGS overrides this.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= $XLA_FLAGS"

# Across nodes, data parallelism all-reduces each layer's expert gradients over the NICs inside the backward
# layer scan. These flags let XLA run up to 4 collectives at once, move each all-reduce into the next scan
# iteration and start collectives early, so most of that traffic overlaps compute. They are opt-in because more
# than one collective in flight has silently corrupted training on GB200, and parity has not been checked from a
# trained checkpoint here; the caller's XLA_FLAGS still override any of them. README.md has the measurements,
# the per-flag ablation and that caveat.
if [ "${1:-}" = "--overlap-collectives" ]; then
    shift
    OVERLAP_XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_experimental_parallel_collective_overlap_limit=4 --xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_experimental_collective_start_as_early_as_possible=true"
    export XLA_FLAGS="$OVERLAP_XLA_FLAGS $XLA_FLAGS"
fi

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

# Drop ROCm runtime log lines that repeat on every step. Exit with Python's status: grep exits 1
# when it prints nothing.
uv run --no-sync python "$@" 2>&1 | grep --line-buffered -v -E "rocm_pcie_bandwidth|rocm_executor"
exit "${PIPESTATUS[0]}"
