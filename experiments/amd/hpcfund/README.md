# AMD HPC Fund cluster

Tooling for running Marin experiments on the AMD HPC Fund Slurm cluster
(hpcfund.amd.com). Experiments that are not tied to this cluster live elsewhere
under `experiments/amd/`.

## Partitions

| Partition | GPUs per node | ROCm target |
| --- | --- | --- |
| `mi3508x` | 8x MI350X | gfx950 |
| `mi3008x` | 8x MI300X | gfx942 |

## Submit a job

`run_gpu.sh` runs a Python entry point in the existing venv. Submit it from the
repository root; the job changes to `$SLURM_SUBMIT_DIR` before it starts:

```bash
mkdir -p logs
sbatch -N1 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_gpu.sh \
    experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --expert-axis 8
```

The wrapper unloads the cluster's `rocm` module, turns off XLA command buffers
and allocator preallocation, keeps XLA autotune results out of the JAX
compilation cache, and filters repeated ROCm log lines. Variables already set in
the environment take precedence.

The wrapper leaves `RAGGED_DOT_IMPL` unset, so Haliax picks the `ragged_dot`
implementation: on GPU it tries Triton and falls back to XLA. Set
`RAGGED_DOT_IMPL=triton` or `RAGGED_DOT_IMPL=xla` explicitly when
benchmarking, so results stay comparable across runs. XLA's `ragged_dot`
rejects bf16 on MI350X (gfx950). On MI300X (gfx942), `RAGGED_DOT_IMPL=xla` is
the fastest path: in per-call microbenchmarks on one MI300X, XLA's bf16 grouped
GEMM took 11-28% less time per forward+backward triple than the best Triton
configuration.

## Build the venv

The wrapper runs `uv run --no-sync`, so build the venv once on a login node:

```bash
uv sync
uv pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "rocm[libraries,device-gfx942,device-gfx950]==10.0.0" \
    "jax_rocm10_plugin==0.11.0+rocm10.0.0" "jax_rocm10_pjrt==0.11.0+rocm10.0.0"
uv pip install jax==0.11.0 jaxlib==0.11.0
```

## Known issues

- The cluster loads the `rocm/7.2.0` module by default. Its `LD_LIBRARY_PATH`
  takes precedence over the `RUNPATH` of the `jax_rocm10_plugin` libraries, so
  the plugin loads the HIP runtime, hipBLASLt, rocBLAS, RCCL and MIOpen from
  `/opt/rocm-7.2.0` instead of the ROCm 10 wheels. The wrapper runs
  `module unload rocm`. On the full June model on 8x MI350X (batch 64,
  `ring_dedup`, `save_moe`), the ROCm 10 libraries took 1.695 s per step
  against 1.600 s with the ROCm 7.2 libraries on the same node (median of steps
  2-19).
- XLA command buffers (HIP graphs) break training a few steps in, with NaN
  gradients or `ROCM_ERROR_ILLEGAL_ADDRESS`. The faulting kernel is the
  hipBLASLt grouped GEMM that XLA uses for `jax.lax.ragged_dot`
  (`RAGGED_DOT_IMPL=xla`). It faults when XLA replays a recorded graph without
  recording it again. With `RAGGED_DOT_IMPL=triton` the full June model runs
  with command buffers on, but on 8x MI350X it took 2.46 s per step against
  1.70 s with them off. The wrapper passes `--xla_gpu_enable_command_buffer=`
  to turn them off.
- JAX stores XLA's per-fusion autotune results inside the compilation cache
  directory. Reusing them across configurations produced executables 2-3x
  slower: `save_moe` at batch 64 took 4.95 s per step with reused results and
  1.83 s with fresh autotuning. The wrapper sets
  `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`.
- With a preallocated allocator pool, steps whose temp buffer is 94-97 GiB
  (batch 96 with reference attention, or `ring_dedup` with `save_moe` at batch
  64) can fail with out-of-memory around step 5. XLA returns the temp buffer to
  the pool while the step still runs, and a small buffer allocated in that
  window can land right after it. The next step's temp then no longer fits in
  the free space. The wrapper sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so the
  pool grows by region. Step time is unchanged.
- Processes can report a `double free` at exit, after the last step. It does
  not affect the results the run already printed or wrote.
