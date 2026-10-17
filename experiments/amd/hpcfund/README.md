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

The wrapper turns off XLA command buffers, allocator preallocation and HSA scratch reclaim, keeps XLA autotune results out of the JAX compilation cache, and filters repeated ROCm log lines. Variables already set in the environment take precedence.

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
    "rocm[libraries,device-gfx942,device-gfx950]==10.1.0" \
    "jax_rocm10_plugin==0.11.1+rocm10.1.0" "jax_rocm10_pjrt==0.11.1+rocm10.1.0"
uv pip install jax==0.11.1 jaxlib==0.11.1 jax-triton==0.3.1 triton==3.6.0
```

jaxlib 0.11.0 does not load `jax_rocm10_plugin`'s kernel modules, so kernels written with `jax_triton`, such as Grug's routing top-k, need 0.11.1. `jax-triton` and `triton` match Levanter's GPU pins. Do not add `jax-triton` to a 0.11.0 venv: importing it there raises `ImportError`, and `levanter.grug.grug_moe` then fails to import.

## Known issues

- XLA command buffers (HIP graphs) corrupt memory on this ROCm stack. Runs
  produced NaN gradients and segfaults after a few steps. The wrapper passes
  `--xla_gpu_enable_command_buffer=` to turn them off.
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
- The cluster's default module puts `/opt/rocm-7.2.0` on `LD_LIBRARY_PATH`, so jobs load its HIP runtime and RCCL rather than the wheel's. With jax 0.11.1, RCCL then aborts at the first collective with `HSA_NO_SCRATCH_RECLAIM=1 must be set`; the wrapper sets it. Removing `/opt/rocm` from the paths also avoids the abort, but makes `jax.lax.top_k` on rows under 1,024 entries 2.4-4.2x slower on MI350X.
