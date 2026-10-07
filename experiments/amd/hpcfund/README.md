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

The wrapper turns off XLA command buffers, keeps XLA autotune results out of
the JAX compilation cache, and filters repeated ROCm log lines. Variables
already set in the environment take precedence.

The wrapper leaves `RAGGED_DOT_IMPL` unset, so Haliax picks the `ragged_dot`
implementation: on GPU it tries Triton and falls back to XLA. Set
`RAGGED_DOT_IMPL=triton` or `RAGGED_DOT_IMPL=xla` explicitly when
benchmarking, so results stay comparable across runs. XLA's `ragged_dot`
rejects bf16 on MI350X (gfx950).

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

- XLA command buffers (HIP graphs) corrupt memory on this ROCm stack. Runs
  produced NaN gradients and segfaults after a few steps. The wrapper passes
  `--xla_gpu_enable_command_buffer=` to turn them off.
- JAX stores XLA's per-fusion autotune results inside the compilation cache
  directory. Reusing them across configurations produced executables 2-3x
  slower: `save_moe` at batch 64 took 4.95 s per step with reused results and
  1.83 s with fresh autotuning. The wrapper sets
  `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`.
- Processes can report a `double free` at exit, after the last step. It does
  not affect the results the run already printed or wrote.
