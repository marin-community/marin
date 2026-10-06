# amd-matmul-ceiling

Measures the dense matmul throughput one AMD Instinct GPU actually reaches, as a realistic ceiling for the MFU numbers in [#9812](https://github.com/marin-community/marin/issues/9812). Two measurements on the same node:

- [MAMF-finder](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py) in PyTorch with TunableOp, which searches GEMM shapes for the best achievable FLOP/s.
- [`jax_matmul.py`](./jax_matmul.py), which times `jnp.matmul` through XLA over a shape grid, a shapes file such as [`snowball_shapes.txt`](./snowball_shapes.txt), or both. This is the number that bounds our JAX training runs.

Both environments use ROCm 10.0.0 user-space libraries, so PyTorch and JAX call the same hipBLASLt.

## Running on the AMD HPC Fund cluster

Sync the checkout to the cluster and build the venvs once from the login node. They go in `$WORK/agents/<checkout folder>/`, as do results and logs:

```bash
cluster-sync
cluster "cd agents/<name> && bash experiments/amd_matmul_ceiling/setup_envs.sh"
```

Submit from the synced checkout root. Both scripts use `--no-requeue`, because this cluster requeues failed batch jobs by default and a job that wedges a GPU would otherwise move on to the next node:

```bash
sha=$(git rev-parse HEAD)
cluster "cd agents/<name> && sbatch --export=ALL,MARIN_COMMIT=$sha -p mi3508x -t 60 -J <name> \
  -o \$WORK/agents/<name>/logs/%x-%j.out experiments/amd_matmul_ceiling/mamf.sbatch --dtype bfloat16"
cluster "cd agents/<name> && sbatch --export=ALL,MARIN_COMMIT=$sha -p mi3508x -t 60 -J <name> \
  -o \$WORK/agents/<name>/logs/%x-%j.out experiments/amd_matmul_ceiling/jax_matmul.sbatch \
  --dtype bfloat16 --shapes-file experiments/amd_matmul_ceiling/snowball_shapes.txt"
```

Keep XLA's GEMM autotuning on. With `--xla_gpu_autotune_level=0`, bf16 matmuls on MI350X ran at 15-17 TFLOP/s, about 1% of the autotuned rate (job 453316), so level 0 does not isolate hipBLASLt's default kernel choice from XLA's tuning.
