# amd-matmul-ceiling

Measures the dense matmul throughput one AMD Instinct GPU reaches, as a ceiling for the MFU numbers in [#9812](https://github.com/marin-community/marin/issues/9812). Two measurements on the same node:

- Stas Bekman's [MAMF-finder](https://github.com/stas00/ml-engineering/blob/0359db89793c313e90e4f8a8bc8a2b1514ba00ae/compute/accelerator/benchmarks/mamf-finder.py) in PyTorch with TunableOp, which searches GEMM shapes for the maximum achievable matmul FLOPS (MAMF, short bursts) and the maximum sustainable rate (MSMF). His [published results](https://github.com/stas00/ml-engineering/blob/0359db89793c313e90e4f8a8bc8a2b1514ba00ae/compute/accelerator/README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table) cover MI300X, MI325X and MI355X.
- [`jax_matmul.py`](./jax_matmul.py), which times `jnp.matmul` through XLA over a shape grid, a shapes file such as [`snowball_shapes.txt`](./snowball_shapes.txt), or both. This is the rate that bounds our JAX training runs. [`summarize.py`](./summarize.py) prints the fastest shapes and the Snowball shapes from its result files.

Both venvs install AMD's `rocm-sdk-libraries==10.0.0` wheel, which provides hipBLASLt to PyTorch and JAX. On MI300X this setup reproduces Stas Bekman's bf16 row within 3.3% (MAMF 698 against 676 TFLOP/s). Results for MI350X and MI300X are in [#9812](https://github.com/marin-community/marin/issues/9812).

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

On MI350X, `amdsmi` reports 38 MHz and 308 W under load, so MAMF-finder's boost-clock check does not work there; its throughput numbers do not depend on telemetry.
