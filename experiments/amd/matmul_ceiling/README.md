# AMD matmul ceiling

Measures the dense matmul throughput one AMD Instinct GPU reaches, as a ceiling for the MFU numbers in [#9812](https://github.com/marin-community/marin/issues/9812). Two measurements:

- Stas Bekman's [MAMF-finder](https://github.com/stas00/ml-engineering/blob/0359db89793c313e90e4f8a8bc8a2b1514ba00ae/compute/accelerator/benchmarks/mamf-finder.py) in PyTorch with TunableOp, which searches GEMM shapes for the maximum achievable matmul FLOPS (MAMF, short bursts) and the maximum sustainable rate (MSMF). His [published results](https://github.com/stas00/ml-engineering/blob/0359db89793c313e90e4f8a8bc8a2b1514ba00ae/compute/accelerator/README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table) cover MI300X, MI325X and MI355X.
- [`jax_matmul.py`](./jax_matmul.py), which times `jnp.matmul` through XLA over a shape grid, a shapes file such as [`snowball_shapes.txt`](./snowball_shapes.txt), or both. This is the rate that bounds our JAX training runs. [`summarize.py`](./summarize.py) prints the fastest shapes and the Snowball shapes from its result files.

Both venvs install AMD's `rocm-sdk-libraries==10.0.0` wheel, which provides hipBLASLt to PyTorch and JAX. On MI300X this setup reproduces Stas Bekman's bf16 row within 3.3% (MAMF 698 against 676 TFLOP/s). Results for MI350X and MI300X are in [#9812](https://github.com/marin-community/marin/issues/9812).

## Wall-clock and kernel-time rates

[`jax_matmul.sbatch`](./jax_matmul.sbatch) runs `jax_matmul.py` under `rocprofv3 --kernel-trace`, and [`kernel_trace.py`](./kernel_trace.py) then adds a second rate to each shape from the trace. Each result row has both:

- Wall clock (`median_tflops`): FLOPs divided by host time over back-to-back `jax.jit` calls. It includes host dispatch and the idle GPU time between calls. On MI350X, dispatching a trivial jitted op takes about 43 µs on the host, and the GPU idles about 12-15 µs between executions even when the host keeps ahead (job 454529). Shapes that take less than about 100 µs per call therefore read low, and the tracer's own host overhead lowers them further.
- Kernel time (`kernel_median_tflops`): FLOPs divided by the time the GPU spends in the kernels that start and end inside each timed window, which leaves out warmup, compilation and autotuning. `kernels` lists their names. `Cijk_*_UserArgs_*` kernels are hipBLASLt's Tensile-generated GEMMs; `gemm_fusion_dot*` kernels are GEMMs XLA generates itself with Triton. In bf16 on MI350X, XLA picks Triton for four of the Snowball k and v shapes (every shape with a dimension of 640, and 2560x1280x4096), and `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` switches them to hipBLASLt kernels that run 17-33% faster (jobs 454577 and 454580).

Compare MAMF and MSMF with the kernel-time rate. MAMF-finder records GPU events immediately before and after each `torch.mm`, so its rates also leave out the gaps between calls. It also clears a cache-sized buffer before every call, which `jax_matmul.py` does not.

## Running on the AMD HPC Fund cluster

Sync the checkout to the cluster and build the venvs once from the login node. They go in `$WORK/agents/<checkout folder>/`, as do results and logs:

```bash
cluster-sync
cluster "cd agents/<name> && bash experiments/amd/matmul_ceiling/setup_envs.sh"
```

Submit from the synced checkout root, and pin both jobs to one node with `-w <node>`: separate jobs otherwise land on different nodes. On the 28 bf16 shapes MAMF-finder shortlisted on MI350X ([`mamf_mi350x_bf16_shapes.txt`](./mamf_mi350x_bf16_shapes.txt)), JAX ran a median of 3% faster on k007-002 than on k007-004, ranging from 8% slower to 5% faster per shape (jobs 453412 and 453338). Both scripts use `--no-requeue`, because this cluster requeues failed batch jobs by default and a job that wedges a GPU would otherwise move on to the next node:

```bash
sha=$(git rev-parse HEAD)
cluster "cd agents/<name> && sbatch --export=ALL,MARIN_COMMIT=$sha -p mi3508x -w <node> -t 60 -J <name> \
  -o \$WORK/agents/<name>/logs/%x-%j.out experiments/amd/matmul_ceiling/mamf.sbatch --dtype bfloat16"
cluster "cd agents/<name> && sbatch --export=ALL,MARIN_COMMIT=$sha -p mi3508x -w <node> -t 60 -J <name> \
  -o \$WORK/agents/<name>/logs/%x-%j.out experiments/amd/matmul_ceiling/jax_matmul.sbatch \
  --dtype bfloat16 --shapes-file experiments/amd/matmul_ceiling/snowball_shapes.txt"
```

The JAX headline in [#9812](https://github.com/marin-community/marin/issues/9812), 1,192 TFLOP/s in bf16 on MI350X, is the best shape of a 125-shape grid with M, N and K each running from 4096 to 20480 in steps of 4096. `--m-range`, `--n-range` and `--k-range` take START STOP STEP with STOP inclusive; `--m`, `--n` and `--k` take explicit values instead:

```bash
cluster "cd agents/<name> && sbatch --export=ALL,MARIN_COMMIT=$sha -p mi3508x -w <node> -t 60 -J <name> \
  -o \$WORK/agents/<name>/logs/%x-%j.out experiments/amd/matmul_ceiling/jax_matmul.sbatch \
  --dtype bfloat16 --m-range 4096 20480 4096 --n-range 4096 20480 4096 --k-range 4096 20480 4096"
```

Keep XLA's GEMM autotuning on. With `--xla_gpu_autotune_level=0`, bf16 matmuls on MI350X ran at 15-17 TFLOP/s, about 1% of the autotuned rate (job 453316), so level 0 does not isolate hipBLASLt's default kernel choice from XLA's tuning.

MAMF-finder samples `amdsmi` GPU 0 for power and clock, but `amdsmi` and HIP number GPUs differently. On MI350X node k007-002, HIP device 0 is `amd-smi` GPU 3, so the MI350X logs show an idle GPU (about 308 W) and the boost-clock check is invalid there. Its TFLOP/s do not depend on telemetry. To read the busy GPU, run `/opt/rocm-7.2.0/bin/amd-smi metric --power --clock` alongside the job and match GPUs by PCI address. Under a sustained bf16 matmul, MI350X holds its 1,000 W limit with its compute dies at about 1,400 MHz, 64% of the 2.2 GHz used for the 2,307 TFLOP/s peak.
