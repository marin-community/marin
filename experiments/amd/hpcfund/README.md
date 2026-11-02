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

The wrapper turns off XLA command buffers and allocator preallocation, keeps
XLA autotune results out of the JAX compilation cache, and filters repeated
ROCm log lines. Variables already set in the environment take precedence.

The wrapper leaves `RAGGED_DOT_IMPL` unset, so Haliax picks the `ragged_dot`
implementation: on GPU it tries Triton and falls back to XLA. Set
`RAGGED_DOT_IMPL=triton` or `RAGGED_DOT_IMPL=xla` explicitly when
benchmarking, so results stay comparable across runs. XLA's `ragged_dot`
rejects bf16 on MI350X (gfx950). On MI300X (gfx942), `RAGGED_DOT_IMPL=xla` is
the fastest path: in per-call microbenchmarks on one MI300X, XLA's bf16 grouped
GEMM took 11-28% less time per forward+backward triple than the best Triton
configuration.

## Multi-node runs

Pass `--overlap-collectives` to the wrapper, before the script, to add five XLA flags that overlap collectives with compute: the latency-hiding scheduler, up to 4 collectives in flight (XLA's default is 1), pipelined all-reduce (each loop iteration's all-reduce moves into the next iteration), while-loop double buffering, and starting each collective as early as possible. With data parallelism across nodes, each layer's expert-weight gradients are all-reduced over the network inside the backward layer scan, and without these flags that all-reduce runs with nothing to overlap it. Set any of the flags in `XLA_FLAGS` to override it.

```bash
sbatch -N2 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_gpu.sh --overlap-collectives \
    experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --gpus-per-node 8 --expert-axis 8 --replica-axis 2
```

The flags are off by default because of the correctness caveat below.

June Snowball (`synthetic_benchmark.py --size full`), EP8 inside each node and data parallelism across nodes (`--expert-axis 8 --replica-axis 2`), 8 sequences of 4,096 tokens per GPU, `--moe-impl ring_filtered --pooled-transport-capacity-factor 1.1 --attention gpu_pallas_triton_flash --remat-mode save_moe`, step time as the median of steps 2-19:

| Hardware | Nodes | Without the flags | With the flags | Change |
|---|---|---|---|---|
| MI350X (`mi3508x`, this wrapper and venv) | 2 | 1.975-1.982 s | 1.715-1.717 s | -13% |
| MI350X, all nine flags of the ablation below | 1 | 1.529-1.534 s | 1.526-1.529 s | none |
| MI355X (TheRock 7.14 container, JAX 0.11.0) | 2 | 1.955-1.971 s | 1.709-1.736 s | -12% |

In both two-node MI350X jobs, the second run without the flags was slower on every step (2.134 s and 3.026 s); the cause is unknown, and the table uses the first run of each job. Peak memory rises by at most 1 GiB.

Each flag's share, from two MI355X nodes: removing one flag from the full set of nine changes the step as follows. With all nine flags, five runs took 1.707-1.740 s (median 1.725 s); each removal is a single run.

| Flag removed | Step | Change |
|---|---|---|
| `--xla_gpu_enable_latency_hiding_scheduler` | 2.065 s | +20% |
| `--xla_gpu_experimental_parallel_collective_overlap_limit=4` | 1.964 s | +14% |
| `--xla_gpu_experimental_collective_start_as_early_as_possible` | 1.793 s | +3.9% |
| `--xla_gpu_enable_pipelined_all_reduce` | 1.768 s | +2.5% |
| `--xla_gpu_enable_while_loop_double_buffering` | 1.752 s | +1.6% |
| `--xla_gpu_enable_pipelined_all_gather` | 1.729 s | none |
| `--xla_gpu_enable_pipelined_reduce_scatter` | 1.719 s | none |
| 8 GiB all-gather and reduce-scatter combine thresholds | 1.715 s | none |

`--overlap-collectives` keeps the first five. With the overlap limit at 1, the other flags give no gain (1.964 s against 1.955-1.971 s without flags).

On GB200, more than one collective in flight silently corrupted training with Grug's ragged all-to-all transport and the layer-carry offload, so the NVIDIA EP hero forces the limit to 1 (`experiments/grug/moe_hero_ep/train.py`). Here, two runs with the five flags and two without differ in per-step loss by at most 2.7e-4 and in capacity-drop rate by at most 0.4 points; the two runs without the flags already differ from each other by 1.5e-4 and 0.3 points. These runs use synthetic tokens and an untrained router, which detect such races poorly; check loss and drop-rate parity from a trained checkpoint before a long run, or set `--xla_gpu_experimental_parallel_collective_overlap_limit=1` in `XLA_FLAGS` to rule it out at the cost of the gain.

The flags were measured only with this layout on two nodes, not with cross-node FSDP or EP16.

With this venv, runs with the flags sometimes crash as the processes shut down after the last step (`double free or corruption` while `jax.distributed` disconnects, or a segfault): 4 of 17 runs with some of these flags on k007-002 and k007-004, against 0 of 6 runs without them, and one more run with the flags segfaulted about 4 minutes into compilation. Output up to the crash is complete, but Slurm records the step as failed. Twenty-two flagged runs in the MI355X container exited cleanly.

## Build the venv

The wrapper runs `uv run --no-sync`, so build the venv once on a login node:

```bash
uv sync
uv pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "rocm[libraries,device-gfx942,device-gfx950]==10.0.0" \
    "jax_rocm10_plugin==0.11.0+rocm10.0.0" "jax_rocm10_pjrt==0.11.0+rocm10.0.0"
uv pip install jax==0.11.0 jaxlib==0.11.0
```

## Run in AMD's TheRock container

`run_jax_aiter.sh` runs a command inside AMD's TheRock 7.14 JAX image (`ghcr.io/rocm/jax-dev-ubu24.therock-7.14`) under Apptainer, with the same XLA settings as `run_gpu.sh`. JAX-AITER supports only this stack, and the Vultr MI355X nodes use it for all runs. The comments at the top of the script cover pulling the image and building its venv with `setup_jax_aiter.sh`.

The image ships no RDMA libraries, so across nodes RCCL falls back to TCP sockets on one Ethernet port. When the host has `libibverbs`, the wrapper mounts it, `librdmacm`, the host's NIC providers and their config, and the libraries they link into the container. On two Vultr MI355X nodes, a 16-GPU `ragged_all_to_all` that moves 1.26 GB per call took 26.7 ms over sockets and 5.87 ms over RDMA. Check that a multi-node job logs `NET/IB` rather than `NET/Socket` with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`.

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
  (batch 96 with reference attention, or `ring_filtered` with `save_moe` at batch
  64) can fail with out-of-memory around step 5. XLA returns the temp buffer to
  the pool while the step still runs, and a small buffer allocated in that
  window can land right after it. The next step's temp then no longer fits in
  the free space. The wrapper sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so the
  pool grows by region. Step time is unchanged.
- Processes can report a `double free` at exit, after the last step. It does
  not affect the results the run already printed or wrote.
