# Minimal reproducer for the silent collective wedge (#7344)

A single-file, `jax`+`numpy`-only program that reproduces the silent collective wedge
tracked in marin issue #7344 on GB200 NVL72 (cluster `cw-us-east-08a`).

The point of this reproducer is that it contains **none of our model code**: no attention,
no MoE, no `gpu_fa4_cute`, no `sonic_cute`, no CuTe/CUTLASS kernels, no levanter, no marin
executor. It is a dense MLP stack that preserves the mesh, sharding, collective structure and
runtime flags of the production training job, and nothing else.

## Files

| file | role |
|---|---|
| `minimal_wedge_repro.py` | the reproducer. Dependencies: `jax`, `numpy`. Nothing else. |
| `minrepro_launch.py` | Iris/marin launcher. Only needed on our cluster (see below). |

## Prerequisites

**Standalone path** (any cluster): a Python environment with `jax` built against CUDA, plus
`numpy`. Nothing else. Verified with jax 0.11.0, NCCL 2.28.9, driver 595.71.05 on GB200. A
CPU-only jaxlib will run but silently exercises no GPU collectives -- the program prints
`backend=cpu` in its banner, so check that line.

**Iris path** (this repo): a fresh clone has no environment. `uv run` builds one from the
committed `uv.lock` on first use, which takes several minutes and a few GB:

```bash
git clone --branch mcwitt/7344-minimal-repro https://github.com/marin-community/marin.git
cd marin
uv sync            # or let the first `uv run --frozen ...` below do it
```

## Running it

### Anywhere with Slurm or OpenMPI (no wrapper needed)

`minimal_wedge_repro.py` is self-contained and needs no wrapper. `--distributed auto`
detects Slurm/OMPI coordinator env vars:

```bash
srun -N 32 --ntasks-per-node=1 --gpus-per-task=4 \
  python minimal_wedge_repro.py --dp-racks 2 --num-steps 20000
```

Mesh is `(dp_racks, nodes_per_rack * gpus_per_node, 1, 1)`; defaults assume 16 nodes/rack and
4 GPUs/node, so `--dp-racks 2` wants 32 processes x 4 GPUs = 128 GPUs. Override with
`--nodes-per-rack` / `--gpus-per-node` for a different topology.

### On our Iris cluster

A bare `iris job run` does **not** work, for two reasons worth knowing:

1. The environment lives in `$IRIS_VENV`; the system `python` has no `iris` module.
2. That venv resolves to a **CPU-only jaxlib** unless the GPU extra is selected, and the extra
   is chosen by `extras_for_resources(resources)` *inside* `dispatch_grug_training_run`. A
   direct submission silently falls back to CPU ("a CUDA-enabled jaxlib is not installed").

`minrepro_launch.py` routes through the same marin `ArtifactStep` + dispatch path as
`moe_hero_fsdp`, so the reproducer runs on an identical image, dependency set and launcher to
the production job — any behavioural difference is attributable to the program, not the stack.
It also calls `iris.runtime.jax_init.initialize_jax()`, which the runtime does **not** do for
you (the production job gets it via its trainer framework). Without it every task comes up
standalone with `device_count()==4` instead of 128 and performs no cross-node collectives at
all, while still printing entirely plausible output.

```bash
uv run --frozen iris --cluster=marin job run \
  --target-cluster cw-us-east-08a --priority production --no-wait \
  --timeout 43200 --max-retries 0 \
  --job-name minrepro-2rack-coord \
  -- python -m experiments.grug.minrepro_launch \
     --run-id minrepro-2rack --dp-racks 2 --num-steps 20000 --version dev --run
```

## What we observed

Reproduced on 2 racks (32 nodes x 4 GB200 = 128 GPUs), NCCL 2.28.9, driver 595.71.05.

**Time to wedge: seconds.** Trials wedged between step 0 and step 347, i.e. 8-28 s of
training. The production job takes ~40 min. The *collective operation counts* at wedge
(1000-17450) are comparable to the production job's (~9920-10604), so the trigger appears to
track collective operations rather than wall-clock; this reproducer simply issues them ~300x
faster.

### The signature

- No progress, indefinitely. `failures=0`, all ranks alive, no timeout ever fires --
  including `--xla_gpu_nccl_termination_timeout_seconds=600`, which is set and does not fire
  after an hour.
- **Power frozen**, GPUs pinned: 100% SM utilisation, 0% memory-controller utilisation,
  ~210 W (vs ~950 W healthy), clocks pinned at 1950 MHz. Power is identical to two decimal
  places across samples minutes apart.
- **NCCL RAS operation-count gap**: `echo verbose status | nc localhost 28028` reports one
  rank behind the other 127 by a few operations, static forever.
- **Zero errors anywhere**: no Xid in the host kernel log, no non-zero InfiniBand error
  counters (`packet_seq_err`, `local_ack_timeout_err`, `out_of_buffer`, CQE errors all 0), and
  zero warn/error/timeout/retry lines in 1.2 MB of `NCCL_DEBUG=INFO` output.

### Where it is stuck

py-spy across all 32 processes gives a clean 1-vs-31 split, reproducible on every trial:

| processes | Python line |
|---|---|
| 1 | `train_step(...)` -- inside collective *dispatch* |
| 31 | `float(loss)` -- blocked on the device->host readback |

The single process at `train_step` is always on the node NCCL RAS independently names as
lagging. A single py-spy dump therefore identifies the lagging rank without RAS.

Native stacks on the lagging rank (4 threads spinning at 100% CPU; every other rank's threads
are asleep):

```
ncclLocalOpAppend <- SaveProxy <- ncclProxySaveOp <- uploadProxyOps
  <- hostStreamPlanTask <- ncclLaunchKernelAfter_NoCuda <- doLaunches
  <- groupLaunch <- ncclGroupEndInternal <- ncclGroupEnd
  <- xla::gpu::NcclGroupLaunch
```

i.e. the rank cannot **launch** its next collective -- it spins on `sched_yield` because the
proxy op queue is full.

### Why the queue never drains -- root cause

*(This section supersedes an earlier draft that attributed the stall to the GPUDirect-RDMA
flush. `NCCL_NET_GDR_LEVEL=0` -- verified effective via `via NET/IB/N` connection lines with
no `/GDRDMA` suffix -- still wedged at step 163, killing that hypothesis. The `flushed=0`
reading was where a victim pipeline froze, not the cause.)*

NCCL 2.28.9 leaks proxy-op slots on aarch64. The slot-return path in
`ncclProxyGetPostedOps` (proxy.cc:845-852) publishes freed slots with a weak
`__atomic_compare_exchange_n` whose retry condition checks the observed value
(`while (swap != oldFree)`) instead of the CAS result. A weak CAS may fail spuriously --
return false with the value unchanged -- and that case exits the loop without storing,
orphaning the whole batch of freed slots. On aarch64 the compiled sequence is ldaxr/stlxr
with the stlxr success flag never tested; the starved producer's
`__atomic_exchange_n(&freeOps[i], -1)` spin bounces the cache line and breaks the
reservation while leaving the value equal, so leaks concentrate at backpressure episodes.
x86 `lock cmpxchg` cannot fail spuriously, which is why only GB200 (Grace) sees this.

Measured on a live wedge (see `wedge_forensics/`): the lagging rank's 2048-slot partition
was fully leaked (freeOps=-1, producer cache empty, nothing posted, consumer idle in
`pthread_cond_wait` with `active=NULL`), and the three healthy local ranks showed deficits
of 830/446/702 slots -- the leak runs continuously everywhere and the first rank to hit
zero wedges the job: its launcher spins in `ncclLocalOpAppend` forever, every peer blocks
in the collective, and nothing errors or times out.

NVIDIA fixed exactly this loop in NCCL v2.29.3-1 (commit 25368a7f78ba, loop rewritten to
retry on the CAS return value). Causal A/B on this reproducer: `nvidia-nccl-cu13==2.28.9`
wedged 7/7 trials by step 347; overriding to `2.30.7` (one uv.lock line, see the pin commit
on this branch) ran clean -- see the issue thread for the final step count.

Caveat: `moe-hero-fsdp-10rack-t1nccl-20260805` (1-layer hero, 10 racks) wedged with 2.30.7
verifiably loaded, so a second mechanism may exist at hero scale. The forensics in
`wedge_forensics/` distinguish the slot-leak deadlock from anything else in ~15 minutes on
a live wedge; run them before theorizing about any future wedge.
