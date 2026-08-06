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

### Why the queue never drains

Walking `ncclProxyProgressState->active` with lldb (the build carries debug symbols), every
stalled op on every proxy thread reads, frozen and identical across samples 30 s apart:

```
peer=17/18/19  channelId=0  nsteps=2  nbytes=524288
posted=2   received=2   flushed=0   transmitted=0   done=0
```

NCCL's receive path advances `posted -> received -> flushed -> transmitted -> done`.
`received=2, flushed=0` means **the data arrived but the GPUDirect-RDMA flush never
completed**. The flush is a small RDMA *read* issued after the incoming writes to guarantee
they are visible to the GPU before the receive is signalled. It is an ordinary RDMA operation
that simply never returns a completion -- which is exactly why nothing errors, nothing times
out, and no counter increments anywhere.

The backpressure chain: flush never completes -> op never reaches `done` -> proxy queue never
drains -> `ncclLocalOpAppend` spins -> the rank cannot launch -> RAS shows it behind -> the
other 127 ranks block on the collective.

Three *different* peers (17, 18, 19) are stuck at flush simultaneously on the same node, which
argues against a single broken peer link and for something local to that node's NIC/PCIe/GPU
path.

The stalled collectives are always **large** AllReduces (37,748,736 bf16 elements = 75.5 MB,
`pattern=10` NvlsTree, `protocol=2` SIMPLE, chunked into 512 KB steps). The healthy ranks sail
past to a `count=1` fp32 op -- the loss scalar. Small collectives are not implicated.

## Open questions -- please read before drawing conclusions

1. **Reproducible solo, but not on an idle cluster.** 6/6 trials wedged. Five ran alongside
   3-4 concurrent 128-GPU jobs of ours; the sixth ran with no other job of ours anywhere and
   still wedged, at step 211 after ~14 s. That removes our own concurrency as the cause. Other
   tenants' jobs still shared the fabric, so "solo" means solo-for-us, not a quiet cluster.
2. **Whether this is the same bug as the production wedge is not settled.** It matches on the
   four-part signature, but differs in three ways: the production job lags a *whole
   64-rank rack* where this lags a *single rank*; production py-spy showed lagging and healthy
   ranks on the *same* line where this splits 389/392; and time-to-wedge differs ~100x. We have
   never inspected proxy state on a production wedge -- the `flushed=0` finding comes only from
   this reproducer.
3. **`NCCL_DEBUG=INFO` perturbs timing** -- the run with it enabled wedged at step 0 rather
   than 18-347. Do not read significance into that.
4. **`flushed` semantics** are read from NCCL's receive state machine, not verified against
   the 2.28.9 source.

## Suggested next test

`NCCL_NET_GDR_LEVEL=0` disables GPUDirect RDMA, so receives stage through host memory and no
flush is required. If the wedge disappears, the GDR flush path is confirmed. At ~20 s to
wedge this is a few minutes of cluster time.

(Disabling only the flush is the sharper test but removes a correctness barrier and risks
silent data corruption -- acceptable as a diagnostic, never as a fix.)
