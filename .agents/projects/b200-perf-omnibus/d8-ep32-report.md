# D-8: EP32 after the EP-aware Newton–Schulz fix

## TL;DR

Still undetermined. The pre-registered fast OOM did not occur: the final
attempt compiled far enough to hold 148,525 MiB (145.0 GiB) resident on GPU 0
without the recorded approximately 104 GiB allocation failure. It never emitted
a training step, however. Its 64 GPUs were fragmented 12/52 across two physical
slices, and the first-step compile/execution phase stopped producing logs while
all four probed GPUs remained at 100% low-power utilization. The job was stopped
after one hour, at the experiment budget. This report was pre-registered at
2026-07-28 22:32 UTC, before any result was seen.

## Pre-registration

The predicted result is an OOM at or before step 0. If the failure is unchanged,
`jit_train_step` should request one approximately 104 GiB temporary, specifically
within 104.0–104.2 GiB of the two recorded EP32 requests (104.03 and 104.13 GiB),
alongside the SPMD involuntary-full-rematerialization signature for the
microbatch input reshard into the `(data, expert)` mesh.

The operational pass criterion is all 120 steps with finite loss and a populated,
finite drop metric, with no approximately 104 GiB allocation failure. A
mechanistic pass for the shared-root-cause claim additionally requires the
microbatch input reshard to stop emitting the involuntary-full-rematerialization
signature. Reaching step 0 but failing later for a different reason would show
that the original EP32 memory wall moved, but would not satisfy the 120-step
criterion.

The run holds the d5120, 8-of-256 EP64 reference configuration fixed and changes
only `SCALE_EXPERT_AXIS` from 64 to 32. On 64 GPUs with replica axis 1, this
changes the compact mesh from `data=1, expert=64` to `data=2, expert=32`.

### Code-path gate

The D-2 padded-Muon change is not needed for this diagnosis. On this branch,
4D expert leaves route to `_newtonschulz_4d_distributed`, while padded Muon is a
separate opt-in branch for 3D non-expert optimizer leaves in
`lib/levanter/src/levanter/optim/grugmuon.py`. Commit `497423bc6` changes only
the outbound reshard of that 3D padded optimizer stack and its test. The
microbatch metadata and activation resharding use `_batch_reshard` in
`experiments/grug/moe/model.py`; they do not call either padded-Muon helper.

### Allocator ambiguity

The allocator ambiguity is already resolved by the contemporaneous record. The
original job was `/mwittmann/mfu-64g-a2aep`; the B200MFU-033 logbook records its
reference config as `recompute_all` with
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`. The preceding B200MFU-032 issue update
also states that all reference rerun arms used `cuda_async` unless marked
otherwise. The current Iris controller no longer retains
`/mwittmann/mfu-64g-a2aep`, so its serialized job environment cannot be recovered
from live history. A pass in this run therefore does not require a default-BFC
follow-up: both the old failure and this arm use `cuda_async`.

## Job

Job ID: `/mwittmann/deri-d8-ep32-c2-120-0728-1715`

The first parent attempt used `--version d8-ep32-c2-0728-1532`, which the
launcher rejected because version labels must be calendar versions or end in
`-dev`. It exited before requesting GPUs. The second parent attempt used the
valid label but exited 137 during dependency sync under Iris's 1 GB default
parent-memory request, also before requesting GPUs. A third attempt with 2 GB
used the federated `marin` route at interactive priority and never entered the
target cluster's scheduler; it was cancelled without running. The current
attempt keeps the 2 GB parent-memory request and submits directly to
`cw-us-east-08a` at production priority. No experiment configuration changed
across these attempts.

The first direct-production attempt,
`/mwittmann/deri-d8-ep32-c2-120-0728-2323`, then failed in 11 seconds with
`Init:Error stage-workdir`, before setup or GPU-child creation. The next
attempt, `/mwittmann/deri-d8-ep32-c2-120-0728-2328`, exposed a nested-priority
bug: its parent requested production, but the Fray-created GPU child defaulted
to interactive. It was stopped before training. The launcher now maps
`IRIS_CHILD_PRIORITY` to the existing Fray `JobRequest.priority` field, and the
current command sets it to production (`1`). Controller state confirmed
production on the parent, child, and all 16 GPU tasks.

The current child used its first two automatic retries on transient
`Init:Error stage-workdir` failures. Its third attempt was placed across three
physical slices and failed before JAX initialization because the generic
trainer requires 64 devices to be divisible by the physical slice count. The
fourth and final attempt was placed across two slices and reached the intended
logical mesh `(replica=1, data=2, expert=32, model=1)`. These were retries of
the same submitted job; no experiment parameter changed.

Exact submission command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
  --priority production --memory 2GB \
  --job-name deri-d8-ep32-c2-120-0728-1715 \
  -e RUN_ID deri-d8-ep32-c2-120-0728-1715 \
  -e IRIS_CHILD_PRIORITY 1 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_GATHER_DISPATCH 1 -e SCALE_A2A_CUSTOM_ADJOINT 1 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_REPLICA_AXIS 1 -e SCALE_EXPERT_AXIS 32 \
  -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 120 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_MOE_QB 1 -e SCALE_CAPACITY_FACTOR 1.0 -e SCALE_REPORT_DROPS 1 \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER deri-d8-ep32-c2-120-0728-1715.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d8-ep32-c2-dev --run
```

## Result

The fourth attempt initialized the exact pre-registered configuration and
entered compilation/execution of `jit_train_step`. It did not reproduce either
recorded EP32 failure: there was no failed 104.03/104.13 GiB allocation, no
`RESOURCE_EXHAUSTED`/OOM, and no involuntary-full-rematerialization diagnostic.
There is therefore no failed allocation size to report. A live `nvidia-smi`
probe instead showed a stable resident footprint of 148,525 MiB (145.0 GiB) on
GPU 0 and 148,503 MiB on each of the other three GPUs on task 0, out of
189,471 MiB per device.

This was not a pass. No loss, drop-fraction, throughput, or completed-step
metric was emitted. The hardware-topology record shows that processes 0–2
(12 GPUs) were on physical slice 0 and processes 3–15 (52 GPUs) were on
physical slice 1. After XLA initialized the first 64-device clique, the run
produced no further trainer output. Repeated probes showed the same resident
memory, 100% reported utilization, and only 206–233 W on task 0. All 16 tasks
remained in that state until the exact root job was stopped at the budget
boundary. Iris records the final attempt duration as 1:01:07 and both the
parent and child as `killed: Terminated by user`.

The allocator interpretation trap does not require a follow-up arm here. The
historical EP32 failure and this arm both used `cuda_async`, so the absence of
the old OOM cannot be attributed to an allocator change. The arm did not pass
the 120-step criterion, and the scope permits no default-allocator leg for a
non-pass.

## Verdict

The shared-root-cause hypothesis is still undetermined. The missing
approximately 104 GiB failure is evidence that the old EP32 memory wall moved
on the C2 code, under the same allocator. It is not enough to establish that C2
fixed the microbatch input-resharding path: the run completed no step and
emitted no mechanistic reshard/rematerialization evidence, while the fragmented
two-slice placement introduced a separate collective-progress confounder.

No tuning, throughput arm, allocator arm, or placement retry was submitted. A
decisive rerun would keep this command unchanged and require one contiguous
physical slice; that was not attempted under the one-job and approximately
one-rack-hour scope.
