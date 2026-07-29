# D-8: EP32 after the EP-aware Newton–Schulz fix

## TL;DR

Supported. The empty-cluster retry completed all 120 steps with finite loss and
drop metrics on one physical slice. Tail steps 100–119 averaged 276.1K tok/s,
17.82% MFU, and 11.42% drops; final loss was 5.6268. The run emitted neither
the recorded 104.03/104.13 GiB allocation failure nor the involuntary-full-
rematerialization diagnostic. The old failures and this arm both used
`cuda_async`, so the allocator cannot explain the change.

The pre-registered prediction below was registered at 2026-07-28 22:32 UTC,
before any original or retry result was seen, and is carried forward verbatim.
It predicted a fast OOM and was falsified. This result supports the shared C2
root-cause hypothesis; it does not make EP32 a candidate operating point.

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

Predecessor job ID: `/mwittmann/deri-d8-ep32-c2-120-0728-1715`

### Empty-cluster retry

The prediction in the pre-registration section above is carried forward
verbatim. It was registered at 2026-07-28 22:32 UTC, before any result from the
original run or this retry was seen. The retry was pre-recorded here at
2026-07-29 04:08 UTC, before submission and before inspecting any retry result.

The retry keeps every experiment variable from the predecessor's command. The
only operational changes are the new run ID, the now-required default federated
route through `marin`, and default interactive priority on both the parent and
child. The cluster had emptied since the inconclusive 12/52 two-slice placement.

Exact retry command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --user mwittmann --memory 2GB \
  --job-name deri-d8-ep32-c2-120-0728-2108-r1 \
  -e RUN_ID deri-d8-ep32-c2-120-0728-2108-r1 \
  -e IRIS_CHILD_PRIORITY 0 \
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
  -e SCALE_JSON_LOGGER deri-d8-ep32-c2-120-0728-2108-r1.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d8-ep32-c2-dev --run
```

Retry job ID: `/mwittmann/deri-d8-ep32-c2-120-0728-2108-r1`

Iris accepted the job at 2026-07-29 04:08:42 UTC through the `marin`
controller and reported that it was federating to `cw-us-east-08a`. The peer
controller recorded priority band 2, interactive, for both the parent and all
16 child tasks.

The first GPU-gang attempt failed during `stage-workdir` on node `s1zsxs64`.
The other 15 tasks were atomically bounced as coscheduled siblings. This is a
new failing node, distinct from the three known bad nodes `s4bk6j84`,
`s5kvxs64`, and `s6xvdgb4`. Iris's built-in retry placed all 16 tasks on one
physical slice: the 64-device hardware-topology record reported
`slice_index=0` for every device. The clean attempt initialized the exact
EP32 configuration at 2026-07-29 04:10:35 UTC.

The pre-registered fast-death prediction was falsified by step 12 of the
120-step run. At step 12, loss was finite at 9.3383, throughput was
299.9K tok/s, reported MFU was 19.36%, and the populated drop fraction was
0.8706. The drop figure is an early-schedule observation, not a steady-tail
comparison. No approximately 104 GiB allocation failure or
involuntary-full-rematerialization signature had appeared.

At step 55 of 120, loss remained finite at 6.7591 and the drop fraction had
fallen to 0.2431. Throughput was 281.9K tok/s and reported MFU was 18.20%.
The falling MFU is coupled to lower drops and therefore more executed expert
work; it is not a like-for-like throughput regression. No NaN, Inf, OOM,
rematerialization warning, or new task failure appeared.

At step 95 of 120, loss was 5.9725, drop fraction was 0.1335, throughput was
278.0K tok/s, and reported MFU was 17.95%. The loss and drop trajectories
remained finite and downward; no OOM, rematerialization warning, or numerical
failure had appeared.

### Predecessor attempt

The first parent attempt used `--version d8-ep32-c2-0728-1532`, which the
launcher rejected because version labels must be calendar versions or end in
`-dev`. It exited before requesting GPUs. The second parent attempt used the
valid label but exited 137 during dependency sync under Iris's 1 GB default
parent-memory request, also before requesting GPUs. A third attempt with 2 GB
used the federated `marin` route at interactive priority and never entered the
target cluster's scheduler; it was cancelled without running. The final
predecessor attempt kept the 2 GB parent-memory request and submitted directly to
`cw-us-east-08a` at production priority. No experiment configuration changed
across these attempts.

The first direct-production attempt,
`/mwittmann/deri-d8-ep32-c2-120-0728-2323`, then failed in 11 seconds with
`Init:Error stage-workdir`, before setup or GPU-child creation. The next
attempt, `/mwittmann/deri-d8-ep32-c2-120-0728-2328`, exposed a nested-priority
bug: its parent requested production, but the Fray-created GPU child defaulted
to interactive. It was stopped before training. The launcher now maps
`IRIS_CHILD_PRIORITY` to the existing Fray `JobRequest.priority` field, and the
predecessor command set it to production (`1`). Controller state confirmed
production on the parent, child, and all 16 GPU tasks.

The predecessor child used its first two automatic retries on transient
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

The retry passed the pre-registered operational and mechanistic criteria. Both
the parent and 16-task child succeeded, and the metric log contains all
120 steps from 0 through 119. The child ran for 38:59 after one automatic gang
retry. Its first attempt failed during `stage-workdir` on `s1zsxs64`; the
successful attempt used one physical slice and none of the four recorded
stage-workdir nodes.

Loss stayed finite and fell from 9.3383 at step 12 to 5.6268 at step 119. Drop
fraction fell from 87.06% at step 12 to 9.32% at step 119. Over tail steps
100–119 of this 120-step schedule:

- throughput averaged 276.1K tok/s with a 276.7K median;
- reported MFU averaged 17.82% with a 17.86% median;
- drop fraction averaged 11.42% with a 10.34% median;
- loss averaged 5.7691 with a 5.7517 median.

Every MFU figure above is paired with tok/s and drops from the same tail
window. The 2.5 PFLOP/s-per-GB200 bf16-dense denominator is unchanged. These
numbers characterize the diagnosis; the high drop rate and EP32 dispatch cost
exclude them from an operating-point comparison.

There is no failed allocation size to report. The complete log contains no
104.03/104.13 GiB allocation failure, `RESOURCE_EXHAUSTED`/OOM, or
involuntary-full-rematerialization diagnostic. Coordinator connection-refused
messages appeared only during shutdown after task 0 exited; all 16 tasks
finished successfully and the full metric series was already present.

The allocator interpretation trap is resolved. Commit `8198bd364` records the
original `/mwittmann/mfu-64g-a2aep` failures with `recompute_all` and
`cuda_async`, including both allocation sizes and the SPMD diagnostic. This arm
also used `cuda_async`. No default-allocator follow-up was run.

## Verdict

The shared-root-cause hypothesis is supported. On the C2 branch and the same
allocator as the historical failures, a one-slice placement completed the real
rematerialized step 120 times without the allocation or SPMD signature that
defined the old EP32 failure. This is not a single-commit A/B, so it does not
prove C2 is the only relevant code difference, but it meets both registered
criteria for the hypothesis.

No allocator leg, tuning arm, throughput arm, or additional placement draw was
submitted. EP32 still pays dispatch cost without EP64's memory relief, and this
120-step run retained 11.42% mean tail drops. The result closes the D-8 memory
diagnosis without reopening EP32 as a production candidate.
