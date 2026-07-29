# D-1: FSDP baseline token-drop rates

Status: blocked after the protocol's allowed placement retries. The post-release
jobs used the default federated route and default priority, but D-1a never
reached model code.

The retry established that the earlier federation bug is fixed: every parent
handed off promptly from `marin` to `cw-us-east-08a`, and SQL inspection showed
all child tasks at the default interactive `priority_band=2`. Capacity was not
the constraint. D-1a received three honest post-release placements; they failed
in the `stage-workdir` init container on `s4bk6j84`, `s1zsxs64`, and
`s1zsxs64`, respectively. The first node was already in the shared bad-node
list; `s1zsxs64` is a newly observed failure with the same signature.

D-1b's second placement was clean and reached the training loop, but it emitted
only an isolated step-0 drop metric before being stopped. Because the D-1a
positive control never ran, that zero is not interpreted as a D-1 measurement.
No node or cluster state was mutated, and no other user's job was touched.

## TL;DR

The measurements remain unresolved. D-1a completed 0 of 350 configured steps
across all retry attempts. D-1b logged `moe/drop_fraction=0.0` and
`moe/dropped_assignments=0` only at step 0 of a configured 120-step run, then
was stopped because that number was uninterpretable without the positive
control.

No D-1b zero will be interpreted unless D-1a first reports a nonzero drop
fraction. If D-1a reports exactly `0.000000`, the metric port is unproven and
both results will be withheld while the port is debugged.

The reporting denominator remains 2.5 PFLOP/s per GB200, bf16-dense. Neither
leg produced a reportable MFU or tok/s figure.

## Pre-registration

Recorded before inspecting any D-1 run metrics.

The prediction block below was committed in `4b03d1019` before the first D-1
submission and is carried forward verbatim.

- D-1a prediction: the 350-step d6144, 4-of-128, chunk-2 FSDP run will report
  nonzero drops, with a late-run drop fraction in the 6–13% range. This is the
  positive control because `_moe_mlp_local_sonic_cute_chunked` gives each chunk
  a static expert capacity and discards assignments beyond it.
- D-1b prediction: the 120-step d5120, 8-of-256, unchunked FSDP run will report
  a structural zero. `_moe_mlp_local_sonic_cute` sorts every assignment, sizes
  each expert group with `jnp.bincount`, and introduces no capacity bound.
- Falsification rule: FSDP drops in the 6–13% range make the existing
  EP-versus-FSDP comparison fair on fidelity and leave EP further ahead than the
  published throughput-only comparison indicates. Drops below 3% mean the
  comparison is unfair against FSDP: EP's reported advantage is smaller than
  claimed and may be negative after matching fidelity.
- LR-position rule: drop fractions will be compared only at the same fraction
  of each run's LR schedule. Every reported window will include its step range
  and total run length.
- D-1b scope: its one-rack MFU and tok/s will be recorded for run diagnostics
  only. They will not be compared with the historical two-rack 19.17% baseline.

## Pre-run validation

The drop-reporting flag was exercised on CPU with a fake expert MLP: the same
`MoEMLP.__call__` returned zero with `SCALE_REPORT_DROPS` unset and propagated a
sentinel count of seven with it set. Static inspection then followed the count
through `Block`, the layer scan, `next_token_loss`, and the outer training
logger. The five chunked-path regression cases pass.

The two local expert paths have different semantics:

- D-1a enters `_moe_mlp_local_sonic_cute_chunked`. Each expert chunk has a
  static assignment capacity. Its overflow mask drops assignments, and the
  next chunk starts after the overloaded chunk's full assignment range.
- D-1b enters `_moe_mlp_local_sonic_cute`. It sorts every assignment, obtains
  exact per-expert group sizes with `jnp.bincount`, and scatters sorted token
  indices that are in bounds by construction. There is no capacity.

The effective capacity factor is `1.0`. `experiments/grug/moe/model.py` passes
its `_DEFAULT_EP_CAPACITY_FACTOR = 1.0` explicitly into every expert MLP. That
explicit value takes precedence over the separate `1.25` default in
`lib/levanter/src/levanter/grug/_moe/common.py`. Each job's runtime
hyperparameters will be checked below to confirm that it ran the intended
model and implementation.

The CPU propagation probe was repeated before this retry: the same
`MoEMLP.__call__` reported zero with `SCALE_REPORT_DROPS` unset and propagated a
sentinel count of seven with it set. The five chunk-capacity regression cases
also passed.

## Federated retry after the cluster release

The retry uses the default federated route through `marin`, pins the handoff to
`cw-us-east-08a`, and leaves both the parent and Fray child at Iris's default
interactive priority. These commands were recorded before either job was
submitted or any new result was inspected.

D-1a retry job ID:
`/mwittmann/d1a-fsdp-drops-350-r8-20260728-2108`

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --user mwittmann --memory 2GB \
  --job-name d1a-fsdp-drops-350-r8-20260728-2108 \
  -e RUN_ID d1a-fsdp-drops-350-r8-20260728-2108 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 \
  -e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 1 -e SCALE_REPLICA_AXIS 1 \
  -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_NUM_KV_HEADS 8 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4 \
  -e SCALE_INTERMEDIATE 3072 -e SCALE_SHARED_INTERMEDIATE 6144 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_SLIDING_WINDOW 512 \
  -e SCALE_GLOBAL_EVERY 6 -e SCALE_BATCH 1024 -e SCALE_STEPS 350 \
  -e SCALE_DATA slimpajama -e SCALE_OPTIMIZER muonh \
  -e SCALE_MOE_QB 1 -e SCALE_XSA 1 -e SCALE_ATTN_GATE 1 \
  -e SCALE_GATED_NORM 1 -e SCALE_OFFLOAD_OPT_STATE 1 \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -e SCALE_MUON_DIST_NONEXPERT 1 -e SCALE_MUON_INTRA_RACK 1 \
  -e SCALE_MUON_PAD_NONEXPERT 1 -e SCALE_MUON_SYRK 1 \
  -e SCALE_WATCH_INTERVAL 20 -e SCALE_MOE_IMPL sonic_cute \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_SCAN_LAYERS 1 \
  -e CE_IMPL liger -e CE_LIGER_CHUNK 8192 \
  -e SCALE_REMAT recompute_all -e SCALE_MOE_EXPERT_CHUNKS 2 \
  -e SCALE_REPORT_DROPS 1 -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER d1a-fsdp-drops-350-r8-20260728-2108.metrics \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d1-fsdp-drops-fc5532108 --run
```

D-1b retry job ID:
`/mwittmann/d1b-fsdp-drops-120-r5-20260728-2108`

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --user mwittmann --memory 2GB \
  --job-name d1b-fsdp-drops-120-r5-20260728-2108 \
  -e RUN_ID d1b-fsdp-drops-120-r5-20260728-2108 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 \
  -e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 1 -e SCALE_REPLICA_AXIS 1 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_NUM_KV_HEADS 10 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_BATCH 1024 -e SCALE_STEPS 120 \
  -e SCALE_DATA slimpajama -e SCALE_OPTIMIZER muonh \
  -e SCALE_MUON_SYRK 1 -e SCALE_WATCH_INTERVAL 20 \
  -e SCALE_MOE_IMPL sonic_cute -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_SCAN_LAYERS 1 -e CE_IMPL liger -e CE_LIGER_CHUNK 8192 \
  -e SCALE_REMAT recompute_all -e SCALE_MOE_EXPERT_CHUNKS 1 \
  -e SCALE_DISABLE_CHECKPOINT 1 -e SCALE_REPORT_DROPS 1 \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER d1b-fsdp-drops-120-r5-20260728-2108.metrics \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d1-fsdp-drops-fc5532108 --run
```

Both jobs handed off immediately and their child gangs were persisted at the
default interactive `priority_band=2`. They failed before Python model code:
D-1a task 1 landed on known-bad node `s4bk6j84` and D-1b task 0 landed on
`s1zsxs64`; both reported `Init:Error stage-workdir`. The other 15 members of
each gang were atomically bounced. No model metrics were emitted.

One further honest placement attempt per leg was recorded before submission.
The configuration is unchanged; only the run identity changes.

D-1a second-retry job ID:
`/mwittmann/d1a-fsdp-drops-350-r9-20260728-2113`

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --user mwittmann --memory 2GB \
  --job-name d1a-fsdp-drops-350-r9-20260728-2113 \
  -e RUN_ID d1a-fsdp-drops-350-r9-20260728-2113 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 \
  -e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 1 -e SCALE_REPLICA_AXIS 1 \
  -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_NUM_KV_HEADS 8 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4 \
  -e SCALE_INTERMEDIATE 3072 -e SCALE_SHARED_INTERMEDIATE 6144 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_SLIDING_WINDOW 512 \
  -e SCALE_GLOBAL_EVERY 6 -e SCALE_BATCH 1024 -e SCALE_STEPS 350 \
  -e SCALE_DATA slimpajama -e SCALE_OPTIMIZER muonh \
  -e SCALE_MOE_QB 1 -e SCALE_XSA 1 -e SCALE_ATTN_GATE 1 \
  -e SCALE_GATED_NORM 1 -e SCALE_OFFLOAD_OPT_STATE 1 \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -e SCALE_MUON_DIST_NONEXPERT 1 -e SCALE_MUON_INTRA_RACK 1 \
  -e SCALE_MUON_PAD_NONEXPERT 1 -e SCALE_MUON_SYRK 1 \
  -e SCALE_WATCH_INTERVAL 20 -e SCALE_MOE_IMPL sonic_cute \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_SCAN_LAYERS 1 \
  -e CE_IMPL liger -e CE_LIGER_CHUNK 8192 \
  -e SCALE_REMAT recompute_all -e SCALE_MOE_EXPERT_CHUNKS 2 \
  -e SCALE_REPORT_DROPS 1 -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER d1a-fsdp-drops-350-r9-20260728-2113.metrics \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d1-fsdp-drops-fc5532108 --run
```

D-1b second-retry job ID:
`/mwittmann/d1b-fsdp-drops-120-r6-20260728-2113`

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --user mwittmann --memory 2GB \
  --job-name d1b-fsdp-drops-120-r6-20260728-2113 \
  -e RUN_ID d1b-fsdp-drops-120-r6-20260728-2113 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 \
  -e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 1 -e SCALE_REPLICA_AXIS 1 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_NUM_KV_HEADS 10 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_BATCH 1024 -e SCALE_STEPS 120 \
  -e SCALE_DATA slimpajama -e SCALE_OPTIMIZER muonh \
  -e SCALE_MUON_SYRK 1 -e SCALE_WATCH_INTERVAL 20 \
  -e SCALE_MOE_IMPL sonic_cute -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_SCAN_LAYERS 1 -e CE_IMPL liger -e CE_LIGER_CHUNK 8192 \
  -e SCALE_REMAT recompute_all -e SCALE_MOE_EXPERT_CHUNKS 1 \
  -e SCALE_DISABLE_CHECKPOINT 1 -e SCALE_REPORT_DROPS 1 \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER d1b-fsdp-drops-120-r6-20260728-2113.metrics \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d1-fsdp-drops-fc5532108 --run
```

The D-1a second retry failed before model code on `s1zsxs64` with the same
`Init:Error stage-workdir`; this is the node that killed D-1b's first retry.
D-1b's second retry received a clean 16-node placement and reached runtime
configuration logging. One final D-1a placement attempt was recorded before
submission:

D-1a final placement-attempt job ID:
`/mwittmann/d1a-fsdp-drops-350-r10-20260728-2118`

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --user mwittmann --memory 2GB \
  --job-name d1a-fsdp-drops-350-r10-20260728-2118 \
  -e RUN_ID d1a-fsdp-drops-350-r10-20260728-2118 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 \
  -e SCALE_GPU_REPLICAS 16 -e SCALE_EXPERT_AXIS 1 -e SCALE_REPLICA_AXIS 1 \
  -e SCALE_HIDDEN_DIM 6144 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_NUM_KV_HEADS 8 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4 \
  -e SCALE_INTERMEDIATE 3072 -e SCALE_SHARED_INTERMEDIATE 6144 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_SLIDING_WINDOW 512 \
  -e SCALE_GLOBAL_EVERY 6 -e SCALE_BATCH 1024 -e SCALE_STEPS 350 \
  -e SCALE_DATA slimpajama -e SCALE_OPTIMIZER muonh \
  -e SCALE_MOE_QB 1 -e SCALE_XSA 1 -e SCALE_ATTN_GATE 1 \
  -e SCALE_GATED_NORM 1 -e SCALE_OFFLOAD_OPT_STATE 1 \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -e SCALE_MUON_DIST_NONEXPERT 1 -e SCALE_MUON_INTRA_RACK 1 \
  -e SCALE_MUON_PAD_NONEXPERT 1 -e SCALE_MUON_SYRK 1 \
  -e SCALE_WATCH_INTERVAL 20 -e SCALE_MOE_IMPL sonic_cute \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_SCAN_LAYERS 1 \
  -e CE_IMPL liger -e CE_LIGER_CHUNK 8192 \
  -e SCALE_REMAT recompute_all -e SCALE_MOE_EXPERT_CHUNKS 2 \
  -e SCALE_REPORT_DROPS 1 -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER d1a-fsdp-drops-350-r10-20260728-2118.metrics \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d1-fsdp-drops-fc5532108 --run
```

The final D-1a attempt also failed before model code on `s1zsxs64` with
`Init:Error stage-workdir`. Across the three post-release attempts, D-1a hit
`s4bk6j84`, then `s1zsxs64` twice. No further placement attempts are permitted
under the shared-infrastructure stop rule.

D-1b's clean second placement compiled and emitted
`moe/drop_fraction=0.0` at step 0. This is not a reportable D-1 result: the D-1a
positive control never reached model code, so the metric port has not been
validated end to end on the chunked FSDP path. The D-1b job was stopped after
step 0 to avoid spending the remainder of the rack leg on an uninterpretable
zero.

## D-1a: d6144, 4-of-128, chunk-2

Post-release parent job IDs:

- `/mwittmann/d1a-fsdp-drops-350-r8-20260728-2108`
- `/mwittmann/d1a-fsdp-drops-350-r9-20260728-2113`
- `/mwittmann/d1a-fsdp-drops-350-r10-20260728-2118`

Each parent spawned the correspondingly named `grug-train-*` child gang. All
three gangs were persisted at default interactive `priority_band=2` and
received placements. Their failing nodes were:

| Attempt | Failing node | Terminal reason |
| --- | --- | --- |
| r8 | `s4bk6j84` | `Init:Error stage-workdir` |
| r9 | `s1zsxs64` | `Init:Error stage-workdir` |
| r10 | `s1zsxs64` | `Init:Error stage-workdir` |

The other 15 tasks in each gang were atomically bounced because
`max_task_failures=0`. The exact commands for all three attempts are recorded
above; only the run identity changed.

The intended run reproduces the recorded one-rack 23.1% configuration: d6144,
48 layers, 4-of-128, routed width 3072, shared width 6144, 4096-token sequences,
local window 512 with a global layer every six layers, global batch 1024, the
full QB/XSA/gating/MuonH stack, and two expert chunks.

Result: unavailable. The configured run length was 350 steps, but every attempt
completed 0 steps. There is therefore no LR position, drop fraction, MFU,
tok/s, or loss trajectory to report. The positive control was not exercised.

Capacity factor: the submitted source bundle explicitly passes `1.0` from
`experiments/grug/moe/model.py`, overriding the separate library default of
`1.25`. No D-1a runtime hyperparameter record was emitted because model code
never started.

## D-1b: d5120, 8-of-256, unchunked

Post-release parent job IDs:

- `/mwittmann/d1b-fsdp-drops-120-r5-20260728-2108`
- `/mwittmann/d1b-fsdp-drops-120-r6-20260728-2113`

The r5 child gang failed before model code on `s1zsxs64` with
`Init:Error stage-workdir`. The r6 child gang received a clean placement on:

`s1wvxs64`, `s2zpxs64`, `sckwxs64`, `sdktxs64`, `sdntxs64`, `sgqtxs64`,
`sgzvxs64`, `sjnsxs64`, `s5trxs64`, `s5wvxs64`, `s6zwxs64`, `s75sxs64`,
`s8wvxs64`, `s9rtxs64`, `sbzvxs64`, and `sc1xxs64`.

Runtime configuration logging confirmed d5120, 48 layers, 8-of-256, routed
width 1280, shared width 5120, 4096-token sequences, local window 2048,
`sonic_cute`, `recompute_all`, 120 configured steps, and one rack of 64 GB200s.
The historical 19.17% measurement used two racks; no performance number from
this diagnostic is compared with it.

At step 0 of 120, r6 logged:

```text
moe/dropped_assignments=0
moe/drop_fraction=0.0
```

This is an observation from the first schedule position, not a measured drop
rate or tail window. The job was stopped after step 0 once the final D-1a
placement failed. It emitted no `train/loss`, MFU, tok/s, or performance-window
duration, so no loss-trajectory or performance sanity check is available. Iris
records r6 as killed by the submitting user, not as a passing run.

Capacity factor: runtime hyperparameters do not serialize a
`capacity_factor` field. The submitted source bundle explicitly passes `1.0`,
and the observed unchunked `sonic_cute` implementation has no capacity bound,
making the value operationally irrelevant for D-1b. The separate `1.25`
library default was not selected.

## Verdict

No EP-versus-FSDP fairness verdict can be rendered from these jobs. The
comparison remains unresolved because the D-1a positive control never reached
model code. Reporting D-1b's isolated step-0 zero as the predicted structural
zero would violate the pre-registered positive-control rule. The evidence does
not establish that the comparison is fair, unfair in EP's favour, or unfair
against EP.

The missing prerequisite is a clean D-1a placement after the infrastructure
owners repair or otherwise handle the broken `stage-workdir` nodes. D-1a must
then complete far enough to produce a nonzero positive-control metric and a
late-run window. Only after that may D-1b's zero be interpreted; a full
120-step D-1b run would still be needed for the requested loss, MFU, tok/s, and
run-level diagnostics.

The post-release failing-node set was `s4bk6j84` and `s1zsxs64`. Historical
pre-release attempts also repeatedly failed on `s6xvdgb4`. No cluster restart,
node cordon, taint, drain, patch, eviction, or other shared-infrastructure
mutation was performed.
