# D-1: FSDP baseline token-drop rates

Status: corrected direct-cluster resubmission prepared; no D-1 metrics inspected.

The first submission attempt used the federated `marin` route and
`interactive` priority. Both jobs remained outside the target cluster's
scheduler and were cancelled without producing a result. Their commands and
IDs remain below as a record of the failed launch; they are not measurements.
The pre-registration predates both that attempt and the corrected resubmission.

## TL;DR

The measurements are pending. D-1a is the positive control: its chunked local
expert path has a static per-chunk capacity and should report nonzero overflow.
D-1b uses the unchunked local expert path, which has no capacity, so its
pre-registered prediction is exactly zero drops.

No D-1b zero will be interpreted unless D-1a first reports a nonzero drop
fraction. If D-1a reports exactly `0.000000`, the metric port is unproven and
both results will be withheld while the port is debugged.

## Pre-registration

Recorded before inspecting any D-1 run metrics.

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

## D-1a: d6144, 4-of-128, chunk-2

Job ID: `/marin/d1a-fsdp-drops-350-20260728-224739`

Training gang:
`/marin/d1a-fsdp-drops-350-20260728-224739/grug-train-d1a-fsdp-drops-350-20260728-224739`

Exact command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name d1a-fsdp-drops-350-20260728-224739 \
  -e RUN_ID d1a-fsdp-drops-350-20260728-224739 \
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
  -e SCALE_JSON_LOGGER d1a-fsdp-drops-350-20260728-224739.metrics \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d1-fsdp-drops-cefc6d47b --run
```

This reproduces the recorded one-rack 23.1% configuration: d6144, 48 layers,
4-of-128, routed width 3072, shared width 6144, 4096-token sequences, local
window 512 with a global layer every six layers, global batch 1024, the full
QB/XSA/gating/MuonH stack, and two expert chunks. The historical command
omitted the two widths because those were the heuristic defaults; they are
explicit here to prevent default drift.

Observed capacity factor: pending

Results: pending

## D-1b: d5120, 8-of-256, unchunked

Job ID: `/marin/d1b-fsdp-drops-120-20260728-224739`

Training gang:
`/marin/d1b-fsdp-drops-120-20260728-224739/grug-train-d1b-fsdp-drops-120-20260728-224739`

Exact command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name d1b-fsdp-drops-120-20260728-224739 \
  -e RUN_ID d1b-fsdp-drops-120-20260728-224739 \
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
  -e SCALE_JSON_LOGGER d1b-fsdp-drops-120-20260728-224739.metrics \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version d1-fsdp-drops-cefc6d47b --run
```

This keeps the recorded d5120 architecture and per-device batch: 48 layers,
8-of-256, routed width 1280, shared width 5120, 4096-token sequences, local
window 2048, MuonH with SYRK, and the sonic-cute/FA4 backends. The historical
19.17% measurement used two racks and global batch 2048; this diagnostic uses
one rack and global batch 1024. `SCALE_MOE_EXPERT_CHUNKS=1` makes the historical
unchunked default explicit.

Observed capacity factor: pending

Results: pending

## Verdict

Pending D-1a validation and both completed measurements.
