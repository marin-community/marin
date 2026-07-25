# EP25 Direction 2 MXFP8 Relay Commands

Run these commands from
`/home/marin/projects/marin/.worktrees/ep25-d2-bakeoff` in the order shown.
Do not advance to the next stage until the current jobs are terminal and their
logs have been checked. All training jobs use the stock Marin toolchain.

If setup fails before user code starts, resubmit the exact command with a fresh
`-vN` suffix on the job name, `RUN_ID`, and JSON logger name.

## 1. GB200 numerical ladder

This compares the fused MXFP8 expert MLP with an independent float32 reference
at the d5120/i1280 four-expert local operating shape. It also verifies exact
zero gradients for zero-token experts.

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-numerics-20260725-v2 \
  --enable-extra-resources --gpu GB200x1 --cpu 16 --memory 96GB \
  --extra gpu \
  -- python experiments/grug/moe/standalone/check_mxfp8_expert_mlp.py \
  --out /tmp/ep25d2-mxfp8-numerics-20260725-v2.json
```

Gate: the job must exit zero, all reported relative Frobenius errors must be
below 0.1, and both empty-expert weight-gradient checks must be exactly zero.

## 2. EP4 drop-parity smoke

Run the BF16 control first, then the MXFP8 treatment. These are four-step,
four-GPU smokes of the fixed gather-dispatch path with the custom adjoint,
queue batching enabled, capacity factor 1.0, and drop reporting enabled.

### BF16 control

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env -u SCALE_MOE_MXFP8 IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-ep4-bf16-20260725-v2 \
  -e RUN_ID ep25d2-mxfp8-ep4-bf16-20260725-v2 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 2048 -e SCALE_NUM_LAYERS 4 \
  -e SCALE_INTERMEDIATE 512 -e SCALE_SHARED_INTERMEDIATE 2048 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 64 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 4 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_GATHER_DISPATCH 1 \
  -e SCALE_A2A_CUSTOM_ADJOINT 1 -e SCALE_MOE_QB 1 \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-mxfp8-ep4-bf16-20260725-v2.metrics \
  -e SCALE_REPORT_DROPS 1 -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-mxfp8-r6-v2-dev --run
```

### MXFP8 treatment

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-ep4-treatment-20260725-v2 \
  -e RUN_ID ep25d2-mxfp8-ep4-treatment-20260725-v2 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 2048 -e SCALE_NUM_LAYERS 4 \
  -e SCALE_INTERMEDIATE 512 -e SCALE_SHARED_INTERMEDIATE 2048 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 64 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 4 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_GATHER_DISPATCH 1 \
  -e SCALE_A2A_CUSTOM_ADJOINT 1 -e SCALE_MOE_QB 1 \
  -e SCALE_MOE_MXFP8 1 \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-mxfp8-ep4-treatment-20260725-v2.metrics \
  -e SCALE_REPORT_DROPS 1 -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-mxfp8-r6-v2-dev --run
```

Gate: both jobs must exit zero and report identical dropped-assignment counts at
every step. The loss values should remain close enough to justify the 120-step
matched pair; record any finite discrepancy rather than imposing a smoke-only
loss threshold.

## 3. Matched QB-on rack pair

Run one arm at a time. Both arms use the d5120, 48-layer, 8-of-256, EP64,
batch-1024, sequence-4096 operating point on one 64-GPU GB200 rack. Queue
batching, gather dispatch, the custom adjoint, capacity factor 1.0, and drop
reporting are enabled in both arms. The only treatment difference is
`SCALE_MOE_MXFP8=1`.

### BF16 control

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env -u SCALE_MOE_MXFP8 IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-rack-bf16-120-20260725-v2 \
  -e RUN_ID ep25d2-mxfp8-rack-bf16-120-20260725-v2 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 120 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_GATHER_DISPATCH 1 \
  -e SCALE_A2A_CUSTOM_ADJOINT 1 -e SCALE_MOE_QB 1 \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-mxfp8-rack-bf16-120-20260725-v2.metrics \
  -e SCALE_REPORT_DROPS 1 -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-mxfp8-r6-v2-dev --run
```

### MXFP8 treatment

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-rack-treatment-120-20260725-v2 \
  -e RUN_ID ep25d2-mxfp8-rack-treatment-120-20260725-v2 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 120 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_GATHER_DISPATCH 1 \
  -e SCALE_A2A_CUSTOM_ADJOINT 1 -e SCALE_MOE_QB 1 \
  -e SCALE_MOE_MXFP8 1 \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-mxfp8-rack-treatment-120-20260725-v2.metrics \
  -e SCALE_REPORT_DROPS 1 -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-mxfp8-r6-v2-dev --run
```

Report p50 MFU from matched steady-state steps, per-step and final drop parity,
the step-119 loss delta, and peak device memory from each Iris summary. Compare
the control against the 22.595%/22.002% reference band. Project the compliant
configuration as:

```text
projected_cf1.15_MFU = measured_MXFP8_cf1.0_MFU * (20.848 / matched_BF16_cf1.0_MFU)
```

The adoption gate remains a long-horizon quality run resolving issue #7271's
reported +0.11–0.21% held-out-loss regression. The CuTe quantization producer
is a follow-up only after its known 16-node executable-load failure is fixed.
