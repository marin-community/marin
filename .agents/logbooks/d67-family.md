---
topic: d67-family
description: Sequential D-7 / D-6a control, trio, and spill-ceiling experiment family
author: mwittmann
---

# D-7 / D-6a family: Task Logbook

## Scope

- Goal: Measure whether the GatedNorm / attention-gate / XSA trio transfers from FSDP to EP, and whether same-step spill continues reducing drops through m=7.
- Primary metrics: D-6a is judged on tokens/s. D-7 is judged on tail-100 drop fraction, with tokens/s beside it. MFU is reported but does not rank D-6a because XSA and the attention gate add uncounted FLOPs.
- Constraints: Six independent placement draws, submitted sequentially and allowed to reach terminal state before the next submission. The two m=3/cf1.0625 controls are shared across D-6a and D-7 and are submitted only once.
- Code: `agent/deri-d67` at `f53f781ce5358965b251dbf78d280dea9e053f73`, based exactly on `agent/ep25-d1-adjoint`.
- Hardware and run length: one rack, 16 nodes × 4 GB200, 350 steps per draw.
- Metric rules: 2.5 PFLOP/s per GB200 bf16-dense denominator; tail-100 drops from steps 250–349; full logs fetched with `--max-lines 400000`; finite declining loss required even when Iris reports success.

## Pre-registered family predictions

### D-6a trio transfer

- Prediction: enabling `SCALE_GATED_NORM=1`, `SCALE_ATTN_GATE=1`, and `SCALE_XSA=1` together increases the two-draw tokens/s band by 1–2% over the shared control. The prior is weak because the reported FSDP gain may not transfer to EP.
- Decision rule: the trio transfers only if it clears at least +1.0% tokens/s and the control and treatment two-draw bands do not overlap. If the bands overlap or straddle zero, the result is unresolved. A negative or sub-gate result is reported without searching for a rescue configuration.
- Conditional follow-up: do not submit the three single-variable arms without explicit user approval, even if the gate clears.

### D-7 spill ceiling

- Prediction: tail-100 drops improve monotonically from m=3 to m=5 to m=7. Starting from the prior m=3/cf1.0625 draw at 1.44%, expect approximately 1.0–1.2% at m=5 and 0.8–1.0% at m=7. Tokens/s should remain within 1% of m=3 because later spill attempts process only the surviving overflow assignments.
- Falsification: the architecture-selection claim is falsified if either increment is flat within the two-draw bands, if m=7 reverses relative to m=5, or if m=3→m=7 costs more than 1% tokens/s without at least a 20% relative drop reduction. Under those outcomes, top-8 has no demonstrated spill-headroom advantage over top-4 at this operating point.

## Entry Log

### 2026-07-28 15:35 PDT - D67-CTL-01 control draw 1 pre-registration

- Hypothesis: This draw reproduces the healthy m=3/cf1.0625 operating point. Predict about 321K tokens/s and 20.7% MFU, allowing ±4% tokens/s for placement (308–334K), with tail-100 drops 1.2–1.7%. Loss must remain finite and decline through step 349.
- Falsification: NaN/non-finite loss, a missing drop metric, fewer than 350 attempted steps without a documented completed-training teardown, or tail-100 drops above 3% makes the draw invalid or non-representative rather than a favorable control.
- Commit Hash: `f53f781ce5358965b251dbf78d280dea9e053f73`
- Command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name d67-control-m3-draw1-0728-1535 -e RUN_ID d67-control-m3-draw1-0728-1535 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_GATHER_DISPATCH 1 -e SCALE_A2A_CUSTOM_ADJOINT 1 \
  -e SCALE_MOE_QB 1 -e SCALE_REPORT_DROPS 1 -e SCALE_A2A_SPILL 3 \
  -e SCALE_CAPACITY_FACTOR 1.0625 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 -e SCALE_INTERMEDIATE 1280 \
  -e SCALE_SHARED_INTERMEDIATE 5120 -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 \
  -e SCALE_SLIDING_WINDOW 2048 -e SCALE_STEPS 350 \
  -e SCALE_MOE_IMPL ragged_all_to_all -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger -e SCALE_JSON_LOGGER d67-control-m3-draw1-0728-1535.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale --version d67-family-f53f781ce --run
```

- Config: control; QB on; spill m=3; capacity factor 1.0625; trio off.
- Job ID: `/mwittmann/d67-control-m3-draw1-0728-1535`
- Result: Submitted at 2026-07-28 15:35 PDT; pending terminal result.
- Interpretation: Pending.
- Next action: Submit this job only, record its canonical job ID, and monitor it to terminal before draw 2.

### 2026-07-28 15:38 PDT - D67-CTL-01 submission-label failure and recovery

- Job ID: `/mwittmann/d67-control-m3-draw1-0728-1535`
- Result: Failed before training or rack allocation. The launcher rejected `--version d67-family-f53f781ce`; versions must be calendar-formatted or use a mutable `-dev` label.
- Interpretation: This is not a placement draw and supplies no experimental result. The control prediction above remains pre-registered.
- Recovery command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name d67-control-m3-draw1-r1-0728-1538 -e RUN_ID d67-control-m3-draw1-r1-0728-1538 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_GATHER_DISPATCH 1 -e SCALE_A2A_CUSTOM_ADJOINT 1 \
  -e SCALE_MOE_QB 1 -e SCALE_REPORT_DROPS 1 -e SCALE_A2A_SPILL 3 \
  -e SCALE_CAPACITY_FACTOR 1.0625 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 -e SCALE_INTERMEDIATE 1280 \
  -e SCALE_SHARED_INTERMEDIATE 5120 -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 \
  -e SCALE_SLIDING_WINDOW 2048 -e SCALE_STEPS 350 \
  -e SCALE_MOE_IMPL ragged_all_to_all -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger -e SCALE_JSON_LOGGER d67-control-m3-draw1-r1-0728-1538.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale --version d67-family-dev --run
```

- Next action: Submit the corrected command and monitor it to terminal.

### 2026-07-28 15:42 PDT - D67-CTL-01 parent setup OOM and recovery

- Job ID: `/mwittmann/d67-control-m3-draw1-r1-0728-1538`
- Result: Failed in parent `[iris setup] step 1/2` with task exit 137 while building the workspace under the default 1 GB driver allocation. No GPU child job was submitted.
- Interpretation: This is not a placement draw and supplies no experimental result. The CoreWeave grug runbook specifies a 2-CPU/3-GB launcher parent; use that allocation and otherwise leave the pre-registered arm unchanged.
- Recovery command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
  --cpu 2 --memory 3GB --extra cpu \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name d67-control-m3-draw1-r2-0728-1542 -e RUN_ID d67-control-m3-draw1-r2-0728-1542 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_GATHER_DISPATCH 1 -e SCALE_A2A_CUSTOM_ADJOINT 1 \
  -e SCALE_MOE_QB 1 -e SCALE_REPORT_DROPS 1 -e SCALE_A2A_SPILL 3 \
  -e SCALE_CAPACITY_FACTOR 1.0625 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 -e SCALE_INTERMEDIATE 1280 \
  -e SCALE_SHARED_INTERMEDIATE 5120 -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 \
  -e SCALE_SLIDING_WINDOW 2048 -e SCALE_STEPS 350 \
  -e SCALE_MOE_IMPL ragged_all_to_all -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger -e SCALE_JSON_LOGGER d67-control-m3-draw1-r2-0728-1542.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale --version d67-family-dev --run
```

- Next action: Submit the resource-corrected driver. Do not retry again if the same setup OOM repeats.

### 2026-07-28 15:45 PDT - D67-CTL-01 training launch

- Job ID: `/mwittmann/d67-control-m3-draw1-r2-0728-1542`
- Result: The 2-CPU/3-GB parent completed setup and launched child `/mwittmann/d67-control-m3-draw1-r2-0728-1542/grug-train-d67-control-m3-draw1-r2-0728-1542`. Both parent and child report `running`; no training step has been harvested yet.
- Interpretation: This is the first actual placement draw for D67-CTL-01.
- Next action: Monitor on a 570-second cadence through terminal state, then fetch the complete log and validate loss, drops, provenance, and run length.

### 2026-07-28 15:55 PDT - D67-CTL-01 gang setup

- Job ID: `/mwittmann/d67-control-m3-draw1-r2-0728-1542/grug-train-d67-control-m3-draw1-r2-0728-1542`
- Result: All 16 child tasks are `building`; failures=0 and preemptions=0. No training metric or error line has appeared.
- Interpretation: Normal gang setup. Do not resubmit or add another arm.
- Next action: Continue on the 570-second cadence.

### 2026-07-28 16:05 PDT - D67-CTL-01 Kueue capacity wait

- Job ID: `/mwittmann/d67-control-m3-draw1-r2-0728-1542/grug-train-d67-control-m3-draw1-r2-0728-1542`
- Result: Direct `cw-us-east-08a` task records show all 16 ranks in `SchedulingGated`: Kueue workload `iris-pg-14c6bf7cf45b04fa-0` is waiting for admission in `cw-use08a-lq`. Failures=0 and preemptions=0.
- Interpretation: Expected interactive-priority capacity wait. Pending/building is not a failed draw.
- Next action: Wait without resubmission.

### 2026-07-28 16:30 PDT - D67-CTL-01 scheduler-route correction

- Prior job ID: `/mwittmann/d67-control-m3-draw1-r2-0728-1542`
- Result: The parent and child were cancelled after the shared protocol identified two submission defects. The parent used the federated `--cluster=marin --target-cluster cw-us-east-08a` route, which does not place the child in the peer scheduler, and the child used the `interactive` priority band, which cannot preempt the occupying interactive pods. Both now report `killed`; neither reached rack allocation or produced an experimental result.
- Interpretation: The D67-CTL-01 prediction registered at 15:35 PDT remains unchanged and was registered before any result was seen. The corrected submission connects directly to `cw-us-east-08a`, omits `--target-cluster`, and uses `--priority production` as authorized in the revised shared protocol.
- Resubmission command:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
  --cpu 2 --memory 3GB --extra cpu --priority production \
  --job-name d67-control-m3-draw1-r3-0728-1630 -e RUN_ID d67-control-m3-draw1-r3-0728-1630 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_GATHER_DISPATCH 1 -e SCALE_A2A_CUSTOM_ADJOINT 1 \
  -e SCALE_MOE_QB 1 -e SCALE_REPORT_DROPS 1 -e SCALE_A2A_SPILL 3 \
  -e SCALE_CAPACITY_FACTOR 1.0625 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 -e SCALE_INTERMEDIATE 1280 \
  -e SCALE_SHARED_INTERMEDIATE 5120 -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 \
  -e SCALE_SLIDING_WINDOW 2048 -e SCALE_STEPS 350 \
  -e SCALE_MOE_IMPL ragged_all_to_all -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger -e SCALE_JSON_LOGGER d67-control-m3-draw1-r3-0728-1630.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale --version d67-family-dev --run
```

- Config: control; QB on; spill m=3; capacity factor 1.0625; trio off; 16 nodes × 4 GB200; 350 steps.
- DRI: mwittmann.
- Source: experiment code at `f53f781ce5358965b251dbf78d280dea9e053f73`; the pre-launch logbook commit changes documentation only.
- Output and retention: local JSON metrics named `d67-control-m3-draw1-r3-0728-1630.metrics`; checkpoints are disabled for this bounded benchmark.
- Monitoring: one owner, 15-minute steady-state cadence after the initial two-minute failure check.
- Next action: Commit this pre-launch record, submit this job only, verify it is in the direct peer queue with a real Kueue scheduling reason, and monitor it to terminal before draw 2.
