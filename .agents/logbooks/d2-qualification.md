---
topic: d2-qualification
description: Numerical, compile, PGLE, and three-draw qualification of the composed D-2 build.
author: Matt Wittmann
---

# D-2 qualification: Task logbook

## Scope

- Goal: qualify and submit the composed D-2 build without running D-4.
- Primary metrics: numerical deviation, compiled-layout evidence, SPMD warning count,
  PGLE coverage, tok/s, 2.5-PF/s MFU, matched-LR drop fraction, and loss.
- Constraints: fail-stop gates; one immutable code SHA; three sequential one-rack
  placement draws; do not contend with the concurrent D-6/D-7 rack family.

## Baseline

- Date: 2026-07-28
- Code ref: `0b305d520`
- Performance baseline: 20.708% MFU at `cf=1.0625`, spill `m=3`, with 1.44%
  drops over 349 samples.
- Additive prediction: 20.7% + 1.78pp padded Muon = approximately 22.5%.

## Decision log

- Numerical gate: relative L2 difference must be at most `2e-3`, cosine must be
  at least `0.99999`, NS2-to-NS5 relative-L2 growth must be at most `2x`, all
  values must be finite, and per-step loss relative divergence must be at most
  `1e-4`. These criteria were fixed before reading GPU results.
- D-2 falsification threshold: the composed gains are falsified if the median of
  the three placement-draw steady-tail p50 MFUs is below 21.5% at matched drop
  and LR position, or if the loss trajectory is unstable. This allows about
  1pp of the 22.5% additive prediction to fail to transfer.

## Entry log

### 2026-07-28 15:52 PDT - Handoff and queue audit

- Hypothesis: the composed build is eligible for GPU qualification, but not for
  rack submission until the numerical, compile, and PGLE gates pass.
- Commit Hash: `0b305d520`
- Command: `IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job list --prefix /mwittmann`
- Result: the concurrent `/mwittmann/d67-control-m3-draw1-r2-0728-1542` rack leg
  is running. `/mwittmann/deri-d8-ep32-c2-120-0728-1532` is also running.
- Interpretation: proceed only with the requested 4-GPU qualification work.
- Next action: build and snapshot the numerical/HLO probe.

### 2026-07-28 16:01 PDT - Numerical qualification submitted

- Hypothesis: the no-merge and pre-reconciliation layouts differ only by
  floating-point reduction order at D-2 matrix dimensions.
- Commit Hash: pending documentation-only command snapshot on top of
  `0e9bfb9f1df5c79bd0d3b13f49319af59c203b39`.
- Command:

  ```bash
  IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
    --target-cluster cw-us-east-08a --priority interactive \
    --job-name d2-muon-num-syrk1-0728-1601 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification --mode numerical --syrk 1
  ```

- Config: four GB200 GPUs; explicit `data=2, expert=2` mesh; FP32 arrays with
  BF16 Newton-Schulz internals; realistic 5120/1280 matrix dimensions; SYRK on.
- Result: pending.
- Next action: monitor to terminal and apply the pre-registered numerical gate.
