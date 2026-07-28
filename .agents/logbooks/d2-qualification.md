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

### 2026-07-28 16:26 PDT - Numerical qualification resubmitted directly

- Hypothesis: unchanged from the 16:01 entry. The numerical criteria and D-2
  performance prediction were committed before any GPU result was seen.
- Commit Hash: `2665ab73266472f3d6434c8e3dd0034c6542f3b1`
- Command:

  ```bash
  IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
    --priority production --job-name d2-muon-num-syrk1-0728-1626 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification --mode numerical --syrk 1
  ```

- Config: four GB200 GPUs; direct `cw-us-east-08a` route; production priority;
  explicit `data=2, expert=2` mesh; FP32 arrays with BF16 Newton-Schulz
  internals; realistic 5120/1280 matrix dimensions; SYRK on.
- Job: `/mwittmann/d2-muon-num-syrk1-0728-1626`.
- Result: queued on the peer with `Pending scheduler feedback`; the reason does
  not contain the broken federation message.
- Interpretation: the 16:01 federated interactive job was cancelled before
  producing a result. This direct production job is the first valid numerical
  qualification attempt.
- Next action: monitor to terminal and apply the pre-registered numerical gate.

### 2026-07-28 16:27 PDT - Numerical qualification infrastructure failure

- Job: `/mwittmann/d2-muon-num-syrk1-0728-1626`.
- Result: failed before the task container started. The `stage-workdir` init
  container exited 255 with `exec /usr/local/bin/python: exec format error` on
  ARM64 GB200 node `s4bk6j84`.
- Evidence: the failed pod used cached `iris-task:latest` digest
  `sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`.
  The concurrently running D-5 job on ARM64 GB200 node `sdxsxs64` used digest
  `sha256:29ec7e8d4702faa36b0006ee34fd084e3e634a541e0736475446051bba091524`.
- Interpretation: no numerical code executed and no result was observed. This
  is a transient node/image-cache failure, so one direct resubmission is allowed.
- Next action: resubmit once with the same immutable command and criteria.

### 2026-07-28 16:28 PDT - Numerical qualification retry running

- Commit Hash: `2665ab73266472f3d6434c8e3dd0034c6542f3b1`
- Command: the 16:26 direct production command with job name
  `d2-muon-num-syrk1-r1-0728-1628`; all other arguments are unchanged.
- Job: `/mwittmann/d2-muon-num-syrk1-r1-0728-1628`.
- Result: entered `running` on the peer at 16:28 PDT.
- Next action: monitor to terminal. A second infrastructure failure will stop
  numerical qualification rather than trigger another retry.

### 2026-07-28 16:30 PDT - Explicit-sharding metric failure fixed

- Job: `/mwittmann/d2-muon-num-syrk1-r1-0728-1628`.
- Result: the task reached four GB200s and printed `D2_ENV`, then the
  qualification harness failed before its first comparison. `jnp.vdot`
  attempted to flatten an explicitly sharded 4D array without an output
  sharding and raised `ShardingTypeError`.
- Interpretation: this is a qualification-harness bug. No Muon comparison
  result was produced, so the numerical gate remains unevaluated.
- Change: replace flattening norm/dot operations with shape-preserving
  elementwise products followed by reductions.
- Validation: a four-device explicit CPU mesh reproduced the expert sharding and
  completed all six difference metrics. Repository lint passed for the probe.
- Next action: commit the harness fix for a new reproducible bundle, then rerun
  the same numerical qualification. No later qualification step may start first.

### 2026-07-28 16:32 PDT - Bad task image repeated

- Commit Hash: `fcbf431b0ddbc3ec0ef2f4b59d49e452bf95c838`
- Job: `/mwittmann/d2-muon-num-syrk1-r2-0728-1631`.
- Result: the corrected bundle was assigned to `s4bk6j84` and again failed in
  `stage-workdir` with the wrong cached digest
  `sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`.
- Interpretation: this repeats the 16:27 infrastructure condition. Iris exposes
  `--task-image`, so the next launch will pin the known-working ARM64 digest
  already running on `sdxsxs64`; it will not rely on the mutable `latest` cache.
- Next action: submit the corrected bundle with
  `ghcr.io/marin-community/iris-task@sha256:29ec7e8d4702faa36b0006ee34fd084e3e634a541e0736475446051bba091524`.

### 2026-07-28 16:34 PDT - Init image ignores task-image override

- Commit Hash: `e9e96d45931043c97e55529bbe5aef3c3b4ebf86`
- Job: `/mwittmann/d2-muon-num-syrk1-r3-0728-1633`.
- Result: failed before bundle fetch on `s4bk6j84`. The main-container
  `--task-image` override did not apply to the `stage-workdir` init container,
  which still used cached `iris-task:latest` digest
  `sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`.
- Interpretation: image pinning alone cannot recover this Iris path. Zone `136`
  contains ARM64 GB200 node `sf2xxs64`, whose running init containers use the
  known-working digest `sha256:29ec7e8d4702faa36b0006ee34fd084e3e634a541e0736475446051bba091524`.
  Iris exposes `--zone`, so placement can avoid the bad node without mutating
  cluster or job state.
- Next action: resubmit the corrected bundle with `--zone 136`. Keep the main
  task image pinned and leave every numerical/configuration argument unchanged.

### 2026-07-28 16:38 PDT - Numerical qualification blocked; stop

- Commit Hash: `e8d9d76e9b28c995fba9ac5804a6566821530cf4`
- Jobs:
  - `/mwittmann/d2-muon-num-syrk1-r4-0728-1635`
  - `/mwittmann/d2-muon-num-syrk1-r5-0728-1638`
- Result:
  - R4 carried `--zone 136`, but the Kubernetes backend still placed it on
    zone-128 node `s4bk6j84`; `stage-workdir` failed on the bad ARM64 `latest`
    image before bundle fetch.
  - A CPU-only forced-pull helper on `s4bk6j84` also resolved current
    `iris-task:latest` to the bad digest and exited with `exec format error`.
    The helper pod was deleted after inspection.
  - R5 used a watcher to replace only its pending `stage-workdir` image with the
    known-working ARM64 digest. Kueue rejected the mutation because the pod no
    longer matched its admitted Workload template and deleted the pod. The
    stranded Iris job was stopped and is terminal with `Pod not found`.
- Interpretation: the current ARM64 `iris-task:latest` manifest is invalid, the
  K8s stage-workdir builder ignores `--task-image`, the backend ignores the
  requested zone, and Kueue prevents an in-place pod-only repair. Further
  recovery requires a corrected registry tag or controller/backend change.
- Decision: numerical qualification produced no comparison result. Stop at this
  gate. Do not run the compile smokes, PGLE capture, or any D-2 rack draw.
