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

### 2026-07-28 16:46 PDT - Node-local retry authorized

- Hypothesis: unchanged. The no-merge and pre-reconciliation layouts differ
  only by floating-point reduction order at D-2 matrix dimensions. The
  numerical criteria and the 22.5% D-2 prediction with a 21.5% falsification
  threshold remain exactly as pre-registered above.
- Commit Hash: `8779abc42b1e7cc4e915d69a004ca4f6af909856`
- Correction: the failures on `s4bk6j84` do not establish a registry-wide image
  fault. Four other GB200 legs are currently running on `cw-us-east-08a`, and
  running and failed pods have overlapping `stage-workdir` image digests. Treat
  `exec format error` as node-local or intermittent. Record every failed node
  and stop after roughly a handful of fresh attempts if no task starts.
- Command:

  ```bash
  IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
    --priority production --job-name d2-muon-num-syrk1-r6-0728-1646 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification --mode numerical --syrk 1
  ```

- Config: four GB200 GPUs; direct `cw-us-east-08a` route; production priority;
  explicit `data=2, expert=2` mesh; FP32 arrays with BF16 Newton-Schulz
  internals; realistic 5120/1280 matrix dimensions; SYRK on.
- Queue audit: D-1a, D-1b, D-6/D-7 control, D-8, and the D-5 census have active
  jobs. This four-GPU qualification retry does not request rack capacity.
- Next action: submit once, verify direct Kueue admission, and monitor to
  terminal. Record the assigned node if startup fails.

### 2026-07-28 16:47 PDT - Node-local failure on retry R6

- Commit Hash: `8779abc42b1e7cc4e915d69a004ca4f6af909856`
- Job: `/mwittmann/d2-muon-num-syrk1-r6-0728-1646`.
- Result: direct Kueue admission succeeded, but `stage-workdir` failed before
  bundle fetch on node `s4bk6j84`. The init container again used
  `iris-task@sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`.
  Iris reported `Init:Error stage-workdir`; no `D2_ENV` or numerical output was
  produced.
- Interpretation: this is the same node-local/intermittent startup fault. Other
  jobs continue to run on the cluster, and recent unrelated attempts also fail
  when assigned to `s4bk6j84`. Do not patch the pod or change Iris.
- Next action: retry the identical four-GPU command once as
  `d2-muon-num-syrk1-r7-0728-1648`, then record its node and result.

### 2026-07-28 16:54 PDT - Numerical qualification passed on retry R7

- Commit Hash: `8779abc42b1e7cc4e915d69a004ca4f6af909856`
- Command:

  ```bash
  IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
    --priority production --job-name d2-muon-num-syrk1-r7-0728-1648 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification --mode numerical --syrk 1
  ```

- Job: `/mwittmann/d2-muon-num-syrk1-r7-0728-1648`.
- Node: `sjxsxs64`; `stage-workdir` completed with
  `iris-task@sha256:29ec7e8d4702faa36b0006ee34fd084e3e634a541e0736475446051bba091524`.
- Result: succeeded after 3 minutes 51.52 seconds. All 15 point comparisons
  across expert gate/up, expert down, non-expert tall, non-expert wide, and
  non-expert square orientations at NS0, NS2, and NS5 had `max_abs=0`,
  `mean_abs=0`, `relative_l2=0`, exact fraction `1.0`, and finite values.
  Cosines ranged from `0.9999998807907104` to `1.0000001192092896`.
- Loss trajectory: all 25 paired updates were exactly equal. The largest
  relative loss divergence was `1.627454638974007e-7` at expert-down step 3;
  every other step was `0`.
- Gate evaluation:
  - finite: pass;
  - relative L2 at every NS depth ≤ `2e-3`: pass (`0`);
  - cosine ≥ `0.99999`: pass;
  - NS5 relative L2 ≤ 2× NS2: pass (`0` and `0`);
  - per-step relative loss divergence ≤ `1e-4`: pass (`1.63e-7` maximum).
- Verdict: the composed and pre-reconciliation Muon paths are numerically
  equivalent at these realistic FP32/BF16 D-2 shapes. There is no structural
  divergence signal. Proceed to the compile/HLO smoke.
- Fresh startup failure nodes after the retry notice: `s4bk6j84` (R6). R7
  succeeded on `sjxsxs64`.

### 2026-07-28 16:55 PDT - Non-SYRK compile/HLO smoke submitted

- Hypothesis: with `SCALE_MUON_SYRK=0`, the GPU lowering preserves the
  no-merge expert layout, the two-hop multi-axis padded inbound reshard, and the
  direct padded outbound sharding without involuntary SPMD rematerialization.
- Commit Hash: `8779abc42b1e7cc4e915d69a004ca4f6af909856`
- Command:

  ```bash
  IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
    --priority production --job-name d2-muon-compile-syrk0-0728-1655 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification --mode compile --syrk 0
  ```

- Next action: monitor to terminal, count involuntary-remat SPMD warnings over
  the complete untruncated log, and inspect every `D2_STRUCTURE` and
  `D2_COMPILE` record before submitting the SYRK arm.

### 2026-07-28 16:57 PDT - Non-SYRK compile/HLO smoke failed the gate

- Job: `/mwittmann/d2-muon-compile-syrk0-0728-1655`.
- Result: Iris succeeded in 30.49 seconds, but the experimental gate failed.
  `D2_STRUCTURE` reported zero `(L,E)->LE` merges, padded reshards
  `P('data',None,None)` then `P(('data','expert'),None,None)`, and zero
  `P(None,None,None)` padded outbound reshards. Both compiled results were
  finite and restored their expected shardings.
- SPMD warning count: `1` over the complete log. While compiling
  `nonexpert_tall`, XLA reported involuntary full rematerialization from
  `{devices=[4,1,1]<=[4]}` to
  `{devices=[1,2,1,2]<=[4] last_tile_dim_replicate}` at
  `jit(current)/vmap()/convert_element_type`.
- Interpretation: the structural jaxpr checks pass, but the zero-warning
  requirement does not. A succeeded Iris job is not a passing compile gate.
- Next action: run the required SYRK compile arm to determine whether the first
  Blackwell SYRK lowering also compiles and whether it adds warnings. Stop
  before PGLE regardless of the SYRK result.

### 2026-07-28 16:57 PDT - SYRK compile/HLO smoke submitted

- Hypothesis: the new EP-capable SYRK branch compiles and executes on Blackwell,
  preserves the same structural layout, and does not add involuntary-remat
  warnings beyond the non-SYRK padded-path regression.
- Commit Hash: `8779abc42b1e7cc4e915d69a004ca4f6af909856`
- Command:

  ```bash
  IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
    --priority production --job-name d2-muon-compile-syrk1-0728-1657 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification --mode compile --syrk 1
  ```

- Next action: monitor to terminal and inspect the complete log. Do not submit
  PGLE capture or D-2 rack work after this arm.

### 2026-07-28 16:59 PDT - Compile/HLO gate failed; stop

- Job: `/mwittmann/d2-muon-compile-syrk1-0728-1657`.
- Node: `sdxsxs64` for both compile arms.
- Result: the first Blackwell EP SYRK lowering compiled and executed finite
  expert and padded non-expert results. `D2_STRUCTURE` again reported zero
  expert merges, the expected two padded inbound reshards, and zero replicated
  padded outbound reshards.
- SPMD warning count: `1` over the complete SYRK log, identical to the non-SYRK
  arm. The warning was the same padded `convert_element_type` transition from
  sharding `{devices=[4,1,1]<=[4]}` to
  `{devices=[1,2,1,2]<=[4] last_tile_dim_replicate}`.
- Compile summary:

  | `SCALE_MUON_SYRK` | Iris state | expert merges | inbound two-hop | replicated outbound | involuntary-remat warnings |
  |---:|---|---:|---|---:|---:|
  | 0 | succeeded | 0 | yes | 0 | 1 |
  | 1 | succeeded | 0 | yes | 0 | 1 |

- Verdict: the structural layout and the new SYRK GPU dispatch both compile,
  but the pre-registered zero-warning gate fails for both settings. The
  isolated Muon smoke failed before a full rematerialized training-step smoke
  was justified.
- Decision: stop. Do not capture PGLE or submit any D-2 rack draw. Record the
  numerical pass, compile failure, and unanswered composition question in the
  qualification report.

### 2026-07-28 21:17 PDT - Production-mesh discriminator pre-registration

- Hypothesis: the one-warning regression is specific to a mesh that shards the
  non-expert parameter matrix dimensions. With a leading-axis-only
  `data=1, expert=4` mesh, the padded outbound transition becomes a plain
  leading-axis all-gather and emits zero
  `spmd_partitioner.cc:668` involuntary-full-rematerialization warnings.
- Falsification threshold: any nonzero warning count in either
  `SCALE_MUON_SYRK=0` or `1` on `data=1, expert=4` refutes the hypothesis.
- Positive control: the same harness, cases, flags, and four GB200s must still
  emit one warning at `data=2, expert=2` for both SYRK settings in this session.
  A zero in the discriminator arm is uninterpretable without that control.
- One-variable rule: the four jobs differ only in the explicit mesh selector
  and `SCALE_MUON_SYRK`. All use the composed branch, the `nonexpert_tall` and
  expert compile cases, four GB200s, JAX from the same Iris environment,
  `cuda_async`, and the same two XLA flags.
- Planned commands:

  ```bash
  IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
    --target-cluster cw-us-east-08a --user mwittmann \
    --job-name d2-muon-mesh-d1e4-syrk0-0728-2120 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification \
      --mode compile --mesh data1-expert4 --syrk 0

  IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
    --target-cluster cw-us-east-08a --user mwittmann \
    --job-name d2-muon-mesh-d1e4-syrk1-0728-2120 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification \
      --mode compile --mesh data1-expert4 --syrk 1

  IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
    --target-cluster cw-us-east-08a --user mwittmann \
    --job-name d2-muon-mesh-d2e2-syrk0-0728-2120 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification \
      --mode compile --mesh data2-expert2 --syrk 0

  IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
    --target-cluster cw-us-east-08a --user mwittmann \
    --job-name d2-muon-mesh-d2e2-syrk1-0728-2120 \
    --enable-extra-resources --gpu GB200x4 --cpu 32 --memory 256GB --disk 256GB \
    --extra gpu --max-retries 0 --timeout 7200 \
    -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
    -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
    -- python -m experiments.grug.moe.d2_muon_qualification \
      --mode compile --mesh data2-expert2 --syrk 1
  ```
- Submission route: default federated `marin` route to `cw-us-east-08a` at
  default interactive priority. No cluster or node mutation is permitted.
- Next action: commit this pre-registration and mesh selector, then submit the
  four jobs in order and inspect complete, untruncated logs.

### 2026-07-28 21:21 PDT - Leading-axis-only non-SYRK discriminator passed

- Commit Hash: `12ce482bbeaf356e5cb0c24d0b52c1aa200b503c`
- Job: `/mwittmann/d2-muon-mesh-d1e4-syrk0-0728-2120`.
- Route: default-priority federated submission through `marin` to
  `cw-us-east-08a`; the peer accepted and ran the job without a persistent
  federation queue.
- Result: Iris succeeded. `D2_ENV` reported JAX 0.10.1 and the explicit
  `data=1, expert=4` mesh. `D2_STRUCTURE` reported zero expert merges, padded
  reshards `P('expert',None,None)` then `P(None,'data','model')`, and zero
  literal `P(None,None,None)` outbound reshards. Expert and `nonexpert_tall`
  outputs were finite and restored their requested sharding.
- SPMD warning count: `0` over the complete log fetched with
  `--max-lines 400000`.
- Interpretation: the non-SYRK discriminator matches the pre-registered
  prediction. The hypothesis is not accepted until SYRK also emits zero and
  both `data=2, expert=2` controls reproduce one warning.
- Next action: submit the otherwise identical `data=1, expert=4`,
  `SCALE_MUON_SYRK=1` arm.

### 2026-07-28 21:23 PDT - Leading-axis-only SYRK discriminator passed

- Commit Hash: `12ce482bbeaf356e5cb0c24d0b52c1aa200b503c`
- Job: `/mwittmann/d2-muon-mesh-d1e4-syrk1-0728-2120`.
- Result: Iris succeeded through the same default-priority federated route.
  `D2_ENV` reported JAX 0.10.1 and `data=1, expert=4`. The structural record
  matched the non-SYRK arm, and both compiled outputs were finite with the
  requested shardings.
- SPMD warning count: `0` over the complete log fetched with
  `--max-lines 400000`.
- Interpretation: both leading-axis-only arms meet the zero-warning prediction.
  The hypothesis still requires the two `data=2, expert=2` positive controls.
- Next action: rerun the original multi-axis mesh with SYRK off, changing only
  `--mesh data1-expert4` to `--mesh data2-expert2`.

### 2026-07-28 21:25 PDT - Multi-axis non-SYRK positive control reproduced

- Commit Hash: `12ce482bbeaf356e5cb0c24d0b52c1aa200b503c`
- Job: `/mwittmann/d2-muon-mesh-d2e2-syrk0-0728-2120`.
- Result: Iris succeeded through the default-priority federated route.
  `D2_STRUCTURE` reported the original inbound sequence
  `P('data',None,None)` then `P(('data','expert'),None,None)`, followed by the
  direct `P(None,'data','model')` outbound reshard. Both compiled outputs were
  finite.
- SPMD warning count: `1` over the complete log. The warning is the same
  `f32[1,5120,1280]` `jit(current)/vmap()/convert_element_type` transition from
  `{devices=[4,1,1]<=[4]}` to
  `{devices=[1,2,1,2]<=[4] last_tile_dim_replicate}`.
- Interpretation: the non-SYRK positive control reproduces the prior failure
  under the current harness and route. One SYRK positive control remains.
- Next action: submit the otherwise identical `data=2, expert=2`, SYRK-on arm.

### 2026-07-28 21:27 PDT - Federated route stalled on final control

- Job: `/mwittmann/d2-muon-mesh-d2e2-syrk1-0728-2120`.
- Result: the job remained pending for more than one normal compile-job
  scheduling interval with `Queued for peer cw-us-east-08a to report free
  capacity`. The first three arms had already traversed the federated route and
  run successfully.
- Interpretation: this matches the shared protocol's known federation failure
  mode, not a compile result.
- Next action: stop only this pending job and use the protocol's direct
  `--cluster=cw-us-east-08a` fallback at default priority. Do not alter any
  cluster, node, or other user's job.

### 2026-07-28 21:29 PDT - Mesh discriminator complete

- Commit Hash: `12ce482bbeaf356e5cb0c24d0b52c1aa200b503c`
- Final control job:
  `/mwittmann/d2-muon-mesh-d2e2-syrk1-direct-r1-0728-2127`.
- Recovery: stopped only the pending federated job
  `/mwittmann/d2-muon-mesh-d2e2-syrk1-0728-2120`, then resubmitted the same
  command through the protocol's direct `cw-us-east-08a` fallback at default
  priority. No shared infrastructure or other job was changed.
- Result: the direct control succeeded in 32.58 seconds with exit 0. It emitted
  one warning on the same `f32[1,5120,1280]` transition as the non-SYRK
  control. Both outputs were finite.
- Warning-count matrix:

  | mesh | `SCALE_MUON_SYRK=0` | `SCALE_MUON_SYRK=1` |
  |---|---:|---:|
  | `data=1, expert=4` | 0 | 0 |
  | `data=2, expert=2` | 1 | 1 |

- Verdict: the pre-registered mesh hypothesis is confirmed at the four-GB200
  discriminator scope. The warning is caused by the multi-axis destination
  layout, not by the leading-axis all-gather that remains when only `expert`
  has size greater than one.
- Replication limit: the `data=1, expert=4` jaxpr contains no literal
  `P(None,None,None)` outbound reshard, but its destination
  `P(None,'data','model')` is physically replicated because `data` and `model`
  both have size one. Zero involuntary-remat warnings do not establish lower
  peak memory or prove that `497423bc6` avoids physical replication at D-2.
  The conditional memory/HLO comparison was not triggered because the warning
  vanished.
- Decision: do not implement the outbound two-hop fix. The `data=2, expert=2`
  compile gate does not block D-2. Before any PGLE capture or placement draw,
  compile the exact composed build on the full-rack
  `replica_dcn=1, data=1, expert=64, model=1` mesh at both SYRK settings. Require
  zero `spmd_partitioner.cc:668` warnings in complete logs, finite realistic
  D-2 outputs, no expert `(L,E)->LE` merge, the single
  `P('expert',None,None)` padded inbound reshard, and restoration to the real
  parameter shardings. Any nonzero warning blocks D-2 and requires the outbound
  two-hop fix before rack draws.
