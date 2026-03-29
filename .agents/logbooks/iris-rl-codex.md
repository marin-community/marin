# Iris RL Migration: Codex Logbook

Source: `.agents/logbooks/iris-rl.md`
Purpose: keep a concise, engineering-focused record of the Iris RL migration, including the latest root-cause analysis and fixes.

## Hard constraints (carried over)

1. All Iris TPU compute is preemptible/spot. Do not set `preemptible=False` for TPU jobs.
2. Child jobs inherit parent region. TPU child jobs must set explicit `regions=` to avoid inheriting CPU-only regions.

## Scope and target

- Goal: migrate RL from Fray v1 (Ray-era topology) to Fray v2/Iris coordinator topology.
- Primary success metric: `exp2039_rl_math500.py` runs end-to-end on Iris with Arrow Flight weight transfer.
- Constraints: no client-side orchestration; no RL-side backwards compatibility required; Arrow Flight first, JAX mode deferred.

## Baseline (from source logbook)

- Date: 2026-03-21
- Branch: `iris_rl` (based on `main` at `62ca4ad82`)
- Initial state: RL code and RL tests still contained Fray v1 coupling.

## Migration progress snapshot

| Step | Status | Notes |
|---|---|---|
| A: backend-independent operational fixes | DONE | watchdogs, timeouts, crash visibility, grading timeout, logging improvements |
| B: Fray v2 orchestration + runtime handles | DONE | coordinator topology, explicit runtime handles, `RLRunState`, v1 removed from core pipeline |
| B-10: RL test fixture migration to v2 | DONE | LocalClient fixtures and actor wiring |
| C-1: RunConfig drift fixes | DONE | retries/slices wiring landed |
| C-2: region support | PARTIALLY DONE | explicit TPU regions applied in orchestration path |
| D: local validation and integration fixes | DONE / IN PROGRESS | weight-transfer tests pass; some unrelated integration constraints remain |
| Iris end-to-end validation | DONE | first confirmed end-to-end Iris RL training loop succeeded |
| Restart robustness after trainer failure | DONE | rollback weight-id handling fixed (Codex, commit `3e9bf64da`) |

## Chronological record (relevant carry-over)

### 2026-03-21 — Step A operational fixes

Landed operational fixes from `on-demand-rl` in adapted form:
- rollout and trainer watchdog/phase logging
- cumulative trainer wait timeout
- stale dummy-weight guard
- top-level exception visibility
- SymPy grading timeout
- timing logs in vLLM/Arrow Flight paths

Recorded commits in source logbook:
- `db8ae206c`
- `5047b78a7`

### 2026-03-21 — Incorrect OOM fix reverted

`copy_and_flatten` OOM fix was reverted (`a134ea7a3`) because it moved tensors back to device via `jnp.asarray`, defeating host-only intent.

### 2026-03-21 — Step B: Fray v2 migration

Core architecture changes:
- Added `run_state.py` with `RLRunState`
- Added `orchestration.py` with v2 coordinator submission and child-job creation
- Workers updated to explicit runtime handles
- Curriculum actor discovery de-globalized
- Arrow Flight coordinator handle passed explicitly

Key commits recorded:
- `a1080fdb3`
- `3984f3793` (drop v1 from core pipeline)
- `5a5edf949` (tests migrated to v2 fixtures)

### 2026-03-21 — test/fix cycle

Main findings fixed during validation:
- bfloat16 serialization compatibility
- hostname/address resolution for gRPC
- runtime/circular import cleanup (`runtime.py` extraction)
- stale API references in tests

Follow-up graceful-shutdown and integration fixes were also applied (`a7d1eb7e1`), including rollout-side run-state checks.

### 2026-03-21 — Codex review feedback applied

Source logbook captured review findings (tracker fallback, run-state recheck timing, SymPy spawn context, outdated callers), with fixes consolidated in commit `84cbb2528`.

### 2026-03-21 to 2026-03-22 — Iris monitoring and capacity phases

Multiple launch attempts exposed infra and packaging pitfalls:
- missing bundled files
- `vllm` dependency path mismatches
- CPU coordinator slot pressure
- TPU pool saturation

Important resolved blockers:
- hosted control-plane actors via `host_actor()` (commit `46f1da861`)
- decoupled config from `vllm.SamplingParams` via `VLLMSamplingConfig` (commit `59e2f7c95`) so coordinator can run on CPU

### 2026-03-22 — root-cause infra misunderstandings fixed

Two high-impact fixes from source logbook:
- `preemptible=False` mismatch fixed (`dd1f9402d`)
- child-region inheritance issue fixed via explicit TPU regions (`c938fa0f3`)

### 2026-03-22 ~19:00 UTC — first end-to-end success on Iris

Source logbook records full loop validated:
- trainer progressed with decreasing loss
- rollout worker generated rollouts and wrote storage
- Arrow Flight transferred ~15GB weights cross-host
- hosted actors resolved correctly from TPU workers
- phase transitions visible and healthy

## Codex deep-dive: stale rollout rejection after restart

### Investigation context (from `docs/debug-log-iris-rl-step-skew.md`)

Debug target:
- explain repeated `Skipping stale rollout batch (rollout_step=1, current_step=0)` in 500-step Iris run.

Code paths inspected:
- `lib/marin/src/marin/rl/replay_buffer.py`
- `lib/marin/src/marin/rl/train_worker.py`
- `lib/marin/src/marin/rl/rollout_worker.py`
- `lib/marin/src/marin/rl/orchestration.py`

Jobs/logs inspected:
- `/ahmed/iris-rl-500step-v2/rl-iris-rl-direct-20260323-025037-train`
- `/ahmed/iris-rl-500step-v2/rl-iris-rl-direct-20260323-025037-rollout-0`

Hypothesis outcomes:
- Hypothesis 1 (steady-state off-by-one): rejected.
  - rollout stamp source is transferred `weight_id`, not rollout loop index.
  - trainer `current_step` is updated from Levanter `StepInfo.step` (completed step semantics).
- Hypothesis 2 (restart skew): confirmed.
  - trainer progressed to step 1, then was OOM-killed and retried.
  - rollout worker continued serving/generating from the newer lineage.
- Hypothesis 3 (coordinator stale guard blocks rollback): confirmed.
  - coordinator rejected rollback updates (`-1`, `0`) while holding `1`.

### Symptom seen

Trainer repeatedly logged:
- `Skipping stale rollout batch (rollout_step=1, current_step=0)`

### What actually happened (step-by-step)

1. Trainer reached step 1 and served weight id `1`.
2. Trainer process was OOM-killed and retried.
3. After restart, trainer attempted to re-serve rollback ids (`-1`, then `0`).
4. Arrow Flight coordinator rejected rollback ids because it treated any lower id as stale.
5. Rollout worker stayed pinned to coordinator-advertised weight id `1` and kept producing `rollout_step=1`.
6. Restarted trainer had `current_step=0`, so replay buffer correctly rejected those as future rollouts.
7. Result: apparent livelock.

Evidence checkpoints from logs:
- trainer step progress seen up to transfer/complete at step 1 (~03:11 UTC).
- trainer crash recorded as `Container was OOM killed by the kernel` (~03:18 UTC).
- restart path showed missing checkpoint recovery (`No checkpoints found ...`), then re-serve attempts.
- coordinator emitted rollback rejection warnings (`Ignoring stale weight update: -1 < 1`, `0 < 1`).

### Why the temporary replay-buffer ±1 patch was not the right fix

Allowing future rollouts (`current_step + delay`) masks restart skew and can mix lineages after rollback.

### Correct fix landed

Commit: `3e9bf64da`

Changes:
- `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
  - coordinator now accepts rollback updates and only ignores exact duplicate ids.
- `lib/marin/src/marin/rl/replay_buffer.py`
  - restored strict no-future-rollout semantics (`max_step = current_step`).
- `tests/rl/test_weight_transfer.py`
  - added regression test `test_arrow_flight_coordinator_accepts_rollback_weight_ids`.

Validation run:
- `./infra/pre-commit.py --fix -- ...` passed.
- `uv run pytest -q tests/rl/test_weight_transfer.py -k "multiple_weight_updates or coordinator_accepts_rollback"` passed (`3 passed`).

Supporting debug log:
- `docs/debug-log-iris-rl-step-skew.md`

## Operational note

At user request, run `/ahmed/iris-rl-500step-v3` and its child jobs were terminated.

## Remaining follow-ups

1. Decide failure policy for trainer child jobs: fail-fast vs retry-with-resume guarantees.
2. Improve checkpoint reliability/atomicity for retry safety.
3. Keep tracking dependency hygiene around `vllm-tpu` install behavior on Iris workers.
4. Re-run long Iris validation with rollback fix in place and monitor recovery behavior under forced failure.

## 2026-03-23 — v4 region/capacity clarification

Snapshot time:
- 2026-03-23T05:38:50Z (`job list --prefix /ahmed/iris-rl-500step-v4 --json`)

Observed live states:
- `/ahmed/iris-rl-500step-v4`: `JOB_STATE_RUNNING` (parent launcher, 0.5 CPU / 1 GiB).
- `/ahmed/iris-rl-500step-v4/rl-llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-041738`: `JOB_STATE_PENDING` (executor step, 1 CPU / 4 GiB).
- Pending reason: `Insufficient CPU ... no_capacity: cpu_vm_e2_highmem_2_ondemand-europe-west4-a=at_max_slices`.

Region behavior, broken down by layer:
1. Launcher layer (parent job): running on Europe CPU worker
   - worker id includes `...ondemand-europ...`.
   - logs show metadata/output under `gs://marin-eu-west4/...`.
2. Executor-step layer (immediate child): same Europe region constraint
   - also blocked on Europe CPU pool capacity.
3. RL TPU layer (train/rollout children created by RL coordinator): not reached yet
   - these request explicit TPU regions `us-central1` and `us-east5` in `orchestration.py`.
   - they are only submitted after the pending executor step starts.

Conclusion at this snapshot:
- v4 is not terminal-failed; it is capacity-blocked in Europe on the CPU executor step.
- Parent and immediate child are region-aligned (both Europe), so the mismatch concern applies only across layers, not between those two jobs.

## 2026-03-23 — placement rule update from live triage

User directive established during incident handling:
- RL runs for this thread must be launched only with placement compatible with `v5p-8` locality:
  - `us-central1` (preferred), or
  - `us-east5-a` when explicitly zoned and available.

Operational consequence:
- Do not launch parent/executor in `europe-west4` or `us-east1` for this RL flow.
- Use explicit submit constraints on launch:
  - `--region us-central1` for the root job by default.
  - optionally `--zone us-east5-a` if switching to east5 path.

Actions taken:
- Stopped `/ahmed/iris-rl-500step-v6` on request.
- Verified terminal state: `JOB_STATE_KILLED` with `Terminated by user`.

## 2026-03-23 — v7 relaunch plan (pre-execution)

Context update from user:
- Continue this thread with active babysitting/monitoring (target: 8h total monitoring window).
- Placement must stay compatible with `v5p-8` path:
  - allow `us-central1` (preferred),
  - or `us-east5-a` if explicitly requested for launcher placement.

Current state before relaunch:
- No active `/ahmed/iris-rl-500step-*` job remains running.
- Most recent jobs (`v6`, `v5`, `v4`) are all `JOB_STATE_KILLED` by user action.

Planned submit command:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait \
  --user ahmed \
  --job-name iris-rl-500step-v7 \
  --region us-central1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -e MARIN_PREFIX gs://marin-us-central1 \
  -- uv run python experiments/exp2039_rl_math500.py
```

Monitoring plan:
- Apply Iris babysit cadence with startup stabilization then periodic checks.
- Record status transitions, pending reasons, failures, and W&B link once emitted.

### Execution result (startup window)

Submit time:
- 2026-03-23T05:58:16Z

Submitted job:
- `/ahmed/iris-rl-500step-v7`

Observed after startup stabilization (~120s):
- Parent job `/ahmed/iris-rl-500step-v7`: `JOB_STATE_RUNNING`.
- Executor child `/ahmed/iris-rl-500step-v7/rl-llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-060031`: `JOB_STATE_RUNNING` (building/startup).
- Worker placed on `...cpu_vm_e2_highmem_2_ondemand-us-ce...` (us-central1 path).
- Executor metadata/output path now in US bucket:
  - `gs://marin-us-central1/experiments/exp2039_rl_math500-d5ac78.json`
  - `gs://marin-us-central1/rl_testing/llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-060031-e3a571`

Interpretation:
- Region-pinning fix is working for launcher/executor layer.
- No immediate startup failure in the stabilization window.

Immediate next action:
- Continue long-cadence monitoring (570s) and capture:
  - RL child train/rollout submission,
  - TPU allocation state,
  - first W&B run link,
  - any retry/failure signatures.

## 2026-03-23 — v7 failure and recovery plan

Failure detected on first long-cadence check:
- Parent `/ahmed/iris-rl-500step-v7`: `JOB_STATE_FAILED`.
- Child `/ahmed/iris-rl-500step-v7/rl-llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-060031`: `JOB_STATE_FAILED`.

Primary error signature:
- Container build failed while installing `vllm-tpu` transitive deps:
  - `nvidia-cublas-cu12==12.8.4.1`
  - `No space left on device (os error 28)` writing under `/uv/cache/...`

Observed child resource shape at failure time:
- CPU step requested `1 cpu / 4 GiB / 16 GiB disk`.

Interpretation:
- This is not TPU scheduling failure yet; it fails earlier in CPU executor-child container build due insufficient disk headroom for dependency extraction.

Recovery change (code):
- Updated `lib/marin/src/marin/rl/rl_experiment_utils.py` so RL executor steps run as remote callables with explicit larger CPU disk:
  - `ResourceConfig.with_cpu(cpu=1, ram="4g", disk="64g")`

Planned resubmit (v8):
- same region policy (`us-central1`), same env overrides, fresh job name.
- continue babysit loop and monitor for repeat build failure vs progression to RL child TPU submissions.

### v8 result

Outcome:
- Failed again with `No space left on device` while extracting CUDA wheels for `vllm-tpu` transitive torch deps.

Important hierarchy finding:
1. Outer executor-step remote job honored the new `64g` disk request.
2. Inner RL coordinator job spawned by `submit_rl_job()` still used default CPU disk (`16g`), and that inner job failed.

Evidence:
- intermediate job had `disk_bytes=68719476736` (64 GiB).
- failing inner RL job still had `disk_bytes=17179869184` (16 GiB).

Next recovery change:
- update `lib/marin/src/marin/rl/orchestration.py` `submit_rl_job()` resources to request larger coordinator CPU disk.
- set coordinator disk to `100g` and relaunch (`v9`) with same region policy.

### v9 result

Outcome:
- Failed again with `No space left on device`, even after increasing:
  - intermediate executor step disk to `64g`,
  - inner RL coordinator disk to `100g`.

New key evidence:
- failing nested RL job shows `disk_bytes=107374182400` (100 GiB) and still hits
  `nvidia-*` wheel extraction failures under `/uv/cache/.tmp...`.

Inference:
- the root issue is dependency selection path, not only disk size.
- `pip_packages=["vllm-tpu==..."]` path is pulling CUDA transitive wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) on CPU/coordinator build layers.

Recovery change (v10 plan):
- switch orchestration environments from `pip_packages` to `extras`.
- force inclusion of `tpu` extra (plus configured dependency groups), so uv source rules route torch/torchvision to CPU wheels for TPU/CPU builds.
- relaunch with same region constraints and continue monitor loop.

### v10 result

Outcome:
- Nested RL job still failed on build with disk error, now on:
  - `libtpu==0.0.24` extraction (`No space left on device`),
  - despite nested job resource showing `100 GiB` disk.

Interpretation:
- Environment layering still installs large runtime artifacts too early and too broadly.
- Coordinator should not install TPU/vLLM stack; those belong to execution workers only.

Recovery change (v11 plan):
- split orchestration dependency environments by role:
  - coordinator extras: minimal (`config extras` minus `vllm`/`tpu`),
  - trainer extras: `tpu` + non-vLLM extras,
  - rollout extras: `tpu` + `vllm` + configured extras.
- relaunch and continue long-cadence monitoring.

### v11 startup result (first stabilization check)

Status:
- `/ahmed/iris-rl-500step-v11`: `JOB_STATE_RUNNING`
- nested RL coordinator chain is alive (no immediate build-fail collapse)
- trainer child is running on `v5p-8` (`JOB_STATE_RUNNING`)
- rollout child is pending TPU capacity in `us-east5-a`:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - `Autoscaler: Waiting for workers in scale group 'tpu_v5p_8-us-east5-a'`

Most important delta vs v7/v8/v9/v10:
- no early nested `No space left on device` build failure in the first window.
- role-split env strategy appears to have bypassed the prior install bottleneck.

Immediate next action:
- continue 570s cadence and watch for:
  - rollout TPU allocation transition to running,
  - first W&B link emission,
  - trainer progress logs and step advancement.

### v11 follow-up (first long-cadence probe)

Status transition:
- rollout TPU worker moved from `PENDING` to `RUNNING`.
- trainer TPU worker remains `RUNNING`.
- no disk-build failures observed after transition.

W&B runs observed:
- train: `https://wandb.ai/marin-community/marin_post_training/runs/llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-064750-train`
- rollout: `https://wandb.ai/marin-community/marin_post_training/runs/llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-064750-rollout-0`

Current rollout/trainer signal:
- rollout generation active at `weight_step=-1` with batches emitted.
- trainer has started runtime and weight server initialization; continued cadence needed to confirm first step>=0 training advancement.

### v11 follow-up (second probe, active training/rollout)

Probe time:
- 2026-03-23 00:11 PT (`07:11 UTC` logs window).

Tree status:
- `/ahmed/iris-rl-500step-v11`: `JOB_STATE_RUNNING`.
- Nested coordinator + train + rollout chain remain alive in `us-central1`.
- Train/rollout workers still allocated as `v5p-8` workers.

Trainer signal (strong progress):
- completed train steps `0, 1, 2, 3` with successful weight transfers (`weight_id=0..3`).
- latest visible progress line:
  - `Progress on:train 4.00it/500it` (at `07:08:51 UTC`).
- intermittently logs `No rollouts received ... retrying` at 60s/120s waits, but receives new batches and continues.

Rollout signal:
- actively generates batches and writes rollout files to GCS.
- observed lines include:
  - `Generated rollout ... at step 2`
  - `PHASE: WRITE_ROLLOUT step=3`
  - `PHASE: IDLE step=4 ... (weight_step=2)`
  - `Generated rollout with 500 groups ... at step 2`.
- this confirms rollout internal loop step can be ahead of current `weight_step`; they are distinct counters.

Failure/regression check:
- no new `No space left on device` build failures in this probe.
- no stale-rollout rejection signature observed in this log window.

## 2026-03-23 — v11 failure after rollout runtime, auto-restart

Failure observed during babysit cadence:
- `/ahmed/iris-rl-500step-v11` transitioned to `JOB_STATE_FAILED`.
- failing subtree: `/.../rl-llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-064750` became `JOB_STATE_KILLED`.
- coordinator traceback propagated:
  - `fray.v2.client.JobFailed: ... rollout-0 finished with status stopped`.

Primary error signature in child logs (rollout runtime):
- repeated XLA scoped-VMEM pressure in vLLM attention path:
  - `memory_space_assignment_util.cc: INVALID_ARGUMENT`
  - `104857600 bytes of scoped Vmem requested ... max valid bytes is 67043328`
  - backend config references `ragged_paged_attention.*`.

Interpretation:
- this is not the earlier disk-install failure class.
- this is runtime TPU/XLA/vLLM memory-space pressure causing rollout worker stop; parent fails via wait-all propagation.

Babysit action taken:
- auto-resubmitted using existing state command and same canonical job id:
  - `/ahmed/iris-rl-500step-v11`
- command keeps region policy at `us-central1` and `MARIN_PREFIX=gs://marin-us-central1`.

Immediate post-resubmit startup check (120s):
- job is currently `JOB_STATE_PENDING`, not failed.
- pending reason is scheduler CPU capacity:
  - `Unsatisfied autoscaler demand: ... cpu_vm_e2_highmem_2_ondemand-us-central1-a=at_max_slices`.
- no new traceback/error lines in first 300s window after resubmit.

Next action:
- continue monitoring on cadence; if pending persists, keep waiting unless user requests region shift to another allowed v5p-8 region (`us-east5-a`).

### v11 restart follow-up (capacity cleared, run active again)

Probe outcome after one full cadence:
- scheduler capacity recovered; parent job moved from pending to running.
- `/ahmed/iris-rl-500step-v11`: `JOB_STATE_RUNNING`.
- new nested subtree is active:
  - `.../rl_testing-...-073345_f8bd418d-526c5471/...-train`: running on `v5p-8`.
  - `.../rl_testing-...-073345_f8bd418d-526c5471/...-rollout-0`: running on `v5p-8`.

New W&B runs:
- train: `https://wandb.ai/marin-community/marin_post_training/runs/llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-073345-train`
- rollout: `https://wandb.ai/marin-community/marin_post_training/runs/llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-073345-rollout-0`

Current runtime signal:
- trainer is in startup synchronization (`Waiting for initial rollouts from step -1`, elapsed climbing from 0s to 60s).
- no fresh scoped-VMEM/XLA crash lines in this 900s window.
- only known recurring warning is `Failed to import from vllm._C`, which has been non-fatal in successful starts.

Next action:
- keep cadence monitoring and specifically watch for recurrence of:
  - `memory_space_assignment_util.cc ... scoped Vmem ... ragged_paged_attention.*`
  - rollout worker stop / coordinator `JobFailed` propagation.

### v11 subsequent cadence (degraded signal, no terminal failure yet)

Status:
- full tree remains `JOB_STATE_RUNNING` (parent/coordinator/train/rollout all running).
- train and rollout TPU workers continue on `v5p-8`.

Progress:
- trainer reached step 1+ (visible `2.00it/500it`), with:
  - step 0 complete (`loss=-0.0002`),
  - step 1 complete (`loss=-0.0002`).
- rollout kept generating/writing batches across weight steps (`-1`, then `0`, then `1`).

Recurring risk signal:
- scoped-VMEM warnings reappeared repeatedly:
  - `memory_space_assignment_util.cc: INVALID_ARGUMENT`
  - `104857600 bytes ... max valid bytes is 67043328`
  - `ragged_paged_attention.*`
- unlike prior failed run, this cadence did not end in immediate rollout stop; run remained alive and progressed.

Interpretation:
- currently a degraded but running state.
- the scoped-VMEM condition is likely a high-risk precursor rather than an instant-fail every time.

Next action:
- continue babysit loop; trigger immediate recovery only if job transitions to terminal non-success or rollout worker stops again.

### v11 in-run failure/retry (train OOM, auto-recovered to running)

New event in later cadence window:
- train worker emitted:
  - `Container was OOM killed by the kernel` at `07:59:11`.
- Iris did not terminate the parent job; instead train child auto-retried.
- current status still shows whole tree `JOB_STATE_RUNNING`, but train child has `failure_count=1`.

Post-OOM behavior:
- train child restarted runtime in-place:
  - re-created Arrow Flight servers,
  - re-connected curriculum actor,
  - reloaded base model from `meta-llama/Llama-3.1-8B-Instruct`,
  - resumed startup sync path.
- rollout child remained alive and kept generating/writing rollouts while polling for new weights (`polling for step > 3`).

Interpretation:
- this is a soft failure with automatic worker retry, not a terminal job failure.
- short-term risk remains elevated because we now have both:
  - recurring scoped-VMEM warnings on rollout side,
  - one confirmed train-container OOM restart.

Next action:
- keep babysit loop active; if train OOM repeats or rollout stops again, escalate to active recovery (stop/resubmit) and consider config-level mitigation for memory pressure.

### v11 post-OOM recovery confirmation (same run, continued progress)

Follow-up cadence confirms auto-retry worked end-to-end:
- train child remained `JOB_STATE_RUNNING` with `failure_count=1` (one prior OOM retry).
- no parent/coordinator terminal failure occurred.

Observed timeline after OOM:
- train restarted and reloaded base model (`Loading initial model from checkpoint: meta-llama/Llama-3.1-8B-Instruct`).
- trainer resumed loop from step indexing restart and made fresh progress:
  - completed step 0,
  - completed step 1,
  - completed step 2.
- rollout stayed alive throughout and continued generating/writing batches, advancing its local loop (`PHASE: WRITE_ROLLOUT step=10` observed).

Current interpretation:
- run is in a recovering-but-fragile state.
- known high-risk signals remain:
  - recurring scoped-VMEM warnings in rollout path,
  - one confirmed train OOM event (auto-retried successfully so far).

Next action:
- continue monitoring; treat any second OOM/restart loop or rollout stop as escalation trigger for active mitigation/resubmission decision.

### v11 repeated OOM churn (second train OOM, auto-retry again)

New cadence finding:
- second train-container OOM occurred:
  - `Container was OOM killed by the kernel` at `08:20:45`.
- train child remained in `JOB_STATE_RUNNING` but `failure_count` increased to `2`.
- parent/coordinator/rollout remained non-terminal (`JOB_STATE_RUNNING`).

Behavior after second OOM:
- train worker restarted again and re-entered startup path:
  - `Loading initial model from checkpoint: meta-llama/Llama-3.1-8B-Instruct`
  - `Progress on:train -/500 ...`
  - `Received initial rollouts! Buffer size: 2048`.
- rollout kept running and continued generating batches (`PHASE: WRITE_ROLLOUT step 11-13` observed).

Interpretation:
- system is alive but unstable for sustained progress due repeated train OOM resets.
- this has crossed from isolated incident into recurring memory-pressure churn.

Next action:
- continue babysit monitoring per user request.
- if OOM recurs again, escalate from passive monitoring to proposing concrete memory mitigation (or controlled stop/resubmit with adjusted config) because repeated restarts will erase effective throughput.

### v11 stability check after second OOM (no third OOM yet)

Most recent cadence:
- job tree remains `JOB_STATE_RUNNING` end-to-end.
- train child still shows `failure_count=2` (reflecting the two OOM restarts), but no additional increment in this window.

Current progress after second restart:
- train resumed and climbed back through early steps:
  - step 0, 1, 2, then 3 completed in this window.
- rollout remained active and continued write phases (`step=14..17` observed in rollout loop).

Risk posture:
- still degraded due prior repeated OOM resets.
- however, immediate behavior is currently forward-progressing rather than crash-looping.

Next action:
- continue babysit cadence; trigger escalation only on:
  - third OOM/restart increment,
  - rollout stop/terminal state propagation,
  - parent/coordinator terminal transition.

## 2026-03-23 — v11 terminal collapse and recovery to v12

Observed terminalization on v11 during subsequent monitoring:
- rollout child moved to `JOB_STATE_FAILED` with:
  - `RuntimeError: Timed out waiting for initial weight transfer.`
- train child moved to `JOB_STATE_FAILED` with:
  - `Exit code 137: OOM killed (container exceeded memory limit).`
- parent wrapper lingered in running state briefly despite failed children.

Recovery actions executed:
1. issued explicit stop for `/ahmed/iris-rl-500step-v11` (and subtree terminated).
2. attempted same-name relaunch (`v11`) but encountered stop/cleanup race:
   - one submit rejected as still running,
   - subsequent submits immediately ended `JOB_STATE_KILLED` (`Terminated by user`) with no useful startup.
3. switched to fresh job name `iris-rl-500step-v12` in same region/prefix policy.

v12 startup status:
- `/ahmed/iris-rl-500step-v12` is running.
- nested coordinator chain also running (`/.../rl_testing-...-153458...`).
- no fresh error/traceback/OOM signatures in first 120s startup window.

Operational note:
- v11 should be considered retired due repeated train OOM churn + rollout weight-transfer timeout + name-level restart instability after manual stop.
- active monitoring now tracks v12.

## 2026-03-23 — v12 moved past startup stall and completed first train step

Latest monitoring window confirms v12 is making real RL loop progress (not just process liveness):
- full tree remains `JOB_STATE_RUNNING` (parent/coordinator/train/rollout).
- trainer moved from long initial wait to active training:
  - `Received initial rollouts! Buffer size: 1024 (waited 325.0s)`
  - `Training step 0 completed: duration=92.03s ...`
  - `Progress on:train 1.00it/500it ...`
  - `Transferring weights at step 0 ...` and successful weight serve.
- rollout consumed served weights and advanced generation/write cycle:
  - `Received new weights from step 0`
  - `Generated rollout ... at step 0`
  - `PHASE: WRITE_ROLLOUT step=1`
  - replay buffer confirmed ingestion: `Collected 1 rollout batches ...`

Important interpretation:
- this resolves the immediate startup symptom (`Still waiting for initial rollouts`), which persisted for ~5+ minutes before first usable batch arrived.
- no fresh terminal signatures observed in this checkpoint (`OOM`, `RuntimeError`, `JOB_STATE_FAILED`, `killed`, `Timed out` absent in latest filtered snapshot).

Next action:
- continue babysit cadence and watch for recurrence of the prior failure modes (train OOM churn or rollout weight-transfer stall).

## 2026-03-23 — v12 step-1 confirmation (post-recovery continuity)

Follow-up cadence (about a minute later) confirms continued forward progress:
- train advanced to step 1:
  - `Progress on:train 2.00it/500it ...`
  - `Transferring weights at step 1 ...`
  - `Training step 1 completed ...`
- rollout consumed updated trainer weights:
  - `Received new weights from step 1`
  - rollout generation continues at weight-aligned step (`Generated rollout ... at step 0` prior to receiving step-1 weights).
- no fresh signatures of terminal failure in this window (`RuntimeError`, `OOM`, `killed`, `Timed out`, `JOB_STATE_FAILED` absent in filtered snapshot).

Interpretation:
- the v12 run is no longer in a startup-stall regime; trainer/rollout handshake is active through at least step 1.

Next action:
- continue cadence monitoring and only intervene on concrete regression (terminal state, repeated OOM, or rollout-transfer stall).

## 2026-03-23 — v12 history analysis for cluster-admin feedback

Objective:
- explain why prior 5-step run looked robust while current 500-step run is restart-churning.

### Concrete timeline (v12)

Observed from v12 logs:
- checkpoint save starts, then train OOM follows shortly after:
  - `15:59:55` save checkpoint step 4 -> `16:00:40` OOM (`+45s`)
  - `16:16:25` save checkpoint step 5 -> `16:17:15` OOM (`+50s`)
  - `16:35:33` save checkpoint step 5 -> `16:36:20` OOM (`+47s`)
- after each OOM, trainer retries and cold-starts:
  - `Loading initial model from checkpoint: meta-llama/Llama-3.1-8B-Instruct`
- checkpoint recovery repeatedly fails after crash:
  - `No checkpoints found ...`
  - path exists but `does not contain a metadata.json` (incomplete checkpoint layout).
- with cold restart at trainer `current_step=0`, replay buffer initially rejects rollout batches from older in-flight rollout lineage:
  - repeated `Skipping stale rollout batch (rollout_step=1..4, current_step=0)`.
- run eventually restabilizes each time (receives initial rollouts, resumes step 0/1/2...), then repeats OOM pattern.

Current status snapshot:
- job tree still `JOB_STATE_RUNNING`.
- train child `failure_count=3` (three OOM retries so far).

### Why 5-step looked much more robust

Direct comparison with `/ahmed/iris-rl-v5p-3` logs:
- 5-step run used `max_weight_transfer_wait_time=300` (blocking sync), while v12 uses `max_weight_transfer_wait_time=0` (non-blocking).
- 5-step run completed quickly (`5/5`) and showed no train OOM line.
- 5-step logs did not show periodic trainer checkpoint-save churn; v12 repeatedly hits save + OOM + retry + incomplete-recovery loop.

Interpretation:
- 5-step run likely avoided this failure regime primarily by shorter duration / fewer cycles and different sync behavior.
- v12 failure pattern is dominated by repeated train-side OOM around checkpoint-save periods, not a one-off stale-rollout bug.

### Suggested feedback for cluster admin

High-signal points to share:
- train container is being kernel-OOM-killed repeatedly on v5p-8 worker despite auto-retry.
- OOMs are temporally clustered right after checkpoint save starts (three occurrences with ~45-50s lag).
- checkpoint path is left in incomplete state (missing `metadata.json`) after OOM, so retries cannot resume and must cold-start.
- this leads to repeated wasted compute and replay-step skew windows.

Data to request from cluster side:
- cgroup/container memory high-water marks and OOM killer details for the train task at the three timestamps above.
- host-level memory pressure logs around checkpoint save windows.
- any storage / fs-side errors during checkpoint finalization that could explain missing metadata manifests.

## 2026-03-23 — W&B-grounded 5-step settings + updated OOM experiment matrix

User requested explicit run-history grounding from:
- train: `marin-community/marin_iris_rl_debug/iris-rl-direct-20260323-013808-train`
- rollout: `marin-community/marin_iris_rl_debug/iris-rl-direct-20260323-013808-rollout-0`

### What the 5-step run actually did

From W&B run config (`iris-rl-direct-20260323-013808-train`):
- `num_train_steps=5`
- `train_batch_size=64`
- `per_device_parallelism=16`
- `checkpointer.save_interval=0:10:00`
- `tracker.project=marin_iris_rl_debug`

From train/rollout Iris logs:
- weight transfer used blocking wait:
  - `max_weight_transfer_wait_time=300`
- vLLM inference envelope:
  - `max_model_len=1024`
  - generation calls at `max_tokens=512`
- rollout payload shape:
  - training-phase rollouts: `Generated rollout with 4 groups ...`
  - eval-phase calls: `generate: starting, 10 prompts ...`
- W&B rollout summary (`...rollout-0`):
  - `inference.rollout/math_full/total_count=64`
  - `inference.rollout_storage/last_size_bytes=329884`
  - `inference.rollout_storage/cumulative_batch_count=5`

Notable:
- no `Saving checkpoint at step ...` log line observed in this 5-step run.
- no `Container was OOM killed by the kernel` in this run.

### Contrast with current 500-step run (same model family, different envelope)

From W&B/logs for `llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-153458-*`:
- train:
  - `num_train_steps=500`
  - `train_batch_size=1024`
  - `checkpointer.save_interval=0:10:00` (same interval, but long run repeatedly reaches it)
  - `max_weight_transfer_wait_time=0` (non-blocking)
- rollout:
  - generation calls at `max_tokens=1024`
  - repeated eval phases at `500 prompts` and rollout phases at `64 prompts`
  - W&B summary:
    - `inference.rollout/math_full/total_count=1024`
    - `inference.rollout_storage/last_size_bytes=6198068`
    - `inference.rollout_storage/cumulative_batch_count=10`

OOM pattern remains strongly checkpoint-coupled:
- save start -> OOM in ~45-50s (three times).

### Updated suggestions (hypotheses -> experiments)

1. **Checkpoint-overlap OOM is primary trigger (most likely).**
- experiment: keep v12 settings but set `checkpointer_save_interval` very large (effectively no mid-run save).
- pass condition: OOMs disappear while training continues beyond prior crash windows.

2. **Runtime memory margin is too thin in v12 envelope.**
- experiment: keep checkpointing enabled, reduce one dimension at a time:
  - `train_batch_size: 1024 -> 512`, then `256`, holding other params fixed.
  - optionally reduce `max_output_tokens: 1024 -> 512`.
- pass condition: checkpoint saves complete without OOM at least twice.

3. **Replay/rollout payload pressure contributes materially.**
- experiment: reduce rollout payload volume while keeping train batch fixed:
  - lower eval prompt count from `500` to `100`,
  - lower training prompt count from `64` to `32`.
- pass condition: lower host memory pressure and fewer post-restart skew windows.

4. **Non-blocking transfer may worsen restart churn (secondary).**
- experiment: same v12 settings, only switch `max_weight_transfer_wait_time` from `0 -> 300`.
- pass condition: fewer stale-rollout bursts after retries; does not by itself prove OOM fix.

5. **Need direct memory evidence for cluster admin.**
- request host/cgroup RSS and OOM killer detail at exact save/OOM timestamps:
  - `15:59:55 -> 16:00:40`
  - `16:16:25 -> 16:17:15`
  - `16:35:33 -> 16:36:20`

## 2026-03-23 — Full experiment census + new isolation setup

### Full `/ahmed/iris-rl*` census

Pulled `iris job list --json` and classified all matching jobs.

High-level:
- `84` matching jobs total.
- Majority of early failures are bootstrap/build/dependency class (e.g. `libcublas`, `torch` resolver, missing modules), not train-runtime OOM.
- Many runs were explicitly terminated by user during capacity/routing validation.

Runs that clearly reached active training and showed OOM:
- `/ahmed/iris-rl-500step`
  - `Saving checkpoint at step 4` -> OOM ~45s later.
- `/ahmed/iris-rl-500step-v2`
  - `Saving checkpoint at step 2` -> OOM ~49s later.
- `/ahmed/iris-rl-500step-v12`
  - repeated save->OOM windows at step 4/5.

Common signatures in all three:
- heavy 500-step profile with large train batch (1024),
- checkpoint save enabled,
- `max_weight_transfer_wait_time=0`,
- post-OOM restart emits `No checkpoints found ...`.

Control that does not show this:
- `/ahmed/iris-rl-v5p-3` (5-step debug) finishes without OOM, with small envelope and no checkpoint-save line observed in that short run.

### Code updates made in this pass

1. Set the active 500-step experiment to log to requested project:
- `experiments/exp2039_rl_math500.py`
  - added `project_name=\"marin_iris_rl_debug\"`.

2. Added a controlled isolation driver:
- `experiments/exp_iris_rl_oom_isolation.py`
  - switchable cases via `OOM_CASE`:
    - `baseline_ckpt_120`
    - `no_ckpt`
    - `reduced_batch_ckpt_120`
  - all cases use W&B project `marin_iris_rl_debug`.

### Isolation-run launch status

Attempted to launch baseline probe immediately, but coordinator submit path is currently blocked by CPU-pool constraints (`reservation-job` + `preemptible` + on-demand group at max slices / no matching zone group).

Actions taken:
- submitted baseline probe,
- captured pending reasons,
- stopped blocked submissions to avoid queue clutter.

Current blocker:
- coordinator outer job scheduling capacity, not TPU worker logic.

## 2026-03-23 PT / 2026-03-24 UTC — New isolation blocker (path-length failure)

After resubmitting the isolation baseline with larger coordinator resources:
- parent job: `/ahmed/iris-rl-oom-baseline-120-cpu2-20260323-215643`
- launch succeeded at parent level, and train/rollout children were created.

Observed child failure (both train and rollout):
- `OSError: [Errno 36] File name too long`
- failing path root: `/dev/shm/iris/workdirs/job__...`
- failure point: `task_attempt.py` during `workdir.mkdir(...)` before user training code runs.

Interpretation:
- this run did **not** reach RL runtime/checkpoint code; it failed in Iris worker setup.
- this failure is separate from the checkpoint-coupled OOM hypothesis and currently blocks the isolation experiment.

Next action recorded:
- shorten experiment/job/task naming envelope for the isolation driver and relaunch, so we can resume OOM-cause isolation on actual runtime behavior.

## 2026-03-23 PT / 2026-03-24 UTC — Outer v5p placement drift diagnosed and fixed

Live re-check of `/ahmed/iris-rl-oom-baseline-120-cpu2-20260323-215643`:
- root parent remains `JOB_STATE_RUNNING`.
- intermediate executor child and nested RL coordinator also remain `JOB_STATE_RUNNING`.
- actual RL worker children are terminal-failed:
  - train: `JOB_STATE_FAILED` after ~35s.
  - rollout: `JOB_STATE_FAILED` after ~38s.
- both RL worker children requested `v5p-8` resources (`32 cpu / 128 GiB / 50 GiB disk`) and each retried 4 times.

Why the job looked "running but with no logs":
- the failing exception is still the same Iris setup failure:
  - `OSError: [Errno 36] File name too long`
  - raised in `iris.cluster.worker.task_attempt.py` during `workdir.mkdir(...)`.
- this happens before user train/rollout code starts, so there are no meaningful RL runtime logs to inspect.
- the parent and intermediate wrapper jobs do not immediately transition terminal, so the tree can appear alive while all productive child work is already dead.

Placement diagnosis from live bug-report triage:
- the root parent bug report showed the outer launcher/executor task on worker:
  - `marin-tpu_v6e_4-europe-west4-a-20260324-0320-e7b531ef-worker-0`
- the same outer layer wrote executor metadata/output under:
  - `gs://marin-eu-west4/...`
- this Europe / `v6e` placement was **only** the outer launcher/executor layer.
- the RL train/rollout children were **not** requesting `v6e`; they were requesting `v5p-8`.
- root cause of the placement drift:
  - trainer/rollout child jobs in `orchestration.py` already had explicit TPU regions `us-central1` and `us-east5`.
  - but the outer RL executor step itself still used generic CPU resources and the experiment still allowed prefix resolution from worker metadata.
  - when the parent landed in Europe, `marin_prefix()` followed that Europe worker, so the executor wrote to `gs://marin-eu-west4` and the outer layer drifted onto a Europe `v6e` worker.

Code changes landed to prevent this for future `v5p` RL launches:
- `lib/marin/src/marin/rl/rl_experiment_utils.py`
  - added `executor_step_resources_for_rl_experiment()`:
    - for `v5p` runs, pin the outer RL executor step to `us-central1` / `us-east5`.
  - added `executor_main_config_for_rl_experiment()`:
    - for `v5p` runs, choose a US Marin prefix from the current allowed region when available;
    - otherwise default to `gs://marin-us-central1` instead of inheriting Europe.
  - updated `make_rl_step()` to use the v5p-aware executor-step resources.
- experiment entrypoints updated to use the helper config:
  - `experiments/exp2039_rl_math500.py`
  - `experiments/exp_iris_rl_oom_isolation.py`
  - `experiments/exp_iris_rl_debug.py`
- regression coverage added:
  - `tests/rl/test_rl_experiment_utils.py`

Validation for the placement fix:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py` -> `4 passed`.
- `./infra/pre-commit.py --fix ...` on touched files -> `OK`.

Net result:
- future `v5p` RL runs should no longer silently drift onto Europe / `v6e` at the outer executor layer.
- this does **not** fix the current path-length blocker; relaunch still requires shortening the naming envelope.
- operational launch rule remains unchanged:
  - use `--region us-central1` by default for root submission,
  - or `--zone us-east5-a` when intentionally choosing the east5 path.

## 2026-03-23 PT / 2026-03-24 UTC — Generic single-region pinning + short-name relaunch plan

Code changes prepared before relaunch:
- generalized RL region pinning beyond `v5p`:
  - added `lib/marin/src/marin/rl/placement.py` to resolve a concrete launcher region from requested TPU variants plus current launcher region.
  - outer RL executor step now pins to that concrete region for any requested TPU variant, not just `v5p`.
  - `ExecutorMainConfig.prefix` now follows the same concrete region via the canonical Marin bucket for that region.
  - `RunConfig` now carries `regions`, and `orchestration.py` uses the same single-region placement for train/rollout TPU children.
- behavior change:
  - if the root Iris launcher is in a region incompatible with the requested TPU type and autoscaler info is available, RL launch now fails fast instead of silently drifting.
- isolation-driver naming was shortened to avoid the prior `/dev/shm/iris/workdirs/job__...` path-length failure:
  - model/run name now uses short form `l31-8bi-<case>-<timestamp>`.

Validation before relaunch:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py` -> `4 passed`.
- `./infra/pre-commit.py --fix ...` on touched files -> `OK`.

Planned relaunch command:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait \
  --user ahmed \
  --job-name iris-rl-oom-b120-uc1 \
  --region us-central1 \
  --cpu 2 \
  --memory 8GB \
  -e OOM_CASE baseline_ckpt_120 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -e MARIN_PREFIX gs://marin-us-central1 \
  -- uv run python experiments/exp_iris_rl_oom_isolation.py
```

### Relaunch attempt 1 (`/ahmed/iris-rl-oom-b120-uc1`) — placement corrected, entrypoint bug exposed

Observed result:
- root job was submitted with explicit `--region us-central1` and `MARIN_PREFIX=gs://marin-us-central1`.
- parent landed on `marin-tpu_v5p_8-us-central1-a-...-worker-0`, confirming the control-plane / launcher placement fix worked.
- job failed early before nested RL child submission with:
  - `TypeError: executor_main() got multiple values for argument 'config'`

Root cause:
- `executor_main` is draccus-wrapped and expects its config as the first positional argument.
- the earlier helper integration passed `config=...` as a keyword in the RL experiment entrypoints.

Fix applied immediately after this failed attempt:
- updated these entrypoints to pass `executor_main_config_for_rl_experiment(...)` positionally:
  - `experiments/exp2039_rl_math500.py`
  - `experiments/exp_iris_rl_oom_isolation.py`
  - `experiments/exp_iris_rl_debug.py`
- re-ran `./infra/pre-commit.py --fix ...` on the touched entrypoints.

Interpretation:
- region placement is now behaving as intended.
- the next relaunch should test the remaining hypothesis of interest: whether the shortened naming envelope clears the prior `File name too long` failure.

### Relaunch attempt 2 (`/ahmed/iris-rl-oom-b120-uc1-r2`) — region + naming fixes held, new HF quota blocker

Observed result:
- root job launched with explicit `--region us-central1` and `MARIN_PREFIX=gs://marin-us-central1`.
- root job started successfully and remained in the intended region.
- prior blockers did **not** recur in this attempt:
  - no Europe / `v6e` control-plane drift,
  - no `executor_main(... config=...)` wrapper error,
  - no `/dev/shm/iris/workdirs/... File name too long` failure before child creation.

New failure:
- root job failed during experiment construction in `make_rl_step()` when resolving the HF model config:
  - `AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")`
  - upstream error was `429 Client Error: Too Many Requests` from Hugging Face for `config.json`.
- final surfaced exception:
  - `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.`

Interpretation:
- the placement and naming fixes are now validated by absence of the earlier failure modes.
- current blocker has shifted back to dependency/model bootstrap:
  - root launcher needs either a warm local HF cache, a retry/backoff path, or an alternate way to materialize the model config without hitting fresh HF resolver requests.
- this failure occurs before nested RL child jobs are submitted, so it does not yet validate end-to-end RL worker startup under the shortened naming envelope.

## 2026-03-23 PT / 2026-03-24 UTC — Correction: executor-managed model artifacts already exist

Follow-up on the new root-launcher failure from `/ahmed/iris-rl-oom-b120-uc1-r2`:
- the problem is **not** that Marin lacks a GCP-hosted copy of `meta-llama/Llama-3.1-8B-Instruct`.
- `experiments/models.py` already defines this model as an executor-managed download step:
  - `llama_3_1_8b_instruct = download_model_step(ModelConfig(hf_repo_id="meta-llama/Llama-3.1-8B-Instruct", hf_revision="0e9e39f"))`
- that step is intentionally wired to produce a stable, region-local GCS artifact rather than an opaque hashed path:
  - `gcs_output_path=this_output_path()` means the download writes into the step's resolved output directory.
  - `override_output_path=f"models/{model_name}--{model_revision}"` forces a deterministic final path under the executor prefix.
  - `versioned(model_config.hf_revision)` keeps the revision in the executor version graph.
- executor resolution then turns that into a concrete regional path under the current Marin prefix. For the us-central1 case, the intended artifact path is:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`

Cross-check with existing downstream usage:
- `experiments/evals/evals.py` already treats downloaded models as executor artifacts and resolves them via `output_path_of(step, "hf")`.
- so the framework already has the right abstraction for "use the region-local downloaded model artifact".

Updated interpretation of the HF 429 blocker:
- the RL stack is bypassing that abstraction.
- current RL experiment code still hardcodes the raw HF repo string in `ModelConfig` / checkpoint / tokenizer fields and then performs launcher-time HF resolution in `make_rl_step()`:
  - `AutoConfig.from_pretrained(config.model_config.name)`
- this causes the root Iris job to contact Hugging Face during experiment construction, which is why `/ahmed/iris-rl-oom-b120-uc1-r2` failed before any nested RL child jobs were submitted.

Implication for the next fix:
- do **not** add retries/backoff around fresh HF reads as the primary solution.
- instead, make RL consume the executor-managed downloaded model artifact (or its resolved GCS path) for config/tokenizer/checkpoint bootstrap, so root launch no longer depends on live HF access.

## 2026-03-23 PT / 2026-03-24 UTC — RL launcher now depends on cached regional model artifact

Implemented follow-up fix for the Hugging Face launcher dependency:
- `make_rl_step()` no longer calls `AutoConfig.from_pretrained(...)` during root experiment construction.
- RL model bootstrap now happens inside the executor step runtime.
- the RL step depends directly on the executor-managed model download step from `experiments.models.llama_3_1_8b_instruct`.
- because the model download is an executor dependency, cached regional artifacts are reused automatically and the RL step no longer requires a live HF lookup at root launch.

Additional runtime/path changes:
- stopped serializing the `LlamaConfig` class object into executor metadata; RL runtime now stores a dotted config-class import path and resolves it inside the worker.
- normalized RL tokenizer loading to `levanter.compat.hf_checkpoints.load_tokenizer(...)`.
- taught RL checkpoint detection to recognize HF exports on `gs://...` and local paths by checking for HF marker files.
- corrected the model artifact path for `download_model_step(...)` outputs: the HF export lives at the step root for these model download steps, not under `/hf`.

Validation in repo:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py` -> `6 passed`
- `./infra/pre-commit.py --fix ...` on touched RL files -> `OK`

Relaunch sequence after this fix:

### Attempt `r3` — executor config still contained raw config class objects
- `/ahmed/iris-rl-oom-b120-uc1-r3`
- root placement remained correct in `us-central1` on `v5p-8`.
- failed before step launch because executor dependency/version traversal hit `AttributeError: type object 'LlamaConfig' has no attribute 'rope'` while walking the class object.

### Attempt `r4` — version traversal fixed, metadata serialization still broken on class object
- `/ahmed/iris-rl-oom-b120-uc1-r4`
- executor reached dependency discovery and identified both steps.
- failed while writing executor metadata because the class object was still present in serialized step config.

### Attempt `r5` — class serialization fixed, wrong model subpath exposed
- `/ahmed/iris-rl-oom-b120-uc1-r5`
- executor launched two steps as intended:
  - `models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f` (skipped as already succeeded in `us-central1`)
  - `rl_testing/l31-8bi-b120-...`
- RL step reached runtime bootstrap, proving the HF launcher dependency was removed.
- new failure: tokenizer bootstrap used the wrong subpath (`.../hf`) for this download step layout and failed inside `AutoTokenizer` with `AttributeError: 'NoneType' object has no attribute 'endswith'`.

### Attempt `r6` — first healthy launch through cached model-step path
- `/ahmed/iris-rl-oom-b120-uc1-r6`
- root job is `RUNNING` in `us-central1`.
- nested executor child `/ahmed/iris-rl-oom-b120-uc1-r6/rl_testing-l31-8bi-b120-20260324-065718_d700d4fc-a5b2ef1c` is also `RUNNING`.
- observed behavior confirms the intended DAG/caching behavior:
  - executor discovered two steps,
  - skipped the cached model download step in `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`,
  - launched the RL step afterward.
- the RL step now gets materially deeper into startup than prior attempts (past root construction, past class serialization, past model-step resolution, past the bad `/hf` tokenizer path).
- latest visible logs show the nested RL step still running after model/bootstrap setup; no new hard failure captured yet at logbook update time.

Interpretation:
- the original requested fix is now working:
  - no fresh HF read in the launcher,
  - regional model artifact is part of the executor DAG,
  - cached model download results are reused automatically in-region,
  - region pinning remains in force for root and RL step launch.
- remaining runtime investigation, if needed, should now focus on actual RL worker startup behavior rather than HF/bootstrap pathing.

## 2026-03-24 UTC — Failure surfacing fix + runtime output-path fix

Observed after `r6`:
- W&B showed repeated trainer crashes with `AttributeError: 'OutputName' object has no attribute 'rstrip'` in `rollout_storage.py`.
- `iris job bug-report /ahmed/iris-rl-oom-b120-uc1-r6` still showed the root job as `running` with `Failures 0`, which hid the real RL failure from the root-level Iris view.

Root causes split into two separate issues:

1. RL runtime config still contained unresolved executor placeholders
- `rl_experiment_utils._build_rl_job_config(...)` was constructing `CheckpointerConfig(base_path=OutputName("checkpoints"))` and `RolloutStorageConfig(path=OutputName("rollouts"))` inside the remote step runtime.
- those `OutputName(...)` values were created *after* executor config instantiation, so they never became concrete strings.
- train and rollout workers then crashed when file-based rollout storage called `.rstrip("/")` on an `OutputName` object.

2. Coordinator failure did not surface because the process stayed alive after the main thread crashed
- the RL coordinator correctly hit `wait_all(jobs, raise_on_failure=True)` and raised `fray.v2.client.JobFailed` once a child TPU job reached terminal failure.
- however, the coordinator hosts in-process actors (`curriculum`, `run_state`, Arrow Flight coordinator) via `client.host_actor(...)`.
- those actor servers own non-daemon managed threads/executors.
- because Marin never called `HostedActor.shutdown()` on the error path, the coordinator main thread could die while the process stayed alive.
- Iris only marks the task failed when the process exits, so the coordinator task and all ancestors stayed `RUNNING` even though logs already contained the traceback.

Iris CLI surfacing gap:
- `iris job bug-report` is shallow for the requested job ID.
- before this fix, it did not summarize descendant job states/failures, so root bug reports hid failing train/rollout child jobs unless logs were queried with `--include-children` or descendant jobs were inspected directly.

Applied fixes:

### RL runtime path fix
- `RLStepConfig` now carries the concrete step `output_path` resolved by executor config instantiation.
- `_build_rl_job_config(...)` now builds concrete string paths under that output root:
  - `.../checkpoints`
  - `.../rollouts`
- this removes the unresolved `OutputName` objects from runtime-built RL worker config.

### Coordinator teardown fix
- `orchestration._create_runtime_handles(...)` now returns both runtime handles and the hosted actor objects.
- added `_shutdown_hosted_actors(...)` and call it in coordinator `finally` cleanup.
- hosted actors are also cleaned up if runtime-handle creation itself fails partway through.
- intended effect: once `wait_all(...)` raises, the coordinator process can actually exit, so Iris sees the task fail and parent jobs stop pretending to be healthy.

### Iris bug-report surfacing fix
- `iris.cli.bug_report` now fetches descendant job statuses using the existing `GetTaskLogs(... include_children=True)` plumbing.
- bug reports now include:
  - descendant job count,
  - descendant failure count,
  - a `Descendant Jobs` table with nested child job state/error.
- `error_summary` now becomes `N descendant job(s) failed` when the root job itself is still `running` but nested jobs have already failed.

Tests / validation:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py tests/rl/test_orchestration.py` -> `8 passed`
- `uv run pytest -q lib/iris/tests/cli/test_bug_report.py -o addopts= --import-mode=importlib` -> `1 passed`
- `./infra/pre-commit.py --fix ...` on all touched files -> `OK`

Additional local debug artifact:
- `docs/debug-log-iris-rl-failure-surfacing.md` captures the live cluster diagnosis and reasoning for the coordinator/process-liveness bug.

## 2026-03-24 UTC — Live relaunch `r7` after surfacing/runtime-path fixes

Operator action:
- checked live `/ahmed/` jobs before relaunch; there were no running `iris-rl-*` jobs left to kill.
- unrelated active jobs were only model-download / Zephyr download jobs and were left untouched.
- launched:
  - `/ahmed/iris-rl-oom-b120-uc1-r7`
  - with explicit root placement `--region us-central1` and `MARIN_PREFIX=gs://marin-us-central1`

Observed startup behavior:
- root job launched on `marin-tpu_v5p_8-us-central1-a-20260324-0544-c021e7c8-worker-0`.
- executor again discovered exactly two top-level steps:
  - cached model download `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
  - RL step `gs://marin-us-central1/rl_testing/l31-8bi-b120-20260324-072435-268031`
- cached model step was skipped as already succeeded in-region.
- the previous RL runtime-path bug did **not** recur:
  - no `AttributeError: 'OutputName' object has no attribute 'rstrip'`

Observed Iris surfacing behavior:
- root `iris job bug-report /ahmed/iris-rl-oom-b120-uc1-r7` immediately showed descendant jobs, confirming the descendant-summary patch is active.
- descendant tree progressed as:
  - root launcher
  - executor child `.../rl_testing-l31-8bi-b120-20260324-072435_796d02a9-e14db7d9`
  - RL coordinator `.../rl-l31-8bi-b120-20260324-072435`
  - later, train and rollout child jobs became visible from the root bug report:
    - `.../rl-l31-8bi-b120-20260324-072435-train`
    - `.../rl-l31-8bi-b120-20260324-072435-rollout-0`

New transient infra issue observed during live monitor:
- the RL coordinator's first CPU task attempt hit:
  - `Worker marin-cpu-vm-e2-highmem-2-ondemand-us-ce-20260324-0657-eaf8c875-worker-0 failed: Request timed out`
- Iris retried that coordinator task automatically onto a fresh worker:
  - `marin-cpu-vm-e2-highmem-2-ondemand-us-ce-20260324-0725-35ada24a-worker-0`
- bug report for the coordinator showed:
  - attempt `0`: `worker_failed`
  - attempt `1`: progressed from `building` to `running`

State at `2026-03-24T07:29:56Z`:
- root job `/ahmed/iris-rl-oom-b120-uc1-r7`: `running`
- RL executor child: `running`
- RL coordinator: `running` on retry attempt `1`
- train child: `pending`
- rollout child: `pending`

Interpretation:
- the requested code fixes are behaving as intended so far:
  - root bug reports now expose descendant RL jobs,
  - RL runtime config now uses concrete output paths,
  - the old hidden-failure + `OutputName` crash path has not reappeared in `r7`.
- the current blocker, if this run later fails, is likely separate:
  - transient CPU worker instability / timeout during coordinator placement or later downstream child startup,
  - not the previously fixed RL config/surfacing bugs.

## 2026-03-24 UTC — Active research goal and handoff contract

Active goal for this thread:
- get the Iris RL path stable enough to run the 500-step case end-to-end.
- this is now the top-level success criterion, superseding "migration landed" as the practical target.

Operational stop criteria:
- stop only when a 500-step RL run completes successfully with the intended train/rollout topology, or when a blocker is proven unrecoverable without a broader design change.
- do not stop at first child-job creation, first W&B link, or first short successful startup.

Current monitoring owner:
- Codex agent in `iris_rl` worktree, using `agent-research` + `babysit-job`.

Current live monitor metadata:
- track: `iris`
- active root job: `/ahmed/iris-rl-oom-b120-uc1-r7`
- Iris config: `lib/iris/examples/marin.yaml`
- monitoring state file:
  - `scratch/20260324-0033_monitoring_state.json`
- current resubmit command on disk:
  - `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-oom-b120-uc1-r7 --region us-central1 --cpu 2 --memory 8GB -e OOM_CASE baseline_ckpt_120 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -e MARIN_PREFIX gs://marin-us-central1 -- uv run python experiments/exp_iris_rl_oom_isolation.py`

Current live interpretation:
- `r7` has validated the recent code fixes further than prior attempts:
  - descendant jobs now surface in root bug reports,
  - concrete output-path fix held,
  - cached regional model artifact dependency held.
- the active failure mode is now infra/startup-adjacent:
  - first RL coordinator CPU worker attempt timed out,
  - Iris retried onto a fresh CPU worker,
  - train and rollout child jobs are now present but still pending at latest check.

Next-agent contract after context compaction:
- resume from `scratch/20260324-0033_monitoring_state.json`.
- continue append-only updates in this logbook with:
  - exact UTC timestamp,
  - exact Iris command(s),
  - current job tree state,
  - failure signatures and recovery steps,
  - decision for next action.
- if `r7` reaches a terminal failure:
  - inspect descendant bug reports first,
  - apply only small/local code fixes automatically,
  - for complex failures, log the exact signature and choose the narrowest next debug path,
  - then stop the failed job and resubmit using the saved command or an explicitly updated replacement command.
- if `r7` runs long enough to produce meaningful train progress:
  - capture progress as `~<current>/500`,
  - record W&B link if visible,
  - keep monitoring until terminal success or clear failure.

## 2026-03-24T07:35Z — `r7` reached live train startup; rollout blocked on TPU capacity

Exact monitor commands:
- `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-oom-b120-uc1-r7/rl_testing-l31-8bi-b120-20260324-072435_796d02a9-e14db7d9/rl-l31-8bi-b120-20260324-072435/rl-l31-8bi-b120-20260324-072435-train`
- `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-oom-b120-uc1-r7/rl_testing-l31-8bi-b120-20260324-072435_796d02a9-e14db7d9/rl-l31-8bi-b120-20260324-072435/rl-l31-8bi-b120-20260324-072435-rollout-0`

Observed state:
- trainer child is now `running` on:
  - `marin-tpu_v5p_8-us-central1-a-20260324-0729-b4b205de-worker-0`
- rollout child is still `pending`.

Useful concrete evidence from train logs:
- W&B run created successfully:
  - project: `https://wandb.ai/marin-community/marin_iris_rl_debug`
  - run: `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/l31-8bi-b120-20260324-072435-train`
- rollout storage reader initialized successfully at:
  - `gs://marin-us-central1/rl_testing/l31-8bi-b120-20260324-072435-268031/rollouts`
- Arrow Flight servers started successfully on the trainer host.

Rollout pending reason:
- `Scheduler: Insufficient TPUs (need 4, available 0)`
- autoscaler detail:
  - waiting for workers in scale group `tpu_v5p_8-us-central1-a` to become ready

Interpretation:
- the previously fixed startup path is now validated more strongly:
  - trainer startup reached JAX distributed init, W&B init, rollout-storage init, and Arrow Flight server startup.
- current blocker is scheduler capacity, not an RL code exception:
  - train is alive,
  - rollout is waiting for a `v5p-8` worker in `us-central1`.

Next action:
- continue babysit loop without resubmitting while rollout remains in scheduler-capacity wait.
- if rollout later fails terminally or trainer emits a real traceback/OOM/HBM error, switch from passive monitor back to active debugging.

## 2026-03-24T07:36Z — Autoscaler evidence for rollout stall

Exact monitor commands:
- `uv run iris --config=lib/iris/examples/marin.yaml rpc controller get-autoscaler-status`
- `uv run iris --config=lib/iris/examples/marin.yaml query -f json "SELECT job_id, name, state, error FROM jobs WHERE job_id LIKE '/ahmed/iris-rl-oom-b120-uc1-r7%' ORDER BY submitted_at_ms ASC"`

Relevant autoscaler facts for the live `r7` rollout wait:
- active unscheduled task is:
  - `/ahmed/iris-rl-oom-b120-uc1-r7/.../rl-l31-8bi-b120-20260324-072435-rollout-0/0`
- routed scale group is:
  - `tpu_v5p_8-us-central1-a`
- one `v5p-8` slice for the trainer was created and became ready:
  - `marin-tpu_v5p_8-us-central1-a-20260324-0729-b4b205de`
- the second `v5p-8` slice needed for rollout failed once at GCP allocation time:
  - `There is no more capacity in the zone "us-central1-a"`
- latest autoscaler action still shows another scale-up attempt for the same group as `pending`.

Interpretation:
- rollout is blocked by real zone capacity in `us-central1-a`, not by an RL code exception or unresolved dependency.
- this is compatible with the earlier design decision to keep driver/CPU and TPU children in one concrete region.
- if the pending rollout wait does not clear, the likely recovery path is not code editing first; it is a placement decision:
  - either keep waiting for `us-central1-a`,
  - or relaunch the whole root job in `us-east5` so the entire RL tree stays region-coherent there instead.

## 2026-03-24T07:37Z — Both TPU children allocated; trainer is waiting on first rollout batch

Exact monitor commands:
- `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-oom-b120-uc1-r7/.../rl-l31-8bi-b120-20260324-072435-rollout-0`
- `uv run iris --config=lib/iris/examples/marin.yaml job logs --since-seconds 600 --include-children /ahmed/iris-rl-oom-b120-uc1-r7 | rg -i ...`

Observed state:
- rollout child is now `running` on:
  - `marin-tpu_v5p_8-us-central1-a-20260324-0732-2552ab5f-worker-0`
- trainer child remained `running`.

Important train-side evidence:
- trainer loaded the regional base checkpoint successfully from:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- trainer entered training setup and published initial Arrow Flight weights:
  - `Serialized model to Arrow ... total size 15316.51 MB`
  - `Served weights for weight_id -1`
- trainer then blocked in the expected place:
  - `Waiting for initial rollouts from step -1...`
  - repeated status: `buffer size: 0`

Interpretation:
- there is still no RL-code exception in the hot path after both TPU children were created.
- current gating condition is now rollout-worker startup / first rollout generation, not scheduling.
- next thing to watch:
  - rollout worker init,
  - first rollout write into `gs://.../rollouts`,
  - trainer exiting the `Waiting for initial rollouts` loop,
  - then actual train-step progress.

## 2026-03-24T07:40Z — Rollout vLLM bootstrap failure identified, patched, and relaunched as `r8`

Failure observed in live `r7` rollout logs:
- rollout child crashed during async vLLM engine init with:
  - `pydantic_core._pydantic_core.ValidationError`
  - `To load a model from S3, 'load_format' must be 'runai_streamer' or 'runai_streamer_sharded', but got 'dummy'`
- trainer behavior during this failure was consistent:
  - kept waiting for initial rollouts with `buffer size: 0`

Diagnosis:
- RL experiment setup still hardcoded `load_format="dummy"` for rollout-side vLLM startup.
- that assumption no longer holds for object-store-backed model artifacts (`gs://...`) now used by the RL path.
- existing Marin inference code already handles this class of path by switching to `runai_streamer`.

Code changes prepared locally:
- `lib/marin/src/marin/rl/rl_experiment_utils.py`
  - added object-store-aware vLLM load-format selection:
    - `gs://` / `s3://` -> `runai_streamer`
    - non-object-store paths -> `dummy`
- `tests/rl/test_rl_experiment_utils.py`
  - added assertions for both branches
- debug log:
  - `docs/debug-log-iris-rl-rollout-vllm-load-format.md`

Validation:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py` -> `8 passed`
- `./infra/pre-commit.py --fix ...` on touched files -> `OK`

Operational recovery:
- stopped stale root job:
  - `/ahmed/iris-rl-oom-b120-uc1-r7`
- relaunched patched bundle as:
  - `/ahmed/iris-rl-oom-b120-uc1-r8`
- updated monitor state file:
  - `scratch/20260324-0033_monitoring_state.json`
  - `restart_count: 1`

Next action:
- babysit `r8` through the same startup sequence and verify that rollout no longer dies in vLLM init.
- only after that is confirmed should the thread move back to 500-step runtime/OOM behavior.

## 2026-03-24T07:43Z — `r8` clears the rollout bootstrap regression; both child runs are live

Exact monitor commands:
- `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-oom-b120-uc1-r8`
- `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-oom-b120-uc1-r8/rl_testing-l31-8bi-b120-20260324-074116_9fa5ded0-be6ed9d9/rl-l31-8bi-b120-20260324-074116/rl-l31-8bi-b120-20260324-074116-train`
- `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-oom-b120-uc1-r8/rl_testing-l31-8bi-b120-20260324-074116_9fa5ded0-be6ed9d9/rl-l31-8bi-b120-20260324-074116/rl-l31-8bi-b120-20260324-074116-rollout-0`
- `uv run iris --config=lib/iris/examples/marin.yaml job logs --since-seconds 900 --include-children /ahmed/iris-rl-oom-b120-uc1-r8 | rg -i -n "wandb|step|rollout|waiting for initial rollouts|buffer size|loss|checkpoint|oom|traceback|exception|error|saved"`

Observed state:
- root job `r8` is `running`.
- descendant RL step job is `running`.
- trainer child is `running` on `marin-tpu_v5p_8-us-central1-a-20260324-0729-b4b205de-worker-0`.
- rollout child is `running` on `marin-tpu_v5p_8-us-central1-a-20260324-0732-2552ab5f-worker-0`.

Important evidence that the latest fix worked:
- rollout no longer dies with the previous vLLM validation error:
  - no `load_format ... got 'dummy'` traceback in `r8`.
- rollout W&B run started successfully:
  - `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/l31-8bi-b120-20260324-074116-rollout-0`
- trainer W&B run started successfully:
  - `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/l31-8bi-b120-20260324-074116-train`
- rollout storage writer initialized successfully at:
  - `gs://marin-us-central1/rl_testing/l31-8bi-b120-20260324-074116-3c262a/rollouts`

New watch item from rollout startup:
- rollout logs contain:
  - `Tensorflow library not found, tensorflow.io.gfile operations will use native shim calls. GCS paths (i.e. 'gs://...') cannot be accessed.`
- this did not prevent rollout startup or rollout-writer initialization, so it is not yet a confirmed blocker.
- if rollout later fails while loading model weights or tokenizer assets from `gs://...`, this warning becomes the first hypothesis.

Interpretation:
- the object-store-aware `runai_streamer` fix is validated live.
- the thread is now past the earlier startup regressions:
  - region/prefix mismatch,
  - HF config fetch on launcher,
  - `OutputName.rstrip` rollout storage path bug,
  - rollout `load_format='dummy'` crash.
- next gate is no longer process startup; it is whether rollout produces the first samples and trainer begins real step advancement.

Next action:
- continue babysitting `r8` until either:
  - trainer exits the initial wait and step counters advance, or
  - a new concrete failure signature appears.

## 2026-03-24T07:48Z — `r8` hits the next rollout-side bug after first weight transfer

Exact monitor command:
- `uv run iris --config=lib/iris/examples/marin.yaml job logs --since-seconds 300 --include-children /ahmed/iris-rl-oom-b120-uc1-r8 | tail -n 500`

Critical live sequence observed:
- rollout async vLLM engine initialized successfully
- rollout loaded initial policy checkpoint path:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- trainer served initial Arrow Flight weights for `weight_id -1`
- rollout fetched those weights successfully:
  - `Received 291 params for weight_id -1 via Arrow Flight`
- failure happened only when applying the weights into the vLLM worker:
  - `KeyError: 'No MODEL_MAPPING registered for model: gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f'`

Interpretation:
- the previous `runai_streamer` fix is confirmed good.
- the new failure is a separate regression introduced by using executor-managed regional model artifacts.
- weight-transfer tensor remapping still keys off the canonical HF model id, but rollout now passes the concrete `gs://...` artifact path into that lookup.
- result:
  - rollout background weight sync loop crashes,
  - rollout main thread keeps waiting for first weight transfer,
  - trainer keeps waiting for initial rollouts with `buffer size: 0`.

Local fix prepared:
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
  - added `canonical_model_name` to the config
  - keep `model_name` as the real checkpoint path
  - use `canonical_model_name` for renderer and tensor-mapping lookups
- `lib/marin/src/marin/rl/environments/inference_ctx/async_vllm.py`
  - async weight updates now send `canonical_model_name`
- `lib/marin/src/marin/rl/rl_experiment_utils.py`
  - RL job builder now sets `canonical_model_name=config.model_config.name`
- tests:
  - `tests/rl/test_rl_experiment_utils.py`
  - `tests/rl/test_inference_ctx.py`

Validation:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py tests/rl/test_inference_ctx.py` -> `17 passed`
- `uv run python -m compileall ...` on the touched RL files -> success

Next action:
- run pre-commit on the touched files,
- stop broken `r8`,
- relaunch with this canonical-model-name fix,
- continue babysitting until either first rollout batch lands or the next concrete failure appears.

## 2026-03-24T07:49Z — Stopped `r8`, relaunched as `r9`

Operational actions:
- stopped the broken job tree:
  - `/ahmed/iris-rl-oom-b120-uc1-r8`
- relaunched with the canonical-model-name fix as:
  - `/ahmed/iris-rl-oom-b120-uc1-r9`
- updated monitor state file:
  - `scratch/20260324-0033_monitoring_state.json`
  - `restart_count: 2`

Validated local bundle before relaunch:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py tests/rl/test_inference_ctx.py` -> `17 passed`
- `./infra/pre-commit.py --fix ...` on touched files -> `OK`

Next action:
- perform the startup-stabilization babysit check on `r9`
- verify that rollout gets past first weight application this time
- only after that return to actual rollout generation / train-step progress

## 2026-03-24T07:52Z — `r9` clears the canonical-model-name regression

Startup-stabilization check:
- `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-oom-b120-uc1-r9`
- `uv run iris --config=lib/iris/examples/marin.yaml job logs --since-seconds 300 --include-children /ahmed/iris-rl-oom-b120-uc1-r9 | tail -n 500`

Observed live state:
- root job `r9`: `running`
- RL step job: `running`
- trainer child: `running`
- rollout child: `running`

Important evidence that the latest fix worked:
- rollout async engine initialized successfully
- rollout streamed the regional model artifact with RunAI Streamer
- trainer served initial Arrow Flight weights for `weight_id -1`
- rollout fetched and applied those weights without the previous `MODEL_MAPPING` KeyError:
  - `Received 291 params for weight_id -1 via Arrow Flight`
  - `receive_weights: update_model complete, total=12.4s`
  - `Received new weights from step -1`

Interpretation:
- the canonical-model-name fix is validated live.
- the thread is now past another startup blocker:
  - rollout no longer crashes when applying the first trainer weights.
- the next gate is finally the real RL handshake:
  - rollout must start generating samples,
  - trainer must stop reporting `buffer size: 0`,
  - then step counters need to advance toward the actual 500-step target.

Next action:
- continue babysitting `r9` until either first rollout files appear and trainer advances, or a new concrete failure signature appears.

## 2026-03-24T07:56Z — `r9` finds the next async weight-update bug after first successful mapping lookup

Exact monitor commands:
- `uv run iris --config=lib/iris/examples/marin.yaml job logs --since-seconds 240 --include-children /ahmed/iris-rl-oom-b120-uc1-r9 | rg -i -n ...`
- `uv run iris --config=lib/iris/examples/marin.yaml job logs --since-seconds 600 --include-children /ahmed/iris-rl-oom-b120-uc1-r9 | nl -ba | sed -n '185,200p'`
- `gcloud storage cat gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/config.json`

Critical new facts:
- the GCS model artifact itself is correct:
  - `architectures: ["LlamaForCausalLM"]`
  - `model_type: "llama"`
- rollout gets past the previous `MODEL_MAPPING` failure and receives the first weight payload.
- the next crash happens deeper inside async vLLM worker-side weight application:
  - `AttributeError: 'dict' object has no attribute 'flat_state'`
- live traceback shows the failing call chain:
  - `WorkerExtension.update_weight`
  - `self.model_runner._sync_weights(...)`
  - `transfer_state_with_mappings`
  - `tgt_state.flat_state()`

Useful rollout-side context from the same logs:
- vLLM resolves the streamed model as:
  - `Resolved architecture: MistralForCausalLM`
- then `tpu_inference` logs:
  - `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`

Interpretation:
- the canonical-model-name fix is working.
- the new blocker is specific to the async worker extension implementation, not to regional model artifacts.
- `WorkerExtension.update_weight` was using the internal `_sync_weights` entry point.
- the safer/public contract already used elsewhere is `sync_weights(...)`; using the internal method appears to bypass logic needed for the current backend state.

Local fix prepared:
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`
  - switched from `self.model_runner._sync_weights(...)` to public `self.sync_weights(...)`
- `tests/rl/test_inference_ctx.py`
  - added a regression test that verifies `WorkerExtension.update_weight` delegates to public `sync_weights` with an `nnx.State`

Validation:
- `uv run pytest -q tests/rl/test_inference_ctx.py tests/rl/test_rl_experiment_utils.py` -> `18 passed`
- `uv run python -m compileall lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/rl/test_inference_ctx.py` -> success

Next action:
- run pre-commit on the touched files,
- stop broken `r9`,
- relaunch with this worker-extension fix,
- keep babysitting toward actual rollout generation and then 500-step stability.

## 2026-03-24T07:59Z — Stopped `r9`, relaunched as `r10`

Operational actions:
- stopped the broken job tree:
  - `/ahmed/iris-rl-oom-b120-uc1-r9`
- relaunched with the public `sync_weights` worker-extension fix as:
  - `/ahmed/iris-rl-oom-b120-uc1-r10`
- updated monitor state file:
  - `scratch/20260324-0033_monitoring_state.json`
  - `restart_count: 3`

Validated local bundle before relaunch:
- `uv run pytest -q tests/rl/test_inference_ctx.py tests/rl/test_rl_experiment_utils.py` -> `18 passed`
- `./infra/pre-commit.py --fix ...` on touched files -> `OK`

Next action:
- perform the startup-stabilization babysit check on `r10`
- verify whether the first rollout weight application now completes end-to-end
- keep pushing until rollout generation and trainer-step progress are real, not just startup-clean

## 2026-03-24T08:02Z — `r10` is currently blocked by TPU capacity before the next code path is exercised

Startup-stabilization evidence:
- root `/ahmed/iris-rl-oom-b120-uc1-r10`: `running`
- trainer child: `running`
- rollout child: `pending`

Autoscaler evidence:
- unmet entry is the rollout task:
  - `/ahmed/iris-rl-oom-b120-uc1-r10/.../rl-l31-8bi-b120-20260324-075919-rollout-0/0`
- routing reason:
  - `no_capacity: tpu_v5p_8-us-central1-a=backoff`
- most recent scale-up failure for the same group:
  - `There is no more capacity in the zone "us-central1-a"`

Interpretation:
- `r10` has not yet reached the previous async weight-update failure boundary.
- the current gate is scheduler capacity in `us-central1-a`, not a fresh RL traceback.
- once rollout TPU allocation clears, the next thing to verify is whether the public `sync_weights` worker-extension fix survives the first trainer->rollout weight application.

Next action:
- continue babysitting `r10` until rollout TPU allocation succeeds or the capacity backoff makes another relaunch decision necessary.

## 2026-03-24T08:14Z — `r10` rollout postmortem after capacity cleared

Primary goal for this thread:
- get the Iris RL path stable through a real 500-step run.
- do not stop at startup-clean; the bar is trainer step advancement plus sustained rollout/trainer weight exchange.

What actually happened after `us-central1-a` capacity cleared:
- rollout TPU worker started and initialized async vLLM.
- rollout loaded the regional cached checkpoint:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- trainer served the initial Arrow Flight payload for `weight_id -1`.
- rollout fetched that payload successfully:
  - `Received 291 params for weight_id -1 via Arrow Flight`
  - `Received new weights from step -1`
- rollout then crashed in the background weight-sync loop before it could mark the first weight transfer complete.
- the foreground rollout thread kept waiting on the "first weight transfer" event, so the process looked alive while being functionally dead.

Exact failing traceback:
- `rollout_worker._sync_weights_loop`
- `AsyncvLLMInferenceContext.reload_model`
- `SyncVLLMWrapper.update_weights`
- `engine_core.collective_rpc_async("update_weight", ...)`
- worker-side `TPUWorker.sync_weights(...)`
- `TPURunner._sync_weights(...)`
- `AttributeError: 'dict' object has no attribute 'flat_state'`

Important current interpretation:
- this is still a rollout-side bug, not a trainer-side bug.
- this is not evidence that the GCS artifact contains the wrong weights.
- the first trainer->rollout transfer succeeded over Arrow Flight; the failure is in rollout-side application of those weights inside vLLM/tpu_inference.

Why this can break now even though a previous run worked:
- the previous successful run used the Hugging Face repo id directly in the rollout model loader path.
- the current path uses a regional GCS-exported Hugging Face artifact.
- that should be equivalent semantically, but it is clearly not equivalent in the vLLM/tpu_inference architecture-resolution path.
- so the most likely explanation is not "the weights changed"; it is "the model-loading backend takes a different code path when the model comes from `gs://...`".

What is definitely known vs still speculative:
- known:
  - the cached GCS artifact's `config.json` is correct for Llama:
    - `architectures: ["LlamaForCausalLM"]`
    - `model_type: "llama"`
  - rollout logs still show:
    - `Resolved architecture: MistralForCausalLM`
  - `tpu_inference` then reports:
    - `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
  - the crash happens after that, during weight application.
- not yet proven:
  - whether `MistralForCausalLM` is the direct cause of the `flat_state` failure, or just a second symptom of the same wrong model-loader path.
  - whether the plain `dict` is introduced by Marin before the vLLM RPC boundary, or inside `tpu_inference` after our worker extension hands off an `nnx.State`.

Postmortem on the previous local fix attempt:
- local patch switched `WorkerExtension.update_weight()` to call public `self.sync_weights(...)` instead of the internal `_sync_weights(...)`.
- that did not resolve the live failure.
- live stack still goes through `TPUWorker.sync_weights(...)` and then hits `TPURunner._sync_weights(...)` with a plain `dict`.
- conclusion:
  - the prior patch was directionally reasonable but incomplete.
  - the bad type is being introduced deeper than the simple wrapper change accounted for.

Live operational wrinkle discovered while debugging:
- the first `r10` subtree died because the root worker itself was preempted/timed out:
  - descendant jobs show `Parent task preempted`
- Iris automatically retried the root task on a fresh worker.
- current active subtree is:
  - `/ahmed/iris-rl-oom-b120-uc1-r10/rl_testing-l31-8bi-b120-20260324-081324_077ed54f-a96952fc/...`

Working hypotheses, ordered by value:
1. GCS-path model loading is selecting the wrong vLLM/tpu_inference architecture path.
   - strongest evidence: correct Llama config in GCS, but runtime resolves `MistralForCausalLM`.
   - likely mechanism: `vllm-tpu` architecture inference differs for `gs://` model ids vs canonical HF ids.
2. The rollout async `collective_rpc` path still serializes/deserializes into a plain nested dict somewhere inside `tpu_inference`.
   - strongest evidence: `TPURunner._sync_weights()` receives something without `.flat_state()`.
   - this could be independent of the architecture mismatch, or architecture mismatch could force a fallback model runner that expects a different object shape.
3. The prior successful run never exercised this exact backend combination.
   - possible differences:
     - HF repo id vs `gs://` artifact
     - sync vs async vLLM subpath
     - slightly different `vllm-tpu` / `tpu_inference` behavior around model registration/fallback
4. The unconditional Qwen2 registry patch is noise, not root cause.
   - logs show Qwen2 registration because Marin patches that class on import.
   - there is no current evidence that Qwen2 registration is what turns Llama into Mistral.
   - keep this lower priority unless source inspection proves registry collision.

Concrete experiments to run next:
1. Inspect pinned `vllm-tpu==0.13.2.post6` and `tpu-inference==0.13.2.post6` source for:
   - model architecture resolution from local/GCS HF exports
   - `collective_rpc("update_weight")`
   - `TPUWorker.sync_weights`
   - `TPURunner._sync_weights`
2. Add rollout-worker instrumentation before the failing handoff:
   - log `type(new_state)`
   - log `hasattr(new_state, "flat_state")`
   - log a compact `repr(type(self))`
   - log canonical model name vs model path
3. Add worker-side instrumentation inside `WorkerExtension.update_weight()`:
   - confirm the converted object is actually `nnx.State` on the remote worker just before `self.sync_weights(...)`.
4. Compare architecture resolution explicitly:
   - minimal local or remote probe that loads:
     - `meta-llama/Llama-3.1-8B-Instruct`
     - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
   - record what `vllm-tpu` resolves as the architecture in each case.
5. If GCS path is the discriminator, force canonical HF config resolution while keeping GCS weights/checkpoint bytes.
   - target outcome: preserve regional cached model artifact but stop vLLM from misclassifying the architecture.
6. Only after the above: relaunch and re-babysit toward the first real rollout batch and trainer step advance.

Current plan for the next agent after compaction:
- continue updating this logbook only after each verified live observation or code change.
- keep babysitting the active job tree and preserve exact job ids.
- prioritize source inspection + logging instrumentation over speculative refactors.
- success condition remains unchanged:
  - rollout survives the first trainer weight update,
  - rollout writes real batches,
  - trainer consumes them and advances,
  - then keep pushing toward a sustained 500-step run.

## 2026-03-24T08:18Z — stronger root-cause diagnosis from upstream source inspection

New verified facts from source inspection and artifact inspection:
- upstream `tpu_inference` `TPUWorker.sync_weights(...)` is a thin wrapper:
  - it forwards directly to `self.model_runner._sync_weights(updated_weights=..., mappings=..., transpose_keys=..., reshard_fn=...)`
- upstream `TPURunner._sync_weights(...)` then calls:
  - `transfer_state_with_mappings(src_state=updated_weights, tgt_state=self.state, ...)`
- upstream `transfer_state_with_mappings(...)` requires both:
  - `src_state.flat_state()`
  - `tgt_state.flat_state()`

Implication:
- the live `AttributeError: 'dict' object has no attribute 'flat_state'` can come from either:
  - the incoming RL update state, or
  - the model runner's existing `self.state`
- the traceback text from the worker shows the failure is on `tgt_state.flat_state()`.
- therefore the bad object is most likely `self.state`, not the trainer weight payload.

This materially changes the diagnosis:
- the local `WorkerExtension.update_weight -> self.sync_weights(...)` patch was not enough because the incoming update object was probably not the main problem.
- the real problem is that the rollout worker booted a model runner whose `self.state` is not a JAX `nnx.State`.

Why that matters:
- the same rollout logs show:
  - `Resolved architecture: MistralForCausalLM`
  - `Resolved MODEL_IMPL_TYPE 'auto' to 'flax_nnx'`
  - then:
    - `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
- if the worker falls back to the vLLM-native PyTorch path, RL hot-reload is no longer targeting the JAX `flax_nnx` state shape that `_sync_weights(...)` expects.
- this cleanly explains why the trainer keeps working while rollout dies at first weight apply.

What this rules out:
- not a corrupted model artifact in GCS:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/config.json` is correct and says:
    - `architectures: ["LlamaForCausalLM"]`
    - `model_type: "llama"`
  - the artifact also contains tokenizer/config/params files, not just safetensors.
- not primarily an Arrow Flight transport issue:
  - rollout fetches the initial weight payload successfully before dying.

Current best hypothesis:
1. GCS-backed `runai_streamer` model loading is causing `vllm-tpu` to infer the wrong architecture (`MistralForCausalLM`) from the streamed local cache path.
2. That wrong architecture triggers the PyTorch fallback path in `tpu_inference`.
3. RL hot-reload then fails because `_sync_weights(...)` expects JAX `nnx.State` for the runner state, but the fallback runner state is a plain dict-like object.

Important operational observation:
- for `inference_type == "vllm"`, the rollout worker does not actually load the initial checkpoint into a Levanter model.
- `_build_models()` logs the checkpoint path, but then sets:
  - `self._policy_model = None`
- this means the rollout worker fundamentally relies on the first Arrow Flight update to populate usable weights.

Consequence for fix design:
- we do not need full model weights from GCS at rollout startup.
- we need only correct model metadata/config/tokenizer so vLLM boots the right architecture.
- that suggests a cleaner path than `runai_streamer` for RL:
  - stage only the HF metadata files locally from the cached GCS artifact,
  - start vLLM with `load_format="dummy"` against that local metadata directory,
  - then let the first Arrow Flight update populate actual weights.

Next experiment/fix candidate:
1. stop using `runai_streamer` for RL rollout startup.
2. for object-store cached model artifacts, stage a small local metadata directory containing:
   - `config.json`
   - tokenizer files
   - generation config / special tokens files as needed
3. point vLLM `model_name` and tokenizer at that local metadata dir.
4. use `load_format="dummy"` so startup does not depend on object-store weight loading.
5. relaunch and verify:
   - architecture resolves as `LlamaForCausalLM`, not `MistralForCausalLM`
   - no PyTorch fallback warning
   - first trainer weight update completes
   - rollout starts producing batches.

## 2026-03-24T08:24Z — rollout-startup patch prepared to bypass `runai_streamer` in the RL hot-reload path

Code change landed locally:
- `lib/marin/src/marin/rl/rollout_worker.py`
  - added a narrow rollout-side startup path for `inference_type == "vllm"` with `inflight_weight_updates=True`:
    - if `model_name` is an object-store URI (`gs://` or `s3://`),
    - copy only small HF metadata files from the cached model artifact into a local temp cache:
      - `config.json`
      - tokenizer files
      - generation / special-tokens metadata
      - small index / params metadata
    - replace the vLLM startup config with:
      - `model_name=<local_metadata_dir>`
      - `load_format="dummy"`
- rationale:
  - rollout startup does not need full checkpoint weights because vLLM rollout workers in this RL path start with `self._policy_model = None` and wait for the first Arrow Flight update anyway.
  - avoiding `runai_streamer` should avoid the GCS-streamed architecture misclassification and the resulting PyTorch fallback path.

Validation:
- `python3 -m compileall lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_rollout_worker.py` -> success
- `uv run pytest -q tests/rl/test_rollout_worker.py tests/rl/test_inference_ctx.py tests/rl/test_rl_experiment_utils.py` -> `20 passed`
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_rollout_worker.py .agents/logbooks/iris-rl-codex.md` -> `OK`

New unit coverage:
- `tests/rl/test_rollout_worker.py`
  - verifies the metadata staging helper copies HF metadata files from a mocked remote model artifact.
  - verifies inflight remote vLLM startup rewrites config to local metadata + `load_format="dummy"`.

Immediate next action:
- stop stale live job tree `r10` because it does not include this patch.
- relaunch a fresh isolation run.
- verify in the new rollout logs:
  - no `Resolved architecture: MistralForCausalLM`
  - no `Falling back to vLLM-native Pytorch definition`
  - first trainer weight update completes and unblocks rollout generation.

## 2026-03-24T08:29Z — `r11` live result: local metadata startup patch deployed, but rollout still falls back to PyTorch and dies on first hot-reload

Live job under observation:
- root: `/ahmed/iris-rl-oom-b120-uc1-r11`
- status at observation time: root/coordinator/train/rollout all still `RUNNING`
- important nuance: this is not a healthy state; rollout remains alive only because the foreground thread is still waiting for the first weight transfer after the background sync loop has already crashed.

What was verified live in the rollout logs:
- the new rollout-startup patch is definitely deployed:
  - `Using local staged vLLM metadata for inflight rollout startup: gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f -> /tmp/marin-rl-vllm-metadata/fa6f62475ed0ccd0`
- despite that patch, vLLM still resolves the model as:
  - `Resolved architecture: MistralForCausalLM`
- that still triggers the same TPU-inference fallback:
  - `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
- the rollout worker then successfully receives the initial trainer payload:
  - `Received new weights from step -1`
- but the first hot-reload still crashes in the background sync loop with the same core failure:
  - `tgt_state.flat_state()`
  - `AttributeError: 'dict' object has no attribute 'flat_state'`
  - `Background weight sync loop crashed`
- after that crash, the rollout foreground thread keeps logging:
  - `Still waiting for first weight transfer ...`

What this means:
- the local-metadata + `load_format="dummy"` patch changed the startup path, but it did not fix the architecture selection problem.
- this strongly weakens the earlier hypothesis that `runai_streamer` or direct GCS weight streaming was the primary source of the misclassification.
- the more robust diagnosis is now:
  1. something in the rollout vLLM startup path is still causing the engine to pick `MistralForCausalLM` for a Llama 3.1 8B artifact,
  2. that architecture is not registered in `tpu_inference`'s JAX-native path,
  3. vLLM therefore falls back to the native PyTorch runner,
  4. RL hot-reload later calls into `_sync_weights(...)`, which expects JAX `nnx.State` on the target side,
  5. the target runner state is dict-like instead, so the first hot-reload dies on `tgt_state.flat_state()`.

Postmortem on the user question "how could this regress if the run worked before?":
- the evidence no longer supports the simple story "switching from HF to cached GCS broke rollout loading".
- the GCS artifact is still correct and contains a proper Llama config.
- the stronger explanation is that this RL rollout path was already relying on a fragile architecture-detection / model-registry interaction inside `vllm-tpu` + `tpu_inference`.
- once we changed startup behavior around cached model resolution, we started exercising that fragile path consistently enough to expose it.
- the trainer continues to work because the trainer side uses the Levanter/JAX stack directly; the failure is specifically in the rollout-side inference engine bootstrap + hot-reload path.

Current ranked hypotheses:
1. most likely: Marin's local patching of the TPU inference registry is incomplete for this rollout path, and `MistralForCausalLM` needs to be explicitly aliased to the JAX Llama implementation used for hot-reload.
2. possible: some vLLM-side config canonicalization rewrites the local staged Llama metadata into a Mistral architecture before `tpu_inference` sees it.
3. less likely: the model registry is correct, but the rollout inflight worker constructs the wrong target state object for JAX hot-reload after fallback.

Concrete next experiments:
1. inspect `marin.rl.environments.inference_ctx.vllm` and upstream patch hooks to see exactly where `Qwen2ForCausalLM` is registered today and whether `MistralForCausalLM` can be safely mapped to the same JAX Llama implementation.
2. add explicit instrumentation or a narrow unit test around the registry patch so we can prove whether `MistralForCausalLM` is absent before engine startup.
3. if the alias is technically sound, patch the rollout startup path to register or coerce `MistralForCausalLM` onto the JAX-native Llama implementation before engine init.
4. relaunch and verify the decisive success criteria:
   - no `Resolved architecture: MistralForCausalLM`, or at minimum no JAX-registry fallback warning,
   - no `Falling back to vLLM-native Pytorch definition`,
   - no `tgt_state.flat_state()` crash,
   - rollout actually leaves `Waiting for first weight transfer` and starts generating batches.

Operational note for the next agent after compaction:
- `r11` is still running but is already logically broken.
- do not treat the current all-`RUNNING` Iris state as success.
- the decisive signal is the rollout log sequence above, not the root job state.

## 2026-03-24T08:37Z — next planned experiment after `r11`: inspect and patch the rollout-side JAX registry path

Hypothesis:
- the remaining live `r11` failure is not a transport bug and not a bad GCS artifact.
- the likely bug is that Marin's TPU-inference registry patch layer is incomplete for this rollout path: `MistralForCausalLM` is not being kept on the JAX-native Llama implementation, so rollout falls back to the PyTorch path and then dies on first hot-reload.

Planned work before the next relaunch:
1. inspect `marin.rl.environments.inference_ctx.vllm` and related rollout inference code for current registry patch behavior.
2. confirm whether Marin only patches `Qwen2ForCausalLM` today.
3. if so, add a narrow experiment patch that keeps `MistralForCausalLM` on the JAX-native Llama implementation used by RL hot-reload.
4. add a focused unit test around the registry patch contract.
5. stop broken live run `r11`, relaunch, and verify whether rollout survives the first trainer weight update.

Success criteria for this experiment:
- no rollout-side fallback to `vLLM-native Pytorch definition`,
- no `tgt_state.flat_state()` crash on first weight update,
- rollout exits the `Waiting for first weight transfer` loop and begins producing batches.

## 2026-03-24T08:45Z — registry-alignment patch prepared: keep `MistralForCausalLM` on the JAX Llama path

Verified upstream facts before patching:
- pinned packages in `uv.lock` are:
  - `tpu-inference==0.13.2.post6`
  - `vllm-tpu==0.13.2.post6`
- source inspection of the pinned wheels showed:
  - `vllm.model_executor.models.registry` already maps `MistralForCausalLM` to the Llama implementation on the vLLM side.
  - `tpu_inference.models.common.model_loader._get_model_architecture(...)` only registers `LlamaForCausalLM` for the JAX-native Llama path, not `MistralForCausalLM`.
- this explains the live behavior cleanly:
  - vLLM accepts `MistralForCausalLM` as a Llama-family model,
  - but `tpu_inference` rejects that architecture string and falls back to the PyTorch path,
  - then RL hot-reload crashes because the target state is no longer the JAX `nnx.State` that `_sync_weights(...)` expects.

Local patch prepared:
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
  - widen the existing TPU-inference registry shim so Marin registers:
    - `Qwen2ForCausalLM -> Qwen2ForCausalLM`
    - `MistralForCausalLM -> LlamaForCausalLM`
- rationale:
  - this does not invent a new model mapping.
  - it aligns Marin's JAX-side registry with the existing vLLM-side architecture routing in the pinned upstream wheel.

Focused validation:
- `uv run pytest -q tests/rl/test_inference_ctx.py tests/rl/test_rollout_worker.py tests/rl/test_rl_experiment_utils.py` -> `21 passed`
- added unit coverage in `tests/rl/test_inference_ctx.py` proving the registry patch registers the Mistral alias.

Immediate next action:
1. stop broken live run `r11`.
2. relaunch a fresh isolation run with this patch.
3. verify the decisive rollout-side log deltas:
   - no `Falling back to vLLM-native Pytorch definition`,
   - no `tgt_state.flat_state()` crash,
   - rollout leaves the first-weight wait loop and starts producing batches.

## 2026-03-24T08:43Z — `r12` first live result: registry alignment changed the rollout path materially

Live run:
- `/ahmed/iris-rl-oom-b120-uc1-r12`

Verified rollout-side deltas relative to `r11`:
- `Resolved architecture: MistralForCausalLM` still appears. That line alone was not the true discriminant.
- the important change is that rollout no longer emits:
  - `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
- instead, the rollout logs now show the new registry path is active:
  - Marin patch log: `Patching tpu_inference to support MistralForCausalLM`
  - vLLM/tpu_inference log: `Registered JAX model MistralForCausalLM with tpu_inference and vLLM registries.`
  - model-loader log: `Resolved MODEL_IMPL_TYPE 'auto' to 'flax_nnx'`
- the async engine initializes successfully after that, and rollout reaches:
  - `Starting background weight sync loop`
  - `Waiting for first weight transfer before starting inference ...`

Interpretation:
- this is the first live evidence that the prior root cause was real.
- the JAX/PyTorch fallback mismatch was not just correlated noise; aligning the Mistral architecture registration changed the execution path exactly where expected.
- `r12` has not yet proven full correctness, because the first live weight transfer still needs to complete without a new crash.

Immediate next checkpoint:
1. wait for train-side Arrow Flight server advertisement to become visible to rollout.
2. verify rollout receives step `-1` or the first real trainer weight id.
3. verify there is still no `flat_state` crash during hot-reload.
4. only after that, judge whether the rollout-side bug is actually fixed.

## 2026-03-24T08:44Z — `r12` passes the old first-weight crash boundary; possible next sync bug under observation

New verified live fact:
- `r12` cleared the exact boundary that killed `r10`/`r11`.
- rollout logs now show:
  - `Registered JAX model MistralForCausalLM with tpu_inference and vLLM registries.`
  - `Received new weights from step -1`
  - `receive_weights: update_model complete`
- critically, there is still no:
  - `Falling back to vLLM-native Pytorch definition`
  - `AttributeError: 'dict' object has no attribute 'flat_state'`
  - `Background weight sync loop crashed`

Interpretation:
- the previous rollout-side bug is very likely fixed.
- the `MistralForCausalLM` registry alignment patch changed the behavior at the exact failing point, and the old `flat_state` crash did not recur on the first live weight application.

Possible new issue now under observation:
- after rollout receives step `-1`, the foreground thread still logs `Still waiting for first weight transfer ...`
- if that persists, the next bug is probably a rollout-side synchronization/ready-signal issue rather than a model-loader bug.

Next immediate check:
1. determine whether rollout eventually logs a successful first-weight handoff and begins generating.
2. if not, inspect the rollout worker's first-weight-ready/event signaling path.
3. keep the run alive while this is still potentially recoverable; only stop it if the wait loop is clearly stuck.

## 2026-03-24T08:45Z — `r12` clears the first live hot-reload handoff end-to-end

Additional verified live result:
- the temporary concern about rollout still waiting after receiving step `-1` resolved without intervention.
- rollout eventually logged:
  - `First weight transfer complete, inference can proceed`

Combined with the earlier observations, `r12` has now cleared all of the rollout-side boundaries that were failing before:
1. no JAX-registry fallback warning,
2. no `flat_state` crash during first weight application,
3. successful first live weight reception,
4. successful first-weight-ready handoff back to the rollout foreground thread.

Interpretation:
- the Mistral-to-Llama JAX registry alignment patch appears to have fixed the previously dominant rollout-side bug.
- the next remaining question is no longer rollout startup or first hot-reload correctness.
- the next question is whether rollout actually generates batches and whether trainer consumes them and advances.

Next immediate checkpoint:
1. watch for rollout generation / rollout writer activity.
2. watch for trainer log lines indicating initial rollouts were received.
3. keep pushing this same run toward actual training-step advance before making any further code changes.

## 2026-03-24T08:46Z — `r12` is alive past first hot-reload; rollout now appears generation/compile-bound

Latest verified live state:
- whole `r12` job tree is still `RUNNING`.
- rollout has already crossed:
  - first live JAX-model bootstrap,
  - first Arrow Flight weight reception,
  - first-weight-ready handoff,
  - entry into the rollout inference loop.
- trainer is still waiting on initial rollouts from step `-1`.

Newest concrete signal from rollout logs:
- after `Evaluating 1 lessons`, the logs show TPU/XLA memory-space-assignment warnings for `ragged_paged_attention.*`:
  - requested scoped Vmem `104857600` bytes
  - max valid bytes `67043328`
  - runtime reports it is lowering the scoped Vmem rather than aborting
- rollout remains alive after those warnings and continues polling for new weights with `step > -1`.

Current interpretation:
- the original rollout code bug appears fixed.
- the run has progressed into a later stage where either:
  1. generation is still compiling / warming up under TPU inference, or
  2. there is a new rollout-generation stall unrelated to the original JAX/PyTorch registry mismatch.

Immediate next checkpoint after compaction if needed:
1. keep watching for the first actual rollout write / trainer batch receipt.
2. if logs remain quiet while the job stays alive, inspect whether `ragged_paged_attention` scoped-vmem warnings are materially slowing or stalling first generation.
3. only make further code changes after we can distinguish compile latency from a true silent stall.

## 2026-03-24T17:24Z — `r12` terminal postmortem: rollout bug fixed, run ultimately died on train-side OOM

Final verified Iris state for `/ahmed/iris-rl-oom-b120-uc1-r12`:
- root job: `JOB_STATE_FAILED`
- executor child `rl_testing-l31-8bi-b120-20260324-084018_9d319663-b292440b`: `JOB_STATE_FAILED`
- coordinator child `rl-l31-8bi-b120-20260324-084018`: `JOB_STATE_FAILED`
- train child `rl-l31-8bi-b120-20260324-084018-train`: `JOB_STATE_FAILED`
- rollout child `rl-l31-8bi-b120-20260324-084018-rollout-0`: `JOB_STATE_KILLED`

Verified root cause from `iris job bug-report` / descendant statuses:
- the train child was OOM-killed with `Exit code 137: OOM killed (container exceeded memory limit)`.
- after the train child failed, the coordinator failed because `...-train finished with status failed`.
- then the executor child failed, and finally the root wrapper failed with `RuntimeError: 1 step(s) failed`.
- the rollout child was killed afterward as cleanup / failure propagation, not as the primary root cause.

Important interpretation:
- W&B showing one rollout/trainer step is not contradictory.
- `r12` made real progress before the terminal failure:
  - rollout booted the JAX-backed vLLM path,
  - first weight transfer completed,
  - rollout entered inference / lesson evaluation,
  - trainer and rollout both emitted live metrics.
- this means the previously dominant rollout-side bug from `r10` / `r11` did not recur on `r12`.
- the terminal blocker for `r12` is now back on the original train-side OOM axis.

What the Iris dashboard page is showing:
- the child-jobs page for the executor step correctly shows the wrapper child as `FAILED` with exit `1`.
- that page is surfacing the step-level failure, not the deepest root cause directly.
- the deeper root cause lives one level below that page in the nested train child, which failed with OOM.

Revised working conclusion after `r12`:
1. switching model bootstrap from live HF reads to the executor-managed GCS model artifact did not reintroduce the old rollout crash.
2. the Mistral/JAX registry alignment patch materially fixed the rollout-side hot-reload bug.
3. the active blocker for getting a stable 500-step run is again train-side memory behavior under the longer live path, not rollout model loading.

Next experiments from this point:
1. inspect train logs immediately before the `Exit 137` to determine whether the OOM is tied to checkpoint save, rollout consumption, or optimizer/state growth.
2. compare `r12` train memory behavior against the earlier pre-GCS successful run once the exact reference run is identified.
3. only revisit rollout if a later run shows a new rollout-side failure mode; `r12` no longer supports rollout-loader mismatch as the leading hypothesis.

## 2026-03-24T17:32Z — old `/ahmed/iris-rl-v5p-3` success was a different launch topology and a much lighter envelope

Verified from `iris job bug-report /ahmed/iris-rl-v5p-3`:
- root job entrypoint was `python experiments/exp_iris_rl_direct.py`
- root state is `killed` only because it was manually terminated later
- descendant jobs:
  - `/ahmed/iris-rl-v5p-3/rl-iris-rl-direct-20260323-013808-train`: `succeeded`
  - `/ahmed/iris-rl-v5p-3/rl-iris-rl-direct-20260323-013808-rollout-0`: `succeeded`

This explains the dashboard difference:
- old screenshot: train + rollout appear directly under the root job because `exp_iris_rl_direct.py` ran the coordinator directly in the outer process.
- current isolation runs: root job is an executor wrapper, so the tree is:
  - root job
  - executor child `rl_testing/...`
  - coordinator child `rl-...`
  - train / rollout children underneath that
- so the train / rollout children are not gone; they moved one level deeper because the launch path changed.

Relevant code-history boundary:
- `9c7bef02d`: introduced `experiments/exp_iris_rl_direct.py`
- `5d1d20d47`: changed it to run `_run_rl_coordinator(...)` directly in the outer job process
- `7e5137314`: changed the direct script from the tiny debug envelope to a production-like 500-step envelope

Important historical distinction:
- the successful `/ahmed/iris-rl-v5p-3` run was against the *earlier* direct-script envelope, not the later 500-step direct script.
- that successful version used:
  - `num_train_steps=5`
  - `train_batch_size=64`
  - `max_seq_len=1024`
  - `max_output_tokens=512`
  - `n_prompts=4`
  - `max_weight_transfer_wait_time=300` (blocking)
  - direct HF repo names (`MODEL_NAME`) instead of executor-managed GCS artifacts
- so `/ahmed/iris-rl-v5p-3` is not an apples-to-apples proof that the current 500-step executor/GCS path should work unchanged.

Current interpretation after this history check:
1. the dashboard topology change is expected from switching from `exp_iris_rl_direct.py` to executor-wrapped experiments.
2. the old success mainly proves the small direct path worked end-to-end.
3. the meaningful behavior boundaries to compare are:
   - direct small run (`5d1d20d47` / `9c7bef02d` lineage),
   - direct production-like run (`7e5137314` lineage),
   - executor + GCS artifact run (`9d290793f` and later).
4. `b8c0f42` is only a logbook commit, so it is not a useful bisect boundary by itself.

## 2026-03-24T17:38Z — divergence-hunt experiment plan from old direct success to current executor/GCS runs

Goal:
- identify the first change that moves us from the old known-good direct 5-step run into the current failure regimes.
- isolate three dimensions separately:
  1. launch topology,
  2. model bootstrap source,
  3. runtime envelope / memory pressure.

Non-negotiable rule for this hunt:
- do not change more than one of those dimensions in a single experiment unless a previous experiment already proved the earlier dimensions are clean.

### Controlled dimensions

Reference A: old successful direct small run
- topology: direct coordinator in outer job (`exp_iris_rl_direct.py` style)
- model source: raw HF repo ids
- envelope:
  - `num_train_steps=5`
  - `train_batch_size=64`
  - `max_seq_len=1024`
  - `max_output_tokens=512`
  - `n_prompts=4`
  - `max_weight_transfer_wait_time=300`

Dimension 1: topology
- `direct`: outer job runs `_run_rl_coordinator(...)` directly
- `executor`: outer job runs `executor_main(...)` -> executor step -> coordinator

Dimension 2: model source
- `hf`: raw repo path / direct HF bootstrap
- `gcs`: executor-managed regional model artifact path

Dimension 3: envelope
- `small`: old 5-step debug envelope
- `prod`: 500-step / bs=1024 / 2048-token production-like envelope

### Success criteria for each probe

Minimum success:
- train child and rollout child both start
- first weight transfer completes
- rollout writes at least one rollout batch
- trainer advances at least one completed step
- no terminal job failure before that point

Strong success:
- run finishes cleanly for small envelope
- for production envelope, run survives long enough to cross the first checkpoint-save boundary without OOM

### Experiment matrix

E1. Direct + HF + small
- purpose: re-validate the old success path on current HEAD.
- expected value: if this fails now, the regression is in core RL runtime code, not executor/GCS.
- interpretation:
  - pass: current core runtime can still do the old known-good path.
  - fail: bug was introduced after the original success, independent of executor/GCS.

E2. Direct + GCS + small
- change from E1: model source only.
- purpose: isolate whether switching bootstrap/checkpoint/tokenizer from HF ids to regional GCS artifacts changes behavior even in the old small topology.
- interpretation:
  - pass: GCS artifact bootstrap is not the divergence by itself.
  - fail: model artifact pathing/loading changed semantics and needs focused debugging.

E3. Executor + GCS + small
- change from E2: topology only.
- purpose: isolate whether the executor wrapper / step indirection introduces a functional regression when the envelope is still tiny.
- interpretation:
  - pass: executor topology is not the divergence for small runs.
  - fail: wrapper-step/runtime-path/job-tree behavior is the regression boundary.

E4. Direct + GCS + prod
- change from E2: envelope only.
- purpose: isolate whether the heavy 500-step production-like envelope alone is sufficient to reproduce train-side OOM without executor topology.
- interpretation:
  - pass: production envelope can survive in direct topology, so executor layering or path differences still matter.
  - fail: heavy envelope itself is enough to trigger the active OOM regime.

E5. Executor + GCS + prod
- this is the current family of failing runs.
- purpose: only run after E3 and E4 to determine whether the failures are additive or already explained by one dimension.
- interpretation:
  - if E4 already fails the same way, executor is likely not the primary cause.
  - if E4 passes but E5 fails, executor/topology/path resolution is still implicated.

### Fast-follow experiments if E4 or E5 fail by OOM

OOM-1. Direct + GCS + prod + checkpoint disabled
- set `checkpointer_save_interval` very large.
- purpose: test the standing checkpoint-coupled OOM hypothesis.

OOM-2. Direct + GCS + prod + reduced batch
- reduce `train_batch_size` from `1024` to `512`.
- purpose: test whether train resident memory alone is already too high before checkpoint overlap.

OOM-3. Direct + GCS + prod + blocking sync
- set `max_weight_transfer_wait_time=300`.
- purpose: test whether the non-blocking sync / restart churn regime is materially increasing memory pressure or instability.

### Decision tree

If E1 fails:
- stop blaming executor/GCS.
- bisect core RL runtime changes after the original direct success.

If E1 passes and E2 fails:
- focus on artifact-path semantics:
  - config loading,
  - tokenizer loading,
  - vLLM load format,
  - checkpoint metadata expectations on GCS.

If E2 passes and E3 fails:
- focus on executor wrapper effects:
  - runtime path materialization,
  - nested job environment differences,
  - output-path / checkpoint-path indirection,
  - failure propagation / resource differences between outer step and direct root.

If E3 passes and E4 fails:
- active blocker is the heavy production envelope itself.
- prioritize checkpoint-coupled train memory investigation.

If E4 passes and E5 fails:
- executor/topology becomes the highest-value suspect again.

### Concrete implementation tasks for next agent

1. create a small direct-with-GCS probe script by adapting `exp_iris_rl_direct.py` to use the executor-managed model artifact path while keeping the old small envelope.
2. create a small executor-with-GCS probe script by adapting `exp_iris_rl_debug.py` to keep the old small envelope and current GCS/bootstrap wiring.
3. keep naming explicit so runs are self-identifying:
   - `direct-hf-small`
   - `direct-gcs-small`
   - `exec-gcs-small`
   - `direct-gcs-prod`
   - `exec-gcs-prod`
4. after each run, record:
   - job tree topology,
   - first weight transfer result,
   - first rollout write,
   - first trainer step,
   - terminal cause if any.

Current best hypothesis ordering before running this matrix:
1. the old success primarily depended on the much smaller envelope.
2. the current executor/GCS path fixed some rollout bugs but is not yet exonerated.
3. checkpoint-coupled train OOM is still the leading blocker for production-like runs.

## 2026-03-24T17:49Z — launching E1 (`direct + HF + small`) as the first divergence probe

User requested execution of `E1` now.

Planned probe:
- topology: direct coordinator in outer Iris job
- model source: raw HF ids (`meta-llama/Llama-3.1-8B-Instruct`)
- envelope:
  - `num_train_steps=5`
  - `train_batch_size=64`
  - `max_seq_len=1024`
  - `max_output_tokens=512`
  - `n_prompts=4`
  - `max_weight_transfer_wait_time=300`
- root placement: explicit `--region us-central1`

Implementation:
- added `experiments/exp_iris_rl_regression_direct_hf_small.py`
- this preserves the old direct topology by calling `_run_rl_coordinator(...)` directly
- unlike the current `exp_iris_rl_direct.py`, it does not use the production-like 500-step envelope

Monitor plan:
- startup stabilization check after submission
- then 5-minute cadence (`300s`) per user request
- success criteria:
  - train child starts
  - rollout child starts
  - first weight transfer completes
  - rollout writes at least one batch
  - trainer advances at least one step

Submission details:
- root job id: `/ahmed/iris-rl-e1-direct-hf-small-20260324-105133`
- root submit command:
  - `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-e1-direct-hf-small-20260324-105133 --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_hf_small.py`

Immediate startup result:
- root job did not start within the first stabilization window.
- current state: `JOB_STATE_PENDING`
- pending reason:
  - `Scheduler: No worker matches constraints ... constraints=['region', 'preemptible', 'reservation-job']`
  - `Autoscaler: Unsatisfied autoscaler demand: no_capacity: cpu_vm_e2_highmem_2_ondemand-us-central1-a=at_max_slices`

Interpretation:
- this is not yet a code failure in `E1`.
- the blocker is root-job placement: Iris CLI auto-tagged the outer job as non-preemptible in `us-central1`, and there is currently no matching ondemand CPU capacity there.
- child train / rollout jobs do not exist yet because the outer root coordinator process has not started.

Next monitor action:
1. keep the job pending under the user-requested 5-minute cadence.
2. if capacity clears, re-evaluate the actual E1 runtime behavior.
3. if capacity does not clear, consider changing only root placement semantics while keeping the direct/HF/small runtime path intact.

## 2026-03-24T18:00Z — E1 cleared the root-capacity block and is now in live startup

First 5-minute cadence check:
- root `/ahmed/iris-rl-e1-direct-hf-small-20260324-105133`: `JOB_STATE_RUNNING`
- child train job exists and is `JOB_STATE_RUNNING`
- child rollout job exists and is `JOB_STATE_RUNNING`

Observed startup logs:
- outer root logged:
  - `Running E1 direct + HF + small probe: direct-hf-small-20260324-175631`
  - `Submitted 2 child jobs (1 trainer + 1 rollout workers)`
- rollout startup has begun far enough to emit early vLLM import/runtime logs

Interpretation:
- the earlier pending state was pure root placement scarcity, not a structural failure of the direct/HF/small probe.
- E1 has now crossed the first real topology boundary:
  - direct root process started,
  - coordinator executed in the outer job,
  - train and rollout children materialized directly under the root job as expected.

Next monitor focus:
1. first weight transfer (`weight_id=-1` / server update).
2. first rollout write.
3. first trainer progress / completed step.

## 2026-03-24T18:05Z — E1 behaves like the old good path through the first real RL boundary

Second 5-minute cadence check shows `E1` is doing real work, not merely sitting in startup.

Verified live signals:
- trainer and rollout children both remain `JOB_STATE_RUNNING`
- trainer emitted:
  - `Progress on:train -/5 ...`
  - later `Progress on:train 1.00it/5.00it ...`
- Arrow Flight emitted:
  - `Updated server: weight_id=-1, params=291, servers=52`
- rollout emitted:
  - `Received new weights from step -1`
  - `First weight transfer complete, inference can proceed`
  - `Generated rollout with 10 groups ... step -1`
  - `Generated rollout with 4 groups ... step -1`

Interpretation:
- `E1` has already passed the core “old run was real” checkpoints on current HEAD.
- this strongly suggests current core RL runtime is still capable of the old direct/HF/small success path.
- therefore the first divergence is probably *not* in the basic direct small runtime itself.

Implication for the experiment matrix:
- if `E1` completes cleanly, the next highest-value probe is `E2` (`direct + GCS + small`), because that isolates model bootstrap/source while keeping the old successful topology and envelope.

## 2026-03-24T18:11Z — E1 no longer looks clean: rollout is stuck on Arrow Flight reconnect errors after step 1

Latest live status:
- root: `JOB_STATE_RUNNING`
- rollout child: `JOB_STATE_RUNNING`
- train child: `JOB_STATE_RUNNING`, but `preemption_count=2`

Verified progression before the new issue:
- trainer completed at least through step 1:
  - `Training step 0 completed ...`
  - `Training step 1 completed ...`
- weight transfer advanced through:
  - `weight_id=-1`
  - `weight_id=0`
  - `weight_id=1`
- rollout consumed updated weights and generated rollouts for:
  - step `-1`
  - step `0`
  - step `1`

New failure signal:
- rollout is now looping on repeated `pyarrow._flight.FlightUnavailableError`
- error shape:
  - `failed to connect to all addresses`
  - `tcp handshaker shutdown`
  - failing against multiple old Arrow Flight ports on trainer host `10.128.0.56`
- example affected trainer ports:
  - `20313`
  - `32685`
  - `2493`
  - `14463`

Current leading interpretation:
- this looks much more like trainer preemption/restart or Arrow Flight server churn than a model/bootstrap bug.
- evidence:
  - train child `preemption_count=2`
  - rollout is still trying to fetch from the old trainer-advertised server list on `10.128.0.56`
  - the old “small direct baseline always finishes cleanly” assumption is now false on current HEAD, at least under this run’s live preemption behavior.

What this means for the divergence hunt:
1. `E1` still proved that current HEAD can reproduce the old path through first weight transfer and multiple train steps.
2. but it also exposed a new candidate divergence boundary:
   - restart / preemption recovery under the direct small path may no longer be robust.
3. this is not the same as the prior rollout model-loader mismatch.
4. before moving to `E2`, we need to determine whether `E1` self-recovers after train preemption or remains permanently wedged on stale Arrow Flight clients.

Immediate next monitor action:
1. keep the job alive for one more cadence window to see if rollout reconnects after trainer recovery.
2. if not, capture this as a preemption/reconnect regression against the old small direct baseline.

## 2026-03-24T18:17Z — E1 did not self-recover; trainer restarted, rollout stayed wedged on stale Flight endpoints

One more 5-minute recovery window was enough to answer the key question.

Observed live facts:
- root still `JOB_STATE_RUNNING`
- train child still `JOB_STATE_RUNNING`
- rollout child still `JOB_STATE_RUNNING`
- train child still shows `preemption_count=2`
- trainer logs now show a fresh restart signature:
  - `Progress on:train -/5 ...` at `18:17:02Z`
- rollout logs continue the same error loop without recovery:
  - repeated `Failed to receive weights via Arrow Flight`
  - repeated `FlightUnavailableError`
  - repeated stale endpoint failures against old trainer host/ports on `10.128.0.56`

Critical interpretation:
- this is now strong evidence of a restart/preemption reconnect regression.
- current HEAD can reproduce the old small direct path through:
  - first weight transfer,
  - multiple rollout writes,
  - two completed train steps,
  - but it does *not* appear to recover cleanly once the trainer disappears and restarts.
- rollout keeps polling the old Arrow Flight client endpoints instead of switching to the trainer’s restarted server set quickly enough to continue.

Implication for divergence plan:
1. `E1` is partially successful:
   - baseline direct/HF/small path still works *until* trainer preemption/restart.
2. `E1` is also sufficient to expose a current regression:
   - direct small path is no longer robust under trainer restart/preemption.
3. this means the next highest-value work is probably not `E2` yet.
4. instead, inspect the restart/reconnect boundary:
   - how coordinator updates Arrow Flight server metadata after trainer restart,
   - whether rollout refreshes `self._flight_clients` on changed server ids/hosts,
   - whether preemption changed semantics compared with the old successful `/ahmed/iris-rl-v5p-3` run.

## 2026-03-24T18:23Z — E1 ultimately recovered and finished; current HEAD still supports the old direct/HF/small success path

Follow-up checks changed the interpretation materially. `E1` did not remain wedged permanently.

Final Iris state:
- root `/ahmed/iris-rl-e1-direct-hf-small-20260324-105133`: `JOB_STATE_SUCCEEDED`
- train child: `JOB_STATE_SUCCEEDED`
- rollout child: `JOB_STATE_SUCCEEDED`

Important live facts:
- train child finished successfully despite `preemption_count=2`
- rollout resumed after the earlier Flight reconnect turbulence and completed cleanly
- root coordinator finalized cleanly after both children finished
- W&B captured both train and rollout runs; rollout summary still shows a high `inference.failed_receives`, but that did not prevent completion

Terminal log signals:
- trainer reached `Progress on:train 5.00it/5.00it`
- rollout logged `Run state is 'completed', stopping rollout worker`
- rollout logged `Inference worker completed after generating 9 rollouts`
- coordinator logged `RL coordinator finished for run direct-hf-small-20260324-175631`

Revised interpretation:
- current HEAD can still execute the old successful topology and envelope:
  - direct coordinator in the root job
  - raw HF model identifiers
  - small 5-step debug envelope
- trainer preemption/restart causes visible Arrow Flight reconnect churn, but the system can recover well enough to finish this baseline run
- this makes `E1` a pass, not a failure

Implication for the divergence hunt:
1. the core direct/HF/small baseline is still viable on current HEAD
2. the next isolation target should move back to `E2` (`direct + GCS + small`)
3. if `E2` fails while `E1` passes, the first clean divergence is the model source / artifact path change rather than the direct topology itself

## 2026-03-24T18:26Z — Launch plan for E2 (`direct + GCS + small`)

Next controlled probe:
- topology: direct root/coordinator, same as `E1`
- model source: switch from raw HF identifiers to the cached regional artifact
- envelope: unchanged tiny 5-step debug run

Concrete model artifact under test:
- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`

Experiment goal:
- isolate whether replacing raw HF identifiers with the GCS-backed HF export changes startup, model resolution, rollout behavior, or trainer/rollout compatibility when everything else stays fixed

Execution note:
- submit `E2` first
- then re-check for any still-running `/ahmed/...rl...` job and stop it if one exists

## 2026-03-24T18:27Z — E2 launched; no other running RL job remained to kill

Submitted:
- root job: `/ahmed/iris-rl-e2-direct-gcs-small-20260324-112640`
- entrypoint: `uv run python experiments/exp_iris_rl_regression_direct_gcs_small.py`
- region: `us-central1`

Immediate post-submit state:
- root entered `JOB_STATE_RUNNING`
- task state was still `building` on the first status check

Post-submit kill sweep:
- explicitly searched for other `/ahmed/...rl...` jobs in `JOB_STATE_RUNNING`
- excluded the newly launched `E2` root job from that sweep
- result: `NO_OTHER_RUNNING_RL_JOBS`

Interpretation:
- the requested sequencing is satisfied:
  1. `E2` was launched first
  2. there was no second live RL job left to kill

## 2026-03-24T18:41Z — E2 is currently infra-blocked on rollout TPU capacity, not rollout code

Latest live status:
- root: `JOB_STATE_RUNNING`
- train child: `JOB_STATE_RUNNING`
- rollout child: `JOB_STATE_RUNNING` at the job level, but its only task is still `pending`

Observed behavior:
- trainer successfully loaded the initial model from the regional GCS artifact:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- trainer served initial weights `weight_id=-1`
- trainer is still waiting for initial rollouts with replay buffer size `0`
- rollout has produced no user-code logs because it never actually started executing

Rollout child bug report:
- first attempt died before user code:
  - `Worker marin-tpu_v5p_8-us-central1-a-20260324-1747-c9be5df2-worker-0 failed: Request timed out`
- current pending reason:
  - `Retrying (attempt 1, last: task_state_worker_failed)`

Autoscaler/controller evidence:
- unmet entry for rollout task:
  - reason `no_capacity: tpu_v5p_8-us-central1-a=backoff`
- `tpu_v5p_8-us-central1-a` has recent explicit create failures:
  - `There is no more capacity in the zone "us-central1-a"`
- `tpu_v5p_8-us-east5-a` is also blocked/backoff at the same time

Interpretation:
- `E2` is not yet a clean pass/fail on GCS-vs-HF behavior
- the current blocker is rollout TPU placement capacity
- because rollout never entered user code, this does not yet implicate the GCS-backed model path

Next monitor focus:
1. whether rollout retry eventually lands on capacity and starts logging
2. whether Iris escalates this into terminal failure
3. only after rollout actually starts can `E2` tell us whether GCS-backed startup diverges from `E1`

## 2026-03-24T19:12Z — After 30 minutes, E2 exposed the first clean GCS-path rollout regression

`E2` did eventually make progress beyond the initial capacity block.

What changed after the wait:
- rollout eventually obtained a TPU worker
- rollout failure count increased to `1`
- rollout child is currently still `JOB_STATE_RUNNING` because Iris is on a later retry attempt
- train child is still `JOB_STATE_RUNNING`, but now with `preemption_count=2`

Most important rollout bug-report evidence:
- attempt `0`: worker-level timeout before user code
- attempt `1`: real rollout process started and failed with:
  - `KeyError: 'No MODEL_MAPPING registered for model: gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f'`
- attempt `2`: currently running on a fresh TPU worker

Why this matters:
- this is the first actual code-level divergence unique to `E2`
- `E1` (`direct + HF + small`) did not fail in rollout model resolution
- `E2` (`direct + GCS + small`) does
- therefore the GCS artifact path is not fully accepted by the rollout/vLLM/tpu-inference model-resolution layer

Current live behavior on the latest retry:
- rollout logs show it got far enough to:
  - receive weights from step `-1`
- but it then degrades into repeated Arrow Flight transport failures against trainer-hosted ports
- trainer remains stuck waiting for initial rollouts and is not making train-step progress

Interpretation:
- `E2` is no longer just infra-blocked
- it has now produced a real regression signal:
  - rollout stack expects a canonical model identifier that maps through `MODEL_MAPPING`
  - passing the raw `gs://...` model path breaks that assumption on at least one rollout startup path
- later Flight errors are downstream noise once rollout/train are out of sync; the first meaningful bug is the `MODEL_MAPPING` failure

Next debugging target:
1. find where rollout/vLLM/tpu-inference derives architecture/model-name from `config.model_name`
2. make the GCS-backed path carry explicit canonical model metadata instead of relying on the raw `gs://...` string
3. rerun `E2` after that fix to see whether it then matches `E1`

## 2026-03-24T19:20Z — Detailed postmortem on E2 rollout logs: GCS artifact metadata is correct; direct probe wiring is not

Pulled the rollout `output.log` directly from W&B for:
- `marin-community/marin_iris_rl_debug/direct-gcs-small-20260324-182801-rollout-0`
- local copy: `scratch/wandb_e2/output.log`

This produced the clearest causal sequence so far and separates two issues that were getting conflated.

### 1. What the rollout W&B logs actually show

Rollout startup sequence from W&B:
- starts weight transfer client normally
- initializes file rollout writer normally
- patches TPU inference registry for:
  - `Qwen2ForCausalLM`
  - `MistralForCausalLM`
- launches vLLM with:
  - `model='gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f'`
  - `load_format='runai_streamer'`
- logs:
  - `Resolved architecture: MistralForCausalLM`
- streams the four safetensor shards successfully from GCS
- initializes engine successfully
- starts rollout worker successfully
- performs first Arrow Flight receive successfully:
  - `Received 291 params for weight_id -1`
  - `Received new weights from step -1`
- only then crashes during hot reload:
  - `KeyError: 'No MODEL_MAPPING registered for model: gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f'`

Critical takeaway:
- GCS loading itself is not the first failure
- `runai_streamer` does work for the initial model load
- the first meaningful code failure is during rollout hot weight reload, not during base checkpoint bootstrap

### 2. The GCS artifact metadata itself is correct Llama metadata

Checked the actual model artifact config directly:
- `gcloud storage cat gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/config.json`

Relevant contents:
- `"architectures": ["LlamaForCausalLM"]`
- `"model_type": "llama"`

This matters because it rules out the most alarming hypothesis:
- the GCS artifact is **not** mislabeled as Mistral
- the model export itself is not obviously corrupted or semantically wrong

### 3. Why rollout says `Resolved architecture: MistralForCausalLM`

This does **not** appear to come from the GCS config metadata. It comes from the TPU vLLM / tpu_inference compatibility path.

Repository evidence:
- [vllm.py:160](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:160) explicitly documents:
  - `vLLM already resolves Mistral onto the Llama implementation`
- [vllm.py:162](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:162) registers:
  - `"MistralForCausalLM": LlamaForCausalLM`
- [test_inference_ctx.py:346](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/tests/rl/test_inference_ctx.py:346) tests exactly that behavior:
  - `registry["MistralForCausalLM"] is FakeLlamaForCausalLM`

So the current best interpretation is:
- `MistralForCausalLM` in these rollout logs is a backend architecture alias inside TPU vLLM/tpu_inference
- it is **not** evidence that we loaded Mixtral
- it is **not** evidence that the underlying artifact stopped being Llama

This also explains why the old successful direct/HF rollout and the new direct/GCS rollout can differ here:
- `E1` used:
  - canonical HF model id
  - `load_format='dummy'`
  - vLLM logged `Resolved architecture: LlamaForCausalLM`
- `E2` used:
  - GCS object-store path
  - `load_format='runai_streamer'`
  - TPU vLLM logged `Resolved architecture: MistralForCausalLM`

So there are two different backend resolution paths:
1. `HF repo id + dummy loader` -> `LlamaForCausalLM`
2. `GCS path + runai_streamer` -> `MistralForCausalLM`

That discrepancy is real and worth understanding, but it is still separate from the actual `MODEL_MAPPING` crash.

### 4. The direct E2 probe has a wiring bug: it does not set `canonical_model_name`

This is the most concrete bug found so far.

In the direct `E2` probe:
- [exp_iris_rl_regression_direct_gcs_small.py:128](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_small.py:128)
- we pass:
  - `model_name=MODEL_PATH`
  - but do **not** pass `canonical_model_name`

In the inference context:
- [vllm.py:117](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:117)
- `self.canonical_model_name = inference_config.canonical_model_name or inference_config.model_name`

So for `E2`, `canonical_model_name` becomes the raw GCS path:
- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`

Then hot reload uses that canonical name for mapping lookup:
- [vllm.py:355](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:355)
- `mappings=MODEL_MAPPINGS[self.canonical_model_name]`

But `MODEL_MAPPINGS` only contains canonical HF-style names such as:
- `meta-llama/Llama-3.1-8B-Instruct`
- [vllm_utils.py:89](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm_utils.py:89)

And the fallback matcher does not understand this GCS path:
- [vllm_utils.py:110](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm_utils.py:110)
- it only special-cases `Qwen2.5`

Therefore the `KeyError` is expected from current code once rollout hot reload reaches mapping lookup with the raw GCS path.

### 5. Important scope boundary: the normal RL experiment path already does the right thing

The direct `E2` probe bug above does **not** automatically mean the main executor-backed RL experiment path has the same exact bug.

In the normal helper path:
- [rl_experiment_utils.py:277](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py:277)
- we pass:
  - `model_name=model_path`
  - `canonical_model_name=config.model_config.name`
- and that canonical name is the HF-style id:
  - `meta-llama/Llama-3.1-8B-Instruct`

So:
- the direct `E2` probe definitely has a bug
- the main executor-backed RL path may already avoid this particular one
- this means `E2` still did its job as a regression probe, but its current failure is partly probe-construction-specific

### 6. Updated interpretation for the experiment matrix

What `E2` has proven so far:
1. using the GCS-backed model path changes the vLLM/Tensor-TPU loader path compared with `E1`
2. that path can resolve architecture labels differently (`MistralForCausalLM` vs `LlamaForCausalLM`)
3. the direct probe currently fails because it did not preserve canonical model identity through hot reload

What `E2` has **not** yet proven:
1. that the underlying GCS model artifact is wrong
2. that the executor-backed RL path has the exact same `MODEL_MAPPING` bug
3. that `MistralForCausalLM` means the wrong family of weights was loaded

### 7. Recommended next actions for the next agent

Highest-value immediate next step:
1. patch the direct `E2` probe to set:
   - `canonical_model_name='meta-llama/Llama-3.1-8B-Instruct'`
2. rerun `E2`
3. check whether it now matches `E1` behavior through:
   - first weight transfer
   - first rollout write
   - first train step

After that:
1. if patched `E2` passes, move to `E3`
2. if patched `E2` still shows `MistralForCausalLM` but otherwise works, treat the architecture-name difference as a backend loader quirk, not a primary bug
3. if patched `E2` still fails in another way, that next failure becomes the real GCS-path divergence

## 2026-03-24T19:26Z — Killed stale E2 and patched the direct probe to preserve canonical model identity

Executed user-requested cleanup first:
- stopped `/ahmed/iris-rl-e2-direct-gcs-small-20260324-112640`
- Iris terminated:
  - root job
  - train child
  - rollout child

Applied the narrow direct-probe fix:
- file: [exp_iris_rl_regression_direct_gcs_small.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_small.py)
- added:
  - `CANONICAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"`
- changed rollout vLLM config to pass:
  - `model_name=MODEL_PATH`
  - `canonical_model_name=CANONICAL_MODEL_NAME`

Why this fix is correct:
- it keeps the experimental variable intact:
  - rollout still loads the base checkpoint from the regional GCS artifact via `runai_streamer`
- but it removes the known probe-construction bug:
  - hot reload will no longer use the raw `gs://...` path for `MODEL_MAPPINGS` / `MODEL_TRANSPOSE_KEYS`

Expected effect on the next rerun:
- if the previous `MODEL_MAPPING` failure was solely due to missing canonical identity, the rerun should get past:
  - `Received new weights from step -1`
  - `reload_model: calling sync_weights(...)`
- if another failure appears after that, that next failure becomes the real post-fix `E2` divergence

Fresh rerun submitted:
- root job: `/ahmed/iris-rl-e2b-direct-gcs-small-20260324-122630`
- purpose: retest `E2` after preserving canonical model identity while keeping GCS-backed checkpoint loading
- next verification boundary:
  - rollout should cross the old failure point after `Received new weights from step -1`

## 2026-03-24T19:54Z — E2b succeeded end to end; direct GCS small is now a pass

Checked the live Iris state for the fixed rerun:
- command:
  - `uv run iris --config=lib/iris/examples/marin.yaml job bug-report /ahmed/iris-rl-e2b-direct-gcs-small-20260324-122630`

Terminal job states:
- root job `/ahmed/iris-rl-e2b-direct-gcs-small-20260324-122630`:
  - `State: succeeded`
  - `Started: 2026-03-24T19:26:49Z`
  - `Finished: 2026-03-24T19:48:49Z`
  - `Duration: 22m 0s`
- descendant `/ahmed/iris-rl-e2b-direct-gcs-small-20260324-122630/rl-direct-gcs-small-20260324-192752-train`:
  - `succeeded`
- descendant `/ahmed/iris-rl-e2b-direct-gcs-small-20260324-122630/rl-direct-gcs-small-20260324-192752-rollout-0`:
  - `succeeded`

This is the key result:
- the fixed `E2b` probe passed with:
  - direct topology
  - GCS-backed base checkpoint loading via `runai_streamer`
  - the same small 5-step envelope used for the old baseline

### 1. What changed between failed `E2` and successful `E2b`

The only intentional code change between the two direct GCS probes was preserving canonical model identity in the rollout inference config.

In the fixed probe:
- file: [exp_iris_rl_regression_direct_gcs_small.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_small.py)
- added:
  - `CANONICAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"`
- rollout vLLM config now passes both:
  - `model_name=MODEL_PATH`
  - `canonical_model_name=CANONICAL_MODEL_NAME`

Interpretation:
- the earlier direct `E2` failure was caused by probe construction, not by a fundamental incompatibility between RL rollout and GCS-backed model artifacts
- hot reload needed the canonical HF-style model identity for:
  - `MODEL_MAPPINGS`
  - transpose-key lookups
  - architecture-specific reload behavior

### 2. What E2b proved at runtime

The successful rerun crossed the exact old failure boundary.

Important rollout sequence from the successful run:
- rollout started with the GCS-backed artifact
- rollout received first weights:
  - `Received new weights from step -1`
- rollout hot reload crossed the previous crash site:
  - `reload_model: starting prefix cache reset`
  - `reload_model: converting state dict`
  - `reload_model: calling sync_weights (...)`
  - `reload_model: sync_weights done ...`
  - `reload_model: complete ...`
  - `First weight transfer complete, inference can proceed`

After that, rollout generated actual batches:
- `Generated rollout with 10 groups from lesson math_full at step -1`
- `Generated rollout with 4 groups from lesson math_full at step -1`
- `Generated rollout with 4 groups from lesson math_full at step 0`
- `Generated rollout with 4 groups from lesson math_full at step 1`
- `Generated rollout with 4 groups from lesson math_full at step 2`
- `Generated rollout with 4 groups from lesson math_full at step 3`

Trainer also made real progress and completed the run:
- train advanced through the 5-step small envelope
- both train and rollout exited successfully
- coordinator/root job exited successfully

### 3. How to interpret the remaining weirdness in the logs

Even on the successful `E2b` run, the rollout path still reported:
- `Resolved architecture: MistralForCausalLM`

That means:
- this architecture-name discrepancy is not by itself a fatal bug
- it is most likely a TPU vLLM / `runai_streamer` loader-path quirk or aliasing behavior
- it should not currently be treated as evidence that we loaded Mixtral or the wrong weights

This conclusion is now stronger than before because:
- the GCS artifact metadata is still known-good Llama metadata
- the fixed direct GCS probe ran to completion with that same `MistralForCausalLM` log line present

There was also a late Arrow Flight transport error near shutdown in the successful run:
- connection refused while polling for another weight update
- but it happened after the run had already reached completion and Iris marked the job tree `succeeded`

Current interpretation of that late transport error:
- teardown/shutdown race noise
- not the primary failure mode for this branch

### 4. Updated experiment-matrix interpretation

Current state of the divergence matrix:

`E1. Direct + HF + small`
- status: pass
- meaning:
  - current HEAD still supports the old direct/HF/small baseline

`E2. Direct + GCS + small`
- original result: failed
- revised interpretation:
  - failure was a probe-specific wiring bug due to missing `canonical_model_name`

`E2b. Direct + GCS + small (fixed canonical identity)`
- status: pass
- meaning:
  - direct GCS-backed rollout + train works on current HEAD once canonical model identity is preserved
  - switching from HF ids to regional GCS artifacts is not, by itself, the fundamental blocker

This narrows the search space materially:
- the main remaining divergence is no longer "HF vs GCS" in the direct small path
- the next controlled comparison should be topology:
  - `E3 = executor + GCS + small`

### 5. Hard conclusions the next agent should rely on

1. The old known-good direct path still works on current HEAD (`E1` pass).
2. The direct GCS path also works on current HEAD once canonical model identity is preserved (`E2b` pass).
3. The earlier `MODEL_MAPPING` error does **not** prove a fundamental GCS artifact bug.
4. The presence of `Resolved architecture: MistralForCausalLM` in rollout logs does **not** currently correlate with failure; it also appears on the successful `E2b` run.
5. The next discriminating experiment should be `E3`:
   - executor topology
   - GCS-backed model artifact
   - small envelope
6. Only after `E3` should we return to the 500-step/OOM branch, because otherwise topology and envelope remain confounded.

### 6. Immediate next step

Run `E3` as the first topology-only change after two green direct baselines:
- keep:
  - GCS-backed model artifact
  - small envelope
- change only:
  - launch topology from direct coordinator to executor-wrapped step

Decision rule:
- if `E3` fails while `E2b` passes, the first clean divergence is executor topology / executor-managed runtime pathing
- if `E3` also passes, the remaining divergence is likely in the production envelope (`500` steps, checkpoint cadence, or resource pressure), not in GCS model loading or basic topology

## 2026-03-24T20:02Z — TODO RESOLVE: why does rollout/vLLM resolve `MistralForCausalLM` for a Llama 3.1 8B artifact?

This is still unresolved and should **not** be forgotten just because `E2b` succeeded.

### The contradiction

For the successful direct GCS probe, rollout logs still reported:
- `INFO ... Resolved architecture: MistralForCausalLM`

But the actual staged / source model metadata says the opposite:
- GCS artifact:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/config.json`
- confirmed contents:
  - `"architectures": ["LlamaForCausalLM"]`
  - `"model_type": "llama"`

Meanwhile the direct HF probe behaved differently:
- direct HF small (`E1`) rollout logs reported:
  - `Resolved architecture: LlamaForCausalLM`

So we have a real loader-path discrepancy:
1. HF model-id path -> `LlamaForCausalLM`
2. GCS / staged-local-metadata path -> `MistralForCausalLM`

### Why this is not closed

It is true that:
- `E2b` completed successfully
- Marin currently patches `MistralForCausalLM -> LlamaForCausalLM` in the TPU inference registry
- the presence of `MistralForCausalLM` did **not** correlate with immediate failure on `E2b`

But that does **not** mean this is benign or understood.

It still indicates one of the following:
1. vLLM/tpu_inference is normalizing the local/staged metadata path onto a Mistral backend architecture key for Llama-family checkpoints.
2. the `runai_streamer` / object-store path and the staged-local metadata path are not preserving architecture identity in the same way as the HF repo-id path.
3. some upstream vLLM or TPU inference heuristic is keying off local path layout or a transformed config object rather than the raw HF `config.json` fields we inspected.

Any of those could matter later for:
- executor-topology runs
- larger envelopes
- future checkpoint families
- correctness of hot-reload behavior
- silent model-family mismatches that happen to "work" today

### Current best evidence

What is known:
1. The artifact itself is correct Llama metadata.
2. The direct GCS probe now stages metadata locally before inflight vLLM startup:
   - [_prepare_vllm_inference_config](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rollout_worker.py:300)
   - rewrites remote `gs://...` to a local staged directory
   - rewrites `load_format` from `runai_streamer` to `dummy`
3. Even after that rewrite, rollout still logs `Resolved architecture: MistralForCausalLM`.
4. The trainer-side Levanter config remains Llama because it is derived from the same HF metadata.
5. Our own patch layer explicitly assumes this alias may occur:
   - [vllm.py:160](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:160)
   - [test_inference_ctx.py:346](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/tests/rl/test_inference_ctx.py:346)

### Required follow-up work

This is a **must-investigate-later** item, not background noise.

Concrete next checks:
1. Add explicit rollout startup logging of:
   - staged local `config.json`
   - `model_type`
   - `architectures`
   - `canonical_model_name`
   - final `inference_config.model_name`
   - final `load_format`
2. Inspect the upstream vLLM / TPU inference model-selection code path that emits `Resolved architecture: ...` and determine exactly what input causes `MistralForCausalLM` to be selected.
3. Compare the exact startup config object between:
   - `E1` direct HF small
   - `E2b` direct GCS small
4. Determine whether the alias is:
   - a harmless backend implementation choice, or
   - a semantically incorrect architecture remap that only happens to work for this run.
5. If the alias is harmless, codify that with a targeted test and an explanatory comment.
6. If the alias is not harmless, fix the startup path so Llama artifacts resolve as Llama consistently across HF and GCS-backed launch paths.

### Operational guidance for the next agent

Do **not** reopen the already-fixed `E2b` regression based solely on this log line.

Do treat this as a high-priority unresolved architecture-identity question that needs its own focused investigation once the main divergence matrix advances to `E3`.

## 2026-03-24T20:04Z — launching E3 (`executor + GCS + small`) as the next clean topology probe

User explicitly requested:
1. kill any currently running RL jobs
2. launch `E3`

Cluster sweep result before launch:
- no live `/ahmed/iris-rl-*` jobs were running
- recent RL jobs under `/ahmed/iris-rl-*` are all terminal (`succeeded`, `failed`, or `killed`)
- unrelated non-RL jobs were left untouched

Purpose of `E3`:
- change only launch topology relative to the now-green `E2b`
- keep:
  - regional GCS-backed model artifact
  - small 5-step envelope
  - `v5p-8` train/rollout placement
  - the same Llama 3.1 8B instruct artifact and canonical model name
- change:
  - direct root/coordinator launch -> executor-wrapped RL step

Planned `E3` configuration:
- model source:
  - `experiments.models.llama_3_1_8b_instruct`
- topology:
  - `executor_main(...)`
  - `make_rl_step(...)`
- trainer / rollout envelope:
  - `num_train_steps=5`
  - `train_batch_size=64`
  - `per_device_parallelism=16`
  - `max_input_tokens=512`
  - `max_output_tokens=512`
  - `n_prompts=4`
  - `n_generations_per_prompt=16`
  - `num_rollout_workers=1`
  - `weight_transfer_sync_interval_steps=1`
  - `max_weight_transfer_wait_time=300`
  - `inflight_weight_updates=False`
- TPU placement:
  - `train_tpu_type="v5p-8"`
  - `inference_tpu_type="v5p-8"`
- launcher region:
  - explicit root submit in `us-central1`
  - executor helper should keep the inner step and RL workers in the same concrete RL region

Success criteria:
1. root executor job starts
2. model artifact prereq resolves / reuses cache
3. nested RL coordinator starts
4. train child starts
5. rollout child starts
6. first weight transfer completes
7. first rollout writes
8. trainer advances at least one step

Failure interpretation rule:
- if `E3` fails while `E2b` passed, the first clean divergence is executor topology / executor-managed runtime pathing, not GCS artifact loading itself

Submission details:
- root job id:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512`
- root submit command:
  - `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-e3-exec-gcs-small-20260324-130512 --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_executor_gcs_small.py`
- monitoring state file:
  - `scratch/20260324-1305_e3_state.json`

Immediate post-submit state:
- root job exists and is currently `JOB_STATE_PENDING`
- current pending reason:
  - `Scheduler: Insufficient CPU (need 0.5 cores, available 0 cores) - 1 worker(s)`
  - constraints include `region`, `preemptible`, `reservation-job`
  - autoscaler detail:
    - `no_capacity: cpu_vm_e2_highmem_2_ondemand-us-central1-a=at_max_slices`

Interpretation:
- this is the same class of launcher-capacity wait seen earlier on `E1`
- no code-level signal yet
- nested executor/coordinator/train/rollout jobs do not exist yet because the root launcher has not started

Next check:
1. wait for the root CPU launcher to clear placement
2. once it starts, verify whether the executor step reuses the cached model artifact and materializes the nested RL coordinator

## 2026-03-24T20:22Z — E3 cleared the root barrier and reached the nested coordinator boundary

First real monitor cadence after launch changed the picture materially.

Current live job tree:
- root `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512`:
  - `JOB_STATE_RUNNING`
- executor child:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512/rl_testing-exec-gcs-small-20260324-201951_dbbb33f0-b01858e8`
  - `JOB_STATE_RUNNING`
  - `preemption_count=1`
- nested RL coordinator:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512/rl_testing-exec-gcs-small-20260324-201951_dbbb33f0-b01858e8/rl-exec-gcs-small-20260324-201951`
  - `JOB_STATE_PENDING`

Current pending reason on the nested coordinator:
- `Scheduler: Insufficient CPU (need 1 cores, available 0.5 cores)`
- constraints still include `region`, `preemptible`, `reservation-job`
- autoscaler detail remains:
  - `no_capacity: cpu_vm_e2_highmem_2_ondemand-us-central1-a=at_max_slices`

Interpretation:
- `E3` has already passed the first executor-topology boundary:
  - root launcher started
  - executor step materialized
  - nested RL coordinator job was submitted
- so the current blocker is no longer the outer root launcher
- the next startup barrier is the nested RL coordinator CPU placement

What is not known yet:
- whether the model artifact prereq has already been resolved/reused inside the executor child
- whether the nested coordinator will start cleanly once CPU capacity clears
- whether the executor-wrapped small path will diverge after this point relative to the green `E2b` direct path

Next monitor target:
1. nested RL coordinator transitions from `PENDING` -> `RUNNING`
2. then verify appearance of train / rollout TPU child jobs

## 2026-03-24T20:43Z — E3 progressed again: nested coordinator is now running; train and rollout exist but are TPU-capacity blocked

Fresh Iris queries show that the earlier dashboard confusion was partly timing.

Current live tree:
- root:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512`
  - `running`
- executor child:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512/rl_testing-exec-gcs-small-20260324-201951_dbbb33f0-b01858e8`
  - `running`
- nested RL coordinator:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512/rl_testing-exec-gcs-small-20260324-201951_dbbb33f0-b01858e8/rl-exec-gcs-small-20260324-201951`
  - `running`
- train child:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512/rl_testing-exec-gcs-small-20260324-201951_dbbb33f0-b01858e8/rl-exec-gcs-small-20260324-201951/rl-exec-gcs-small-20260324-201951-train`
  - `pending`
- rollout child:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512/rl_testing-exec-gcs-small-20260324-201951_dbbb33f0-b01858e8/rl-exec-gcs-small-20260324-201951/rl-exec-gcs-small-20260324-201951-rollout-0`
  - `pending`

Current train / rollout pending reason:
- `Scheduler: Insufficient TPUs (need 4, available 0)`
- `Autoscaler: no_capacity: tpu_v5p_8-us-central1-a=backoff`

Important interpretation:
- the earlier screenshot that showed no train / rollout children was not evidence that executor topology removed them
- at that moment, the tree had simply not progressed far enough yet
- now the executor-wrapped RL path has reached the exact point where the nested coordinator has submitted both TPU children

One additional live signal from the executor-step bug report:
- executor child had one earlier worker failure:
  - `Worker ... failed: Request timed out`
- Iris retried it and the current attempt is healthy
- this is the same class of transient launcher retry already seen on earlier RL runs and is not, by itself, the current blocker

Next monitor target:
1. TPU capacity clears for train / rollout
2. verify first weight transfer and first rollout generation on the executor-wrapped small path

## 2026-03-24T21:02Z — E3 postmortem: the RL run itself succeeded; the executor wrapper failed afterward while trying to serialize an `IrisJobHandle`

Final terminal state:
- root `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512`:
  - `failed`
  - root error:
    - `RuntimeError: 1 step(s) failed`
- executor child:
  - `/ahmed/iris-rl-e3-exec-gcs-small-20260324-130512/rl_testing-exec-gcs-small-20260324-201951_dbbb33f0-b01858e8`
  - `failed`
  - real executor-child error:
    - `TypeError: Object of type IrisJobHandle is not JSON serializable`

Critical descendant facts:
- nested RL coordinator:
  - `succeeded`
- train child:
  - `succeeded`
- rollout child:
  - `succeeded`

This means the executor-wrapped small probe reached full end-to-end RL success:
- model artifact cache reuse worked
- executor topology worked
- nested coordinator ran successfully
- train finished
- rollout finished

The only thing that failed was the executor step's post-run artifact-save behavior.

### Exact failure chain

The executor remote wrapper expects the remote step function to return a JSON-serializable artifact.

Relevant code path:
- [_run_rl_experiment_step](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py:320)
  - currently returns:
    - `RLJob(job_config).run(config.name)`
- [RLJob.run](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_job.py:189)
  - returns the Iris/Fray job handle after waiting:
    - `return handle`
- [_run_remote_step](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/execution/step_runner.py:355)
  - wraps the remote fn and always does:
    - `result = raw_fn(out_path)`
    - `Artifact.save(result, out_path)`
- [Artifact.save](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/execution/artifact.py:43)
  - falls back to `json.dumps(artifact)` for non-dataclass, non-pydantic objects

So the executor step returned an `IrisJobHandle`, and then the wrapper attempted:
- `json.dumps(IrisJobHandle(...))`
- which raised:
  - `TypeError: Object of type IrisJobHandle is not JSON serializable`

### Why this matters for the experiment matrix

This is a strong result, even though the root job is marked failed.

`E3` should be interpreted as:
- RL runtime / topology result:
  - **pass**
- executor artifact contract result:
  - **fail**

In other words:
- executor topology did **not** introduce a train/rollout runtime regression on the small GCS-backed path
- the first executor-only divergence is narrower:
  - the executor step contract currently assumes the step returns a serializable artifact
  - the RL step returns a live job handle object instead

### Scope of the bug

This failure is not:
- a GCS model-loading failure
- a rollout startup failure
- a train/rollout orchestration failure
- a TPU runtime failure

This failure **is**:
- an executor-step return-value bug at the remote artifact-save boundary

### Immediate next fix

Smallest correct fix:
1. change [_run_rl_experiment_step](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py:320) so it does **not** return the live `JobHandle`
2. instead return a serializable artifact, for example:
   - `PathMetadata(config.output_path)`, or
   - `{"status": "completed", "run_id": config.name}`, or
   - `None` if the executor framework treats that sanely

The key requirement is:
- executor-managed RL steps must not return `IrisJobHandle` objects across the artifact-save boundary

### Updated matrix interpretation after E3

`E1` direct + HF + small:
- pass

`E2b` direct + GCS + small:
- pass

`E3` executor + GCS + small:
- RL runtime path passes
- executor wrapper fails after success because of handle serialization

So the first clean executor-only bug is:
- **step return artifact contract**, not RL orchestration/runtime correctness

### Plain-English restatement

What happened in `E3` was **not**:
- "train or rollout crashed"
- "the coordinator failed to run RL"
- "executor topology broke RL itself"

What happened was:
1. the executor child launched the RL coordinator job
2. the RL coordinator launched train + rollout
3. train + rollout both finished successfully
4. the RL coordinator finished successfully
5. control returned back to the executor child
6. the executor child then tried to save the *return value* of the RL step as a JSON artifact
7. that return value was an `IrisJobHandle`
8. JSON serialization of `IrisJobHandle` failed
9. therefore the executor child was marked failed
10. therefore the root executor job was marked failed

So:
- the **RL work** succeeded
- the **executor bookkeeping after RL completed** failed

### Concrete timeline

These timestamps make the sequencing explicit:

1. Train finished successfully:
   - `2026-03-24T20:56:04.047Z`
2. Rollout finished successfully:
   - `2026-03-24T20:56:04.368Z`
3. Nested RL coordinator finished successfully:
   - `2026-03-24T20:56:14.810Z`
4. Executor child hit the serialization traceback:
   - logs show traceback at about `2026-03-24T20:56:17Z`
5. Executor child reached terminal failed state:
   - `2026-03-24T20:56:19.813Z`
6. Root executor job then failed because one step failed:
   - `2026-03-24T20:56:34.807Z`

This ordering matters:
- if train/rollout had been the real failure, they would have finished **after** the executor-child failure or would themselves be marked failed
- instead, they both finished **before** the executor-child failure and are marked `succeeded`

### Why W&B looked healthy while Iris root looked failed

The W&B runs correspond to the nested RL subtree:
- `exec-gcs-small-20260324-201951-train`
- `exec-gcs-small-20260324-201951-rollout-0`

Those runs are healthy because the RL subtree really was healthy.

The Iris root job looked failed because Iris was reporting the outer executor wrapper job, which failed after RL completion when it tried to serialize the returned handle.

So the apparent contradiction:
- "W&B looks successful"
- "Iris root says failed"

is explained by two different layers finishing in two different states:
- inner RL layer: success
- outer executor wrapper layer: failure

### Most precise single-sentence summary

`E3` proved that executor-wrapped RL works on the small GCS-backed path, but the executor step is currently unusable because it returns a non-serializable `IrisJobHandle` and then crashes while saving that return value as an artifact.

## 2026-03-24T21:19Z — applying the E3 executor serialization fix and rerunning

User asked to make the fix and run the job again.

Fixed code path:
- [rl_experiment_utils.py:320](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py:320)
- changed `_run_rl_experiment_step(...)` so it no longer returns the live `JobHandle`
- new behavior:
  - wait for `RLJob(job_config).run(config.name)` to finish
  - return `PathMetadata(path=config.output_path)` instead

Why this is the correct fix:
- the executor remote wrapper requires a serializable artifact result
- `PathMetadata` is already a supported artifact type in the executor framework
- the RL step no longer crosses the artifact boundary with a live `IrisJobHandle`

Regression coverage added:
- [test_rl_experiment_utils.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/tests/rl/test_rl_experiment_utils.py)
- new test verifies `_run_rl_experiment_step(...)` returns `PathMetadata`, not the raw handle

Validation before relaunch:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py`
  - `9 passed`
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/rl_experiment_utils.py tests/rl/test_rl_experiment_utils.py`
  - `OK`

Rerun intent:
- rerun the same `E3` topology probe after the serialization fix
- training-step count remains the same:
  - `num_train_steps=5`
- this rerun should answer the narrow executor question:
  - can the executor-wrapped small GCS-backed RL probe now finish fully green at the root level, not just in the nested RL subtree?

Fresh rerun submitted:
- root job:
  - `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934`
- monitoring state file:
  - `scratch/20260324-1419_e3b_state.json`

Initial post-submit state:
- root is currently `JOB_STATE_PENDING`
- current pending reason:
  - `Scheduler: No worker matches constraints and has sufficient resources`
  - autoscaler detail:
    - `Waiting for worker scale-up in scale group 'cpu_vm_e2_highmem_2_ondemand-us-central1-a'`

Interpretation:
- the fix is in and the rerun is live
- no result yet on the serialization boundary because the root launcher has not started user code

## 2026-03-24T21:42Z — `E3b` passed end to end after the executor serialization fix

Final result for the rerun:
- root `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934`: `JOB_STATE_SUCCEEDED`
- executor child `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3`: `JOB_STATE_SUCCEEDED`
- nested RL coordinator `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3/rl-exec-gcs-small-20260324-212413`: `JOB_STATE_SUCCEEDED`
- train child `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3/rl-exec-gcs-small-20260324-212413/rl-exec-gcs-small-20260324-212413-train`: `JOB_STATE_SUCCEEDED`
- rollout child `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3/rl-exec-gcs-small-20260324-212413/rl-exec-gcs-small-20260324-212413-rollout-0`: `JOB_STATE_SUCCEEDED`

Finish timestamps from `iris job list --json --prefix`:
- rollout finished: `2026-03-24T21:41:21.727Z`
- train finished: `2026-03-24T21:41:22.294Z`
- coordinator finished: `2026-03-24T21:41:32.772Z`
- executor child finished: `2026-03-24T21:41:43.525Z`
- root finished: `2026-03-24T21:42:02.770Z`

Observed RL progress:
- rollout loaded the cached regional GCS model artifact successfully
- rollout completed the first weight transfer and entered inference
- trainer completed all 5 configured training steps
- rollout kept receiving updated weights and generating rollouts for steps `-1` through `3`
- coordinator logged `RL coordinator finished for run exec-gcs-small-20260324-212413`
- executor child logged `Step rl_testing/exec-gcs-small-20260324-212413_db03625d succeeded`

Important shutdown detail:
- rollout logged a late `pyarrow._flight.FlightUnavailableError` during teardown
- this happened after the run state had already been marked completed:
  - `marin.rl.run_state RL run marked as completed`
  - `marin.rl.rollout_worker Run state is 'completed', stopping rollout worker`
- despite the traceback text, the rollout job itself finished `JOB_STATE_SUCCEEDED`
- interpretation: this is teardown noise after normal completion, not a failure boundary for `E3b`

What `E3b` proves:
- the `E3` executor failure was correctly diagnosed as the step return-artifact contract bug
- returning `PathMetadata(path=config.output_path)` instead of the live `IrisJobHandle` fixes the executor wrapper
- `executor + GCS + small` is now green end to end, not just at the nested RL subtree

Updated matrix status after `E3b`:
- `E1` (`direct + HF + small`): pass
- `E2b` (`direct + GCS + small`, canonical-name fixed): pass
- `E3b` (`executor + GCS + small`, serialization fixed): pass

Implication for the larger investigation:
- small-run regressions attributable to HF-vs-GCS model source and direct-vs-executor topology have now been eliminated
- the next remaining divergence to hunt is the production envelope itself (`E4` / `E5`)

UI note observed during `E3b` babysitting:
- the Iris CLI could see the full nested tree immediately via `job list --json --prefix`
- the dashboard pages did not reliably surface the train/rollout descendants while the run was live
- this appears to be a dashboard tree-display limitation/staleness issue, not a job-submission issue in `E3b`

## 2026-03-24T21:59Z — launching `E3c` as a 2-step executor/GCS rerun

User asked to rerun the green `E3b` topology probe, but with only 2 training steps.

To avoid mutating the meaning of the already-established `E3` baseline, I changed
[exp_iris_rl_regression_executor_gcs_small.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_executor_gcs_small.py)
so the probe now accepts explicit CLI arguments:
- `--experiment-name-suffix`
- `--num-train-steps`

Validation for the CLI override change:
- `./infra/pre-commit.py --fix experiments/exp_iris_rl_regression_executor_gcs_small.py` -> `OK`
- `uv run python experiments/exp_iris_rl_regression_executor_gcs_small.py --help` shows both new flags

Rerun intent:
- hold topology fixed at `executor + GCS + small`
- reduce training steps from `5` to `2`
- keep the run visually distinct in Iris/W&B via a new suffix

Submit command for the new rerun:
- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-e3c-exec-gcs-small-2step-<timestamp> --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_executor_gcs_small.py --experiment-name-suffix exec-gcs-small-2step --num-train-steps 2`

Expectation:
- the nested RL subtree should still succeed
- the previously-fixed executor artifact boundary should remain green
- the resulting run names should clearly identify this as the 2-step rerun, not the canonical 5-step `E3b`

Actual launch result:
- root job submitted:
  - `/ahmed/iris-rl-e3c-exec-gcs-small-2step-20260324-150008`
- monitoring state file:
  - `scratch/20260324-1500_e3c_state.json`

Initial scheduler state immediately after submit:
- root: `JOB_STATE_PENDING`
- pending reason:
  - `Scheduler: No worker matches constraints and has sufficient resources (need cpu=0.5 cores, memory=1073741824, constraints=['region', 'preemptible', 'reservation-job'])`
  - `Autoscaler: (scaling up) Waiting for worker scale-up in scale group 'cpu_vm_e2_highmem_2_ondemand-us-central1-a' (1 slice(s) requested)`

## 2026-03-24T22:41Z — why `E3c` is slower than `E3b`, and recommended fix direction

`E3c` exposed a scheduling detail that `E3b` happened to dodge:
- the executor child in `E3b` landed on a TPU worker host CPU
- the executor child in `E3c` landed on the dedicated CPU worker group
- once that happened, the nested RL coordinator could no longer fit on the only available CPU slice in `us-central1`

Concrete evidence from live `E3c`:
- root job worker:
  - `marin-cpu-vm-e2-highmem-2-ondemand-us-ce-20260324-2200-105bf714-worker-0`
- executor child worker:
  - `marin-cpu-vm-e2-highmem-2-ondemand-us-ce-20260324-2200-105bf714-worker-0`
- nested coordinator state:
  - `JOB_STATE_PENDING`
- pending reason:
  - `Insufficient CPU (need 1 cores, available 0.5 cores)`
  - `no_capacity: cpu_vm_e2_highmem_2_ondemand-us-central1-a=at_max_slices`

Autoscaler status for the relevant CPU group:
- group:
  - `cpu_vm_e2_highmem_2_ondemand-us-central1-a`
- machine type:
  - `e2-highmem-2`
- `max_slices: 1`
- current ready slices: `1`
- availability status:
  - `at_max_slices`
- current worker running task count:
  - `2`

This means the current bottleneck is not “global GCP CPU quota” in the abstract.
It is the specific Iris autoscaler group for non-preemptible CPU VMs in
`us-central1-a`, which is capped at one slice.

### Why `E3b` did not hit the same blockage

`E3b` executor child placement:
- executor child worker:
  - `marin-tpu_v5p_8-us-central1-a-20260324-1955-df773a09-worker-0`

So in `E3b`:
- root launcher consumed CPU-worker capacity
- executor child consumed TPU-host CPU instead
- nested coordinator could still fit on the one CPU slice

In `E3c`:
- root launcher is on the CPU slice
- executor child is also on the same CPU slice
- only `0.5 CPU` remains available
- nested coordinator needs `1.0 CPU`
- therefore the coordinator cannot be scheduled

Important interpretation:
- “CPU-only jobs can land on TPU workers” is true
- but it is opportunistic placement, not a guarantee
- we should not design the RL topology to *require* that lucky placement

### Pressure points in current code

Executor child resource request:
- [rl_experiment_utils.py:37](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py:37)
- `RL_EXECUTOR_STEP_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="4g", disk="64g")`

Nested RL coordinator submission:
- [orchestration.py:79](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/orchestration.py:79)
- `ResourceConfig.with_cpu(preemptible=False, disk=RL_COORDINATOR_CPU_DISK)`

Observed coordinator runtime resources in Iris:
- `1 cpu, 4 GiB`

So when root + executor + coordinator overlap on the single `e2-highmem-2`
worker, the CPU budget is approximately:
- root: `0.5 CPU`
- executor child: `1.0 CPU`
- coordinator: `1.0 CPU`
- total: `2.5 CPU`

But the worker only has:
- `2 CPU`

That is why the coordinator starves.

### Recommended fix order

#### Recommended first fix: reduce CPU requirements to fit on one CPU slice

Best immediate change:
- lower executor child from `1.0 CPU` to `0.5 CPU`
- lower nested coordinator from `1.0 CPU` to `0.5 CPU`

Why this is the preferred first move:
- local code change only
- deterministic
- does not require cluster/autoscaler config changes
- removes dependence on TPU-host placement luck
- still leaves enough RAM headroom on the same CPU VM

If we do that, the overlapping CPU budget becomes approximately:
- root: `0.5 CPU`
- executor child: `0.5 CPU`
- coordinator: `0.5 CPU`
- total: `1.5 CPU`

That fits comfortably inside a single `e2-highmem-2` slice.

Current recommendation:
- change CPU only first
- keep RAM unchanged for now

Why keep RAM unchanged initially:
- the current bottleneck is CPU, not memory
- current CPU-layer memory total is already acceptable on the `16 GiB` VM
- shrinking RAM early would add risk without addressing the observed blocker

#### Second-choice structural fix: collapse the nested coordinator layer

Alternative medium-sized cleanup:
- for executor-backed RL runs, do not submit a second CPU coordinator job
- instead, let the executor child act as the RL coordinator directly

Potential upside:
- removes one CPU job layer entirely
- simpler job tree:
  - root
  - executor/coordinator
  - train
  - rollout
- lower launch latency
- less scheduler pressure

Why this is not the first recommendation:
- larger semantic/topology change
- changes job hierarchy and failure-surfacing behavior
- unnecessary if the smaller CPU reduction fully solves the issue

#### Infra fallback: raise the CPU group max-slices

Possible infrastructure-side fix:
- increase `max_slices` for
  - `cpu_vm_e2_highmem_2_ondemand-us-central1-a`

Why this is lower priority:
- broader blast radius
- cluster-cost / autoscaler policy change
- the tiny RL probe should not require extra CPU slice capacity just to avoid
  deadlocking the launcher/executor/coordinator layers

### Explicit recommendation

Recommended plan to try next:
1. lower executor child CPU request from `1.0` to `0.5`
2. lower nested coordinator CPU request from `1.0` to `0.5`
3. rerun the 2-step executor/GCS probe
4. only if that is still flaky, consider collapsing the nested coordinator layer
5. keep autoscaler-group changes as a fallback, not the first fix

### Key design principle

Do not rely on “CPU jobs landing on TPU workers” as a design assumption.
Treat that as opportunistic placement luck, not a correctness path.

The RL stack should be schedulable even when:
- root launcher
- executor child
- nested coordinator

all land on the dedicated CPU worker group.

## 2026-03-24T23:03Z — reducing root/executor/coordinator CPU and relaunching the 2-step probe

User asked to rerun the 2-step executor/GCS probe with lower CPU on all three launcher/control-plane layers:
- root launcher job
- executor child
- nested RL coordinator

Applied code changes:
- [rl_experiment_utils.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py)
  - `RL_EXECUTOR_STEP_RESOURCES`
  - changed from `cpu=1` to `cpu=0.5`
- [orchestration.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/orchestration.py)
  - added `RL_COORDINATOR_RESOURCES`
  - nested coordinator now requests `cpu=0.5` instead of the default `1.0`

Planned root relaunch change:
- use explicit `iris job run --cpu 0.25`

New expected overlapping CPU budget if root + executor + coordinator all land on the same dedicated CPU worker:
- root: `0.25 CPU`
- executor child: `0.5 CPU`
- nested coordinator: `0.5 CPU`
- total: `1.25 CPU`

That should fit on the single `e2-highmem-2` CPU slice where the previous run starved at `2.5 CPU` requested over `2 CPU` capacity.

Validation before relaunch:
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/rl_experiment_utils.py lib/marin/src/marin/rl/orchestration.py` -> `OK`
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py tests/rl/test_orchestration.py` -> `10 passed`

Relaunch intent:
- keep topology = `executor + GCS + small`
- keep `num_train_steps = 2`
- use a new suffix so this low-CPU rerun is distinct from the earlier `E3c`

Follow-up change:
- user suggested also reducing control-plane disk requests
- rationale: executor child and nested coordinator are control-plane layers and do not need the previous oversized local disk reservations

Applied disk reductions:
- [rl_experiment_utils.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py)
  - executor child disk reduced from `64g` to `30g`
- [orchestration.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/orchestration.py)
  - nested coordinator disk reduced from `100g` to `30g`

Validation after the disk change:
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/rl_experiment_utils.py lib/marin/src/marin/rl/orchestration.py` -> `OK`
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py tests/rl/test_orchestration.py` -> `10 passed`

Important note about relaunch:
- the earlier interrupted low-CPU submit actually succeeded in creating a new root job before the shell was interrupted
- therefore no duplicate relaunch was needed

Live low-CPU/low-disk rerun:
- root job:
  - `/ahmed/iris-rl-e3d-exec-gcs-small-2step-lowcpu-20260324-160515`
- current root resources:
  - `cpu=0.25`
  - `memory=1 GiB`
  - `disk=5 GiB`
- this run is the active follow-on to the killed `E3c`

## 2026-03-24T23:17Z — E3d low-CPU rerun cleared the CPU bottleneck but failed again on long Iris workdir paths

Live rerun under watch:
- root job:
  - `/ahmed/iris-rl-e3d-exec-gcs-small-2step-lowcpu-20260324-160515`
- intent:
  - same `E3` topology (`executor + GCS + small`)
  - but with lower control-plane CPU to remove the single-CPU-slice starvation seen in `E3c`

What changed relative to `E3c`:
- root launcher requested `0.25 CPU`
- executor child requested `0.5 CPU`
- nested coordinator requested `0.5 CPU`

What the live tree showed before failure:
- root: `RUNNING`
- executor child: `RUNNING`
- nested coordinator: `RUNNING`
- train child: `PENDING`
- rollout child: `PENDING`

This is the key positive result from `E3d`:
- the low-CPU change did exactly what it was supposed to do at the scheduling boundary
- unlike `E3c`, the nested coordinator was no longer stuck behind the dedicated CPU slice
- train and rollout were both actually created

Observed TPU-capacity progression:
- initial state:
  - `Autoscaler: no_capacity: tpu_v5p_8-us-central1-a=backoff`
- later state:
  - `Autoscaler: Waiting for workers in scale group 'tpu_v5p_8-us-central1-a' to become ready (selected: demand-routed)`
- so Iris did eventually route TPU demand and begin trying to place the RL workers

Terminal failure:
- both train and rollout failed before user code during Iris worker setup with:
  - `OSError: [Errno 36] File name too long`
- failing paths were under:
  - `/dev/shm/iris/workdirs/...`
- concrete failing examples:
  - train:
    - `/dev/shm/iris/workdirs/job__ahmed__iris-rl-e3d-exec-gcs-small-2step-lowcpu-20260324-160515__rl_testing-exec-gcs-small-2step-lowcpu-20260324-230726_2acbc4be-6821a1a9__rl-exec-gcs-small-2step-lowcpu-20260324-230726__rl-exec-gcs-small-2step-lowcpu-20260324-230726-train__0_attempt_3`
  - rollout:
    - `/dev/shm/iris/workdirs/job__ahmed__iris-rl-e3d-exec-gcs-small-2step-lowcpu-20260324-160515__rl_testing-exec-gcs-small-2step-lowcpu-20260324-230726_2acbc4be-6821a1a9__rl-exec-gcs-small-2step-lowcpu-20260324-230726__rl-exec-gcs-small-2step-lowcpu-20260324-230726-rollout-0__0_attempt_3`

Failure chain:
1. train child failed in `task_attempt._setup()` before task startup
2. rollout child failed the same way
3. nested coordinator failed because the train child failed
4. executor child failed because the nested coordinator failed
5. root failed with `RuntimeError: 1 step(s) failed`

Important interpretation:
- `E3d` does **not** indicate a regression in the new low-CPU resource settings
- it indicates that once the CPU-starvation problem was removed, the run once again hit the older long-job-name workdir bug
- this is the same general failure family observed earlier in the OOM-isolation thread: child jobs fail before user code because the nested Iris job path becomes too long for `/dev/shm/iris/workdirs/...`

What `E3d` proved:
- lowering root/executor/coordinator CPU was worthwhile and effective
- the control-plane layers are now schedulable on the constrained CPU side
- the next blocking issue is still naming/path length, not CPU starvation

Important unresolved observation from the live submitted job:
- the active `E3d` job list still reported the executor child disk as `64 GiB` and the nested coordinator disk as `100 GiB`
- that does not match the current source constants after the disk-reduction patch
- likely explanation: the interrupted submit that produced `E3d` may have captured an older workspace snapshot before the disk changes landed
- do not infer from `E3d` that the disk reduction code path is definitely wired correctly; the live job metadata did not reflect it

Recommended next fix direction:
1. shorten the `E3` naming envelope so the full nested Iris job ids are much shorter
2. keep the low-CPU resource changes, since they solved a real scheduler problem
3. rerun the same 2-step executor/GCS probe under the shortened naming scheme before moving back to larger envelopes

The high-level picture after `E3d`:
- CPU-starvation issue: improved / likely fixed for this topology
- executor artifact-serialization issue: fixed earlier in `E3b`
- current blocker for executor-backed nested RL runs: still overlong nested Iris job/workdir paths

## 2026-03-24T23:33Z — immediate workaround for the Iris workdir path bug: relaunch E3 with very short names

Workaround strategy:
- do not wait for an Iris code fix
- keep the low-CPU control-plane settings from `E3d`
- relaunch the exact same `executor + GCS + 2-step` probe, but drastically shorten:
  - the root job name
  - the internal experiment-name suffix

Reasoning:
- `E3d` failed because Iris flattened the full nested task id into one workdir basename and crossed the single-component filesystem limit
- measured flattened train token lengths:
  - `E3b`: `204`
  - `E3c`: `228`
  - `E3d`: `256`
- a short-name relaunch should bring the basename back down into the ~130-150 range without changing RL semantics

Chosen workaround launch shape:
- root job name: short `irl-...` form instead of the long `iris-rl-e3d-exec-gcs-small-2step-lowcpu-...`
- internal experiment suffix: short `e3s`
- train steps: still `2`
- root CPU: still `0.25`
- executor child CPU: still `0.5`
- nested coordinator CPU: still `0.5`

This workaround is intentionally minimal:
- no code-path change to RL runtime
- no topology change
- only the naming envelope is shortened so the nested workdir basename stays under the filesystem component limit

## 2026-03-24T23:50Z — short-name `E3` workaround succeeded end-to-end; path-length bug confirmed as the blocker

Successful workaround run:
- root job:
  - `/ahmed/irl-e3s-2s-0324-163231`
- internal experiment suffix:
  - `e3s`
- train steps:
  - `2`

Why this run matters:
- `E3d` had already shown that the low-CPU control-plane settings removed the earlier CPU-slice starvation problem.
- The only thing changed for this rerun was the naming envelope:
  - much shorter root job name
  - much shorter internal experiment suffix
- This was an intentional workaround for the Iris workdir-path bug from issue `#4103`.

Observed outcome:
- the run crossed the exact old failure boundary that had killed `E3d`
- train and rollout both started real user code
- no `OSError: [Errno 36] File name too long` occurred during Iris worker setup
- the full subtree finished `SUCCEEDED`

Terminal Iris job tree:
- `/ahmed/irl-e3s-2s-0324-163231` -> `JOB_STATE_SUCCEEDED`
- `/ahmed/irl-e3s-2s-0324-163231/rl_testing-e3s-20260324-233847_3e5de1e8-948f62de` -> `JOB_STATE_SUCCEEDED`
- `/ahmed/irl-e3s-2s-0324-163231/rl_testing-e3s-20260324-233847_3e5de1e8-948f62de/rl-e3s-20260324-233847` -> `JOB_STATE_SUCCEEDED`
- `/ahmed/irl-e3s-2s-0324-163231/rl_testing-e3s-20260324-233847_3e5de1e8-948f62de/rl-e3s-20260324-233847/rl-e3s-20260324-233847-train` -> `JOB_STATE_SUCCEEDED`
- `/ahmed/irl-e3s-2s-0324-163231/rl_testing-e3s-20260324-233847_3e5de1e8-948f62de/rl-e3s-20260324-233847/rl-e3s-20260324-233847-rollout-0` -> `JOB_STATE_SUCCEEDED`

Exact resource settings used on the successful run:
- root:
  - `cpu=0.25`
  - `memory=1 GiB`
  - `disk=5 GiB`
- executor child:
  - `cpu=0.5`
  - `memory=4 GiB`
  - `disk=30 GiB`
- nested coordinator:
  - `cpu=0.5`
  - `memory=4 GiB`
  - `disk=30 GiB`
- train:
  - `v5p-8`
  - `disk=50 GiB`
- rollout:
  - `v5p-8`
  - `disk=50 GiB`

Timing from `iris job list --json --prefix /ahmed/irl-e3s-2s-0324-163231`:
- rollout started at `1774395641131`, finished at `1774396016855`
- train started at `1774395639212`, finished at `1774396016632`
- nested coordinator started at `1774395569236`, finished at `1774396037183`
- executor child started at `1774395532843`, finished at `1774396047242`
- root started at `1774395309448`, finished at `1774396067398`

Most important interpretation:
- the path-length bug was the real blocker in `E3d`
- shortening the naming envelope alone was sufficient to restore a fully successful executor-backed RL run
- the low-CPU resource reductions remain compatible with success
- no new RL-runtime regression was exposed by this rerun

This closes the immediate `E3` questions:
- `E3b` proved the executor return-value serialization bug was fixed
- `E3d` proved CPU starvation could be removed, but then exposed the Iris workdir path bug
- this short-name rerun proves that the remaining failure in `E3d` was naming/path length, not RL logic

Dashboard observation:
- the Iris dashboard now shows the nested subtree correctly under `All descendants`
- on the successful short-name run it displayed:
  - executor child
  - nested coordinator
  - train child
  - rollout child
- so the dashboard visibility problem and the run-stability problem are now both cleared for this `E3` probe family

Current matrix status after the short-name rerun:
- `E1` (`direct + HF + small`) -> pass
- `E2b` (`direct + GCS + small`, fixed canonical model name) -> pass
- `E3b` (`executor + GCS + small`) -> pass
- `E3d` (`executor + GCS + 2-step + low CPU + long names`) -> fail only because of Iris workdir path length
- short-name `E3` workaround (`executor + GCS + 2-step + low CPU + short names`) -> pass

Practical next step:
- carry the short-name discipline forward into the larger-envelope executor probes so we do not trip over the same Iris path bug while debugging the actual 500-step RL target.

## 2026-03-24T16:54PT — moving to `E4`: direct + GCS + prod

Current matrix state before `E4`:
- `E1` (`direct + HF + small`) -> pass
- `E2b` (`direct + GCS + small`) -> pass
- short-name `E3` (`executor + GCS + small`) -> pass

Reason for moving now:
- small-run topology and model-source questions are sufficiently resolved
- the remaining meaningful unknown is the heavy production-like envelope itself
- so the next clean branch is `E4`: keep direct topology, keep GCS-backed model bootstrap, restore the 500-step / 2048-token production-like envelope

`E4` intent:
- isolate whether the heavy envelope alone is enough to reproduce the active train-side instability / OOM family
- do this without executor-wrapper behavior as a confounder

Chosen `E4` shape:
- topology:
  - direct coordinator in the outer job (`_run_rl_coordinator(...)`)
- model source:
  - regional GCS artifact for Llama 3.1 8B Instruct
- envelope:
  - `num_train_steps=500`
  - `train_batch_size=1024`
  - `max_seq_len=2048`
  - `max_output_tokens=1024`
  - `n_prompts=16`
  - `n_generations_per_prompt=16`
  - checkpoint save interval `600s`
- naming:
  - short root job name
  - short internal run id / suffix
  - this is deliberate so we do not reintroduce the already-understood Iris workdir path bug while testing the envelope itself

Implementation note:
- do not mutate the small direct/GCS probe into a conditional multi-mode script
- instead add a dedicated `E4` script so the regression matrix remains explicit and the run config is easy to audit later

Launch details for live `E4` run:
- root job:
  - `/ahmed/irl-e4p-0324-1656`
- launch command shape:
  - direct outer job running `experiments/exp_iris_rl_regression_direct_gcs_prod.py --experiment-name-suffix e4p --num-train-steps 500`
- root resources:
  - `cpu=0.5`
  - `memory=1 GiB`
  - `disk=5 GiB`
- region:
  - `us-central1`

Immediate monitoring goal:
- verify that the direct coordinator starts cleanly
- verify train and rollout children are created
- then watch whether the run reaches the first rollout batches / trainer steps and eventually the first checkpoint-save boundary without reproducing the earlier executor-only confounders

## 2026-03-24T17:00PT — relaunching `E4` at 100 steps instead of 500

Change in plan:
- the original live `E4` root job `/ahmed/irl-e4p-0324-1656` was killed by user request before any meaningful RL work started
- next probe keeps the same `direct + GCS + prod` shape, but reduces `num_train_steps` from `500` to `100`

Reason:
- `500` steps is too long for the immediate regression probe cadence
- `100` steps is still large enough to move beyond the tiny debug envelope and exercise the heavier direct/GCS production-like path
- this should give a faster signal on whether the large-envelope branch reproduces train-side instability / OOM / checkpoint-coupled issues

Unchanged aspects of the relaunch:
- direct coordinator topology
- GCS-backed Llama 3.1 8B Instruct artifact
- short naming discipline to avoid the known Iris workdir path bug
- root region pinned to `us-central1`

Live `E4-100` launch details:
- root job:
  - `/ahmed/irl-e4p-100-0324-1701`
- command:
  - `uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --experiment-name-suffix e4p --num-train-steps 100`
- monitoring state file:
  - `scratch/20260324-1701_monitoring_state.json`

`E4-100` babysit update:
- original root `/ahmed/irl-e4p-100-0324-1701` reached real train/rollout startup, then degraded after the outer direct job was preempted / restarted
- observed state before recovery:
  - root `RUNNING` with `preemption_count=1` but task state back to `pending`
  - train child `KILLED` with `Parent task preempted`
  - rollout child `KILLED` with `Parent task preempted`
- this looked like stalled recovery rather than forward progress, so the run was killed and relaunched as:
  - `/ahmed/irl-e4p-100r1-0324-1710`
- monitoring state file was updated with `restart_count=1`

## 2026-03-24T17:17PT — deep dive on `E4-100` root preemption / worker-failure recovery semantics

Observed failure chain on the first 100-step `E4` run:
- root job:
  - `/ahmed/irl-e4p-100-0324-1701`
- it progressed into real RL work:
  - root coordinator started
  - train child submitted
  - rollout child submitted
  - rollout initialized vLLM and loaded the GCS checkpoint
  - train loaded the initial checkpoint and logged `Progress on:train -/100`
- then the outer/root worker died before the run reached a stable checkpoint boundary

Exact Iris evidence from `job bug-report`:
- root task attempt `0` ended as `worker_failed`
- worker error:
  - `Worker marin-cpu-vm-e2-highmem-2-ondemand-us-ce-20260324-2354-a0ab70dd-worker-0 failed: Request timed out`
- root job aggregate state before manual kill:
  - non-terminal
  - `preemption_count=1`
- train child terminal state:
  - `KILLED`
  - error `Parent task preempted`
- rollout child terminal state:
  - `KILLED`
  - error `Parent task preempted`

Important clarification:
- this was not observed as a clean RL-code crash
- and it was not proven to be literal cloud-VM preemption in the GCP sense
- the concrete low-level event recorded by Iris was a root worker failure / timeout
- Iris then routed that event through its preemption-retry path

Why the root looked like it "should have restarted" but the children died anyway:
- the root *was* automatically retrying at the Iris task level
- the job had not exhausted preemption retries
- that is why the root was still non-terminal and had `preemption_count=1`
- however, Iris killed descendant jobs when the parent task was treated as preempted

Relevant Iris controller logic:
- when an executing task loses its worker, Iris increments `preemption_count` and can set the task back to `PENDING` if retries remain
- if the effective preemption policy is `JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN`, Iris also cascades a kill to descendants with reason `Parent task preempted`
- code points:
  - `lib/iris/src/iris/cluster/controller/transitions.py` around `_resolve_preemption_policy(...)`
  - `lib/iris/src/iris/cluster/controller/transitions.py` around `_cascade_children(...)`
  - `lib/iris/src/iris/cluster/controller/transitions.py` around the worker-failure transition that calls `_cascade_children(..., "Parent task preempted")`

Crucial default policy detail:
- single-task parent jobs default to:
  - `JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN`
- direct RL root jobs are single-task jobs
- so absent an explicit override, child train/rollout jobs are intentionally killed when the root task is lost

Why preserving train/rollout children would not actually be safe in the current direct topology:
- in the direct topology, the root job is also the RL coordinator
- that root process hosts critical shared actors in-process via `client.host_actor(...)`:
  - curriculum actor
  - run-state actor
  - Arrow Flight weight-transfer coordinator
- therefore when the root worker dies, those actor endpoints die with it
- train and rollout depend on those endpoints
- so a hypothetical `PRESERVE_CHILDREN` policy would leave train/rollout alive but attached to dead coordinator-hosted infrastructure
- that would not be a correct recovery model with the current architecture

Most precise conclusion:
- the direct RL topology is only restart-safe at the granularity of the *whole subtree*
- it is not child-preserving restart-safe
- current Iris behavior is therefore coherent with the architecture:
  - retry the root task
  - kill child jobs because the parent-hosted control-plane died

Operator/monitoring takeaway:
- manual restart after the first parent preemption was likely premature if the goal was to observe automatic whole-subtree recovery
- the root had not yet failed terminally
- I manually killed `/ahmed/irl-e4p-100-0324-1701` while it was still in the retry path
- that intervention prevented us from seeing whether the root would have eventually restarted cleanly and re-submitted new train/rollout children

However, the existing state before manual kill was legitimately degraded:
- root remained `RUNNING` but its only task was back in a retriable/pending state
- both prior children were already terminal `KILLED`
- there was no evidence yet that a fresh child-generation cycle had begun
- so the decision to restart was operationally understandable, but it should not be confused with proof that automatic recovery was impossible

What this means for robustness:
- current direct/prod runs can survive parent worker loss only if the whole coordinator subtree restarts from the root
- progress loss before checkpoint is expected under that model
- because checkpoint interval is still relatively coarse (`600s`), an early root worker loss can discard meaningful startup work

Implications for next fixes / experiments:
1. Operational policy:
- for the next direct/prod run, do not manually kill on first root preemption / worker failure
- allow the root retry path to continue long enough to confirm whether Iris re-runs the coordinator and re-submits fresh children

2. Better fault-tolerance within the current architecture:
- reduce checkpoint interval for prod probes if we want whole-subtree restarts to lose less work
- this does not preserve children, but it improves recovery cost

3. Architectural requirement for true child-preserving recovery:
- move coordinator-hosted shared actors out of the parent/root process into durable standalone jobs/services first
- only after that would a `PRESERVE_CHILDREN`-style policy become technically safe for train/rollout

4. API / framework gap:
- Fray `JobRequest` currently does not expose Iris preemption policy directly
- even if we add that, switching the direct root to preserve children is not safe today because the parent hosts critical runtime state in-process

Cross-reference:
- detailed note captured in:
  - `docs/debug-log-e4-root-preemption-recovery.md`

Current live follow-up run state at the time of this note:
- restarted root:
  - `/ahmed/irl-e4p-100r1-0324-1710`
- this follow-up run was submitted after killing the degraded first root
- its current state should be checked separately when deciding whether to continue passive monitoring or intervene again

## 2026-03-24 20:45 PT — E4-100r1 postmortem: train OOM at checkpoint save, rollout only collateral

Run under analysis:
- root: `/ahmed/irl-e4p-100r1-0324-1710`
- entrypoint: `uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --experiment-name-suffix e4p --num-train-steps 100`

Terminal Iris states from `iris job bug-report`:
- root `/ahmed/irl-e4p-100r1-0324-1710`: `failed`
- train `/ahmed/irl-e4p-100r1-0324-1710/rl-e4p-20260325-001651-train`: `failed`
- rollout `/ahmed/irl-e4p-100r1-0324-1710/rl-e4p-20260325-001651-rollout-0`: `killed`

Root summary:
- submitted: `2026-03-25T00:10:02.434000+00:00`
- started: `2026-03-25T00:14:23.911000+00:00`
- finished: `2026-03-25T02:28:12.019000+00:00`
- duration: `2h 13m 48s`
- root error: `fray.v2.client.JobFailed: Job /ahmed/irl-e4p-100r1-0324-1710/rl-e4p-20260325-001651-train finished with status failed`

What definitely happened:
1. The direct + GCS + prod topology launched correctly.
2. Both child jobs were submitted and started.
3. Rollout produced initial rollouts; train got past the long initial wait boundary.
4. Train completed at least step `0` and step `1`.
5. Train then began checkpoint save for step `2`.
6. The train container was OOM-killed during that checkpoint save.
7. The root failed because train failed.
8. Rollout was then killed as fallout; it was not the primary cause.

Critical failure boundary:
- train logs show checkpoint start at `2026-03-25 02:27:06 UTC`
- exact lines observed before death:
  - `Saving temporary checkpoint at step 2.`
  - `Saving checkpoint at step 2 to gs://marin-us-central1/checkpoints/e4p-20260325-001651/e4p-20260325-001651-train/step-2`
  - `Saving checkpoint to .../step-2 for step 2`
- then the train task terminated with:
  - `Exit code 137: OOM killed (container exceeded memory limit)`

Train child details:
- state: `failed`
- error: `Exit code 137: OOM killed (container exceeded memory limit)`
- failure_count: `4`
- preemption_count: `2`
- finished: `2026-03-25T02:27:57.062000+00:00`

Rollout child details:
- state: `killed`
- error: `Job exceeded max_task_failures`
- finished: `2026-03-25T02:28:12.019000+00:00`
- interpretation: rollout was not the first failure in the subtree; it died after the train-side failure caused the run to unwind.

Why this matters:
- this run got far enough to rule out launch-time regressions as the active blocker for E4
- it was not blocked by:
  - region placement
  - GCS bootstrap
  - direct coordinator startup
  - initial rollout startup
  - first train-step execution
- the live blocker remains the older hypothesis from the long-run thread:
  - train-side memory blow-up around checkpoint save

Most precise conclusion:
- `E4-100r1` did not die because rollout was buggy.
- `E4-100r1` died because train hit an OOM at the step-2 checkpoint boundary.
- rollout death was secondary fallout, not the root cause.

Implications for next experiments:
1. If the goal is isolating the long-run blocker, the next probe should change checkpoint behavior first, not rollout.
2. Best immediate ablations:
   - disable checkpoint saving entirely
   - or make checkpoint interval much larger than the run length
   - or reduce train memory pressure before the next checkpoint boundary
3. This run is strong evidence that the direct + GCS + prod path is fundamentally alive up to real RL progress, and that the dominant remaining failure mode is checkpoint-coupled train OOM.

## 2026-03-24 20:55 PT — train-memory accounting explains the 8B OOM; zero-KL should not retain extra model copy

Follow-up question investigated:
- how can an 8B run consume ~116 GiB sampled peak host RAM on the train worker?

What the Iris task record actually showed for the failed train task:
- last sampled memory (`resource_usage.memory_mb`): `42659 MB` (~41.7 GiB)
- sampled peak (`resource_usage.memory_peak_mb`): `119193 MB` (~116.4 GiB)
- note: the dashboard task table currently shows only `memory_mb`, not `memory_peak_mb`, so the default job page is misleading for OOM diagnosis.

Important dashboard/metrics clarification:
- task-row `MEM` on the job page is the last sampled current memory, not peak
- Iris samples container stats every `5s`
- Docker's CLI stats on Linux also subtract cache from total memory usage, so even the sampled current/peak values can understate the cgroup limit pressure at the exact OOM instant
- therefore `42 GiB visible in the row` is not contradictory with `OOM at 128 GiB limit`

Code-level memory accounting that explains the high train memory:
1. Trainer params are configured with `p=f32, c=bfloat16` in the E4 probe.
   - file: `experiments/exp_iris_rl_regression_direct_gcs_prod.py`
   - line of interest: `mp=jmp.get_policy("p=f32,c=bfloat16")`
2. Levanter `TrainerState` stores both:
   - `model`
   - `opt_state`
   - file: `lib/levanter/src/levanter/trainer_state.py`
3. `TrainerState.init(...)` creates Adam optimizer state for the trainable parameters.
   - file: `lib/levanter/src/levanter/trainer_state.py`
4. Marin RL train worker also retains a separate bootstrap/reference model object.
   - file: `lib/marin/src/marin/rl/train_worker.py`
5. Checkpointing saves the full `info.state.saveable_state`, not only raw model weights.
   - files:
     - `lib/levanter/src/levanter/trainer.py`
     - `lib/levanter/src/levanter/checkpoint.py`

Rough host-RAM budget for an 8B run under this code path:
- trainable params in `f32`: ~29.8 GiB
- Adam moments (`m`, `v`) in `f32`: ~59.6 GiB
- separately retained bootstrap/reference model in `bf16`: ~14.9 GiB (minimum plausible extra copy)
- subtotal before transient overhead: ~104 GiB
- add replay/batch/checkpoint staging/transient save-time overhead and the observed sampled peak of ~116 GiB becomes entirely plausible

Key code smell found:
- the E4 experiment sets `kl_coef=0.0`
- but the train worker still loads and retains `reference_model`
- `RLOOLoss.create_loss_fn(...)` captures `reference_model` even though the KL branch is disabled for this run
- therefore the zero-KL path is paying unnecessary host RAM for a model copy it does not need during training

Planned fix (implemented next in this session):
1. Teach RL loss modules to declare whether they need a reference model.
2. In the zero-KL RLOO path, do not retain the bootstrap/reference model after trainer state is created.
3. Keep checkpoint/OOM as the main remaining hypothesis even after this fix, but recover the wasted zero-KL memory first because it is the cleanest local win.

## 2026-03-24 21:05 PT — implemented zero-KL train-memory reduction

Follow-through on the memory diagnosis:
- the zero-KL E4 path was retaining an unnecessary extra model object in the train worker
- the goal of this patch is narrow and explicit:
  - if `kl_coef == 0`, do not keep a separate reference-model copy resident after the trainer state has been materialized

Implementation details:
1. RL loss interface now exposes whether a loss requires a retained reference model.
   - file: `lib/marin/src/marin/rl/rl_losses.py`
   - `RLOOLoss.needs_reference_model()` returns `self.kl_coef > 0`
2. `RLOOLoss.create_loss_fn(...)` now accepts `reference_model: eqx.Module | None`.
   - when `kl_coef > 0`, it fails fast if `reference_model is None`
   - when `kl_coef == 0`, the zero-KL path is valid without a retained reference model
3. Train worker bootstrap path was tightened.
   - file: `lib/marin/src/marin/rl/train_worker.py`
   - the initial loaded model is now stored as `initial_model`
   - `reference_model` remains as a compatibility alias during worker construction
   - immediately after `trainer.initial_state(...)` is created, `_drop_bootstrap_model_references()` runs
   - for zero-KL runs this clears both:
     - `initial_model`
     - `reference_model`
   - for positive-KL runs it keeps `reference_model`

Why this change is technically sound:
- the train state already owns the trainable model copy after `trainer.initial_state(...)`
- the zero-KL loss path does not consume `reference_model`
- therefore retaining the separate bootstrap/reference object past initialization is wasted host RAM in the zero-KL case

Expected impact:
- reduce steady-state train-worker host RAM for zero-KL runs by roughly one extra model copy
- for an 8B-class run under this code path, this should save on the order of tens of GiB, not a few MB
- checkpoint spikes may still exist, but the baseline memory floor should drop materially

Validation completed:
- `uv run pytest -q tests/rl/test_loss.py tests/rl/test_train_worker.py`
  - result: `9 passed`
- `./infra/pre-commit.py --fix .agents/logbooks/iris-rl-codex.md lib/marin/src/marin/rl/rl_losses.py lib/marin/src/marin/rl/train_worker.py tests/rl/test_loss.py tests/rl/test_train_worker.py`
  - result: `OK`

Tests added/updated:
- `tests/rl/test_loss.py`
  - asserts `RLOOLoss(kl_coef=0).needs_reference_model() == False`
  - asserts positive-KL rejects missing reference model at loss-construction time
- `tests/rl/test_train_worker.py`
  - asserts zero-KL cleanup clears retained bootstrap/reference model references
  - asserts positive-KL cleanup preserves the retained reference model

Most precise current status:
- the logbook now contains both:
  1. the E4-100r1 checkpoint OOM postmortem
  2. the code change that removes the unnecessary zero-KL retained model copy
- next empirical question remains unchanged:
  - is the checkpoint-step OOM fully solved by dropping the extra model copy, or does checkpoint save still need its own ablation?

## 2026-03-24 22:40 PT — second post-fix train OOM captured with live task-memory polling

Current prod probe remains:
- root: `/ahmed/irl-e4p-100r2-0324-2103`
- active subtree run id: `e4p-20260325-042445`

Status snapshot at log time:
- root job: `RUNNING`
  - `preemption_count=1`
- train child: `RUNNING`
  - `failure_count=2`
- rollout child: `RUNNING`

This matters because it proves the zero-KL memory reduction did **not** eliminate the checkpoint/save-time OOM family. The train task has now failed twice in the same subtree and is running on `attempt_id=2`.

### What happened on the current subtree

Train task id:
- `/ahmed/irl-e4p-100r2-0324-2103/rl-e4p-20260325-042445-train/0`

Controller state now shows three attempts:
1. `attempt 0`
   - `FAILED`
   - `exit_code=137`
   - OOM during checkpoint save at step 2
2. `attempt 1`
   - `FAILED`
   - `exit_code=137`
   - also OOMed
3. `attempt 2`
   - currently `RUNNING`

The train job summary now reports:
- `failure_count=2`
- job state still `RUNNING` because Iris retried the task rather than failing the entire train child immediately

### New direct evidence: live task-memory poller

Added polling tools in this session:
- `scripts/iris/poll_task_memory.py`
- `scripts/iris/run_poll_task_memory.sh`

Purpose:
- sample Iris controller task status at a fixed cadence
- write one JSON object per sample to JSONL
- include UTC timestamps so we can line up memory behavior against train logs

Important schema fix made immediately after first use:
- the initial version wrote the task-level `error` field directly
- for retrying tasks, Iris keeps the last failed attempt's error at the task level even while the current attempt is running
- that made every sample look like a live OOM even when the current attempt was healthy
- patched output now separates:
  - `current_attempt_*`
  - `last_failed_attempt_*`
  - raw `task_*` fields
- console output now reports the live attempt cleanly and only annotates the last failed attempt as context

### Memory ramp captured for `attempt 1`

Using the poller on the train task, the following samples were captured right before the second OOM/retry boundary:
- `2026-03-25T05:35:42.745701+00:00`
  - `attempt=1`
  - `mem_mb=85329`
  - `peak_mb=85329`
- `2026-03-25T05:35:47.921789+00:00`
  - `attempt=1`
  - `mem_mb=102809`
  - `peak_mb=102809`
- `2026-03-25T05:35:53.133527+00:00`
  - `attempt=1`
  - `mem_mb=124006`
  - `peak_mb=124006`

Then the task transitioned:
- `2026-03-25T05:36:13.872538+00:00`
  - `attempt=2`
  - `state=TASK_STATE_BUILDING`
- `2026-03-25T05:36:19.070152+00:00`
  - `attempt=2`
  - `state=TASK_STATE_RUNNING`
  - memory reset near zero as the new attempt began

This is the clearest evidence so far that the second failure was a real second OOM, not just a stale controller error string.

### Current live memory on `attempt 2`

After the retry to `attempt 2`, current controller samples have been much lower than the failed ramp:
- around `40-45 GB` resident
- current sampled peak on `attempt 2` so far is in the mid-40-GB range

Example controller read after retry stabilization:
- `memory_mb=44912`
- `memory_peak_mb=44912`
- `current_attempt_id=2`
- `current_attempt_state=TASK_STATE_RUNNING`

So the current retry is alive and not yet near the old `~124 GB` boundary.

### Interpretation

What this changes:
- before the poller, the second OOM was strongly suspected from controller retry state
- after the poller, we have direct sample-by-sample evidence that `attempt 1` climbed to ~`124 GB` immediately before the task rolled into `attempt 2`

What this does **not** yet prove:
- the exact instantaneous peak that triggered the kernel kill
- Iris samples every few seconds, so the true kill point could have been slightly higher than the recorded `124006 MB`

But the practical conclusion is now hard to avoid:
- even after dropping the unnecessary zero-KL retained model copy, `128 GiB` is still not enough margin for this prod train path under its current checkpoint/save behavior

### Operational takeaway

If the goal is to keep this exact prod probe alive without another redesign first, the obvious next trial is:
1. relaunch with materially higher host RAM for train/rollout TPU jobs
2. preserve the memory poller so we can verify whether the new headroom clears the old checkpoint boundary

### Tooling usage recorded for next agent

Foreground:
```bash
bash scripts/iris/run_poll_task_memory.sh \
  /ahmed/irl-e4p-100r2-0324-2103/rl-e4p-20260325-042445-train/0
```

Detached:
```bash
DETACH=1 bash scripts/iris/run_poll_task_memory.sh \
  /ahmed/irl-e4p-100r2-0324-2103/rl-e4p-20260325-042445-train/0
```

JSONL records include UTC timestamps such as:
- `2026-03-25T05:37:57.236251+00:00`

Most recent conclusion at this point:
- root-job preemption recovery is working
- direct RL subtree recreation is working
- zero-KL memory reduction was directionally correct but insufficient
- the active blocker for the prod run remains train-side OOM under `128 GiB`
- live memory polling is now available, working, and should be kept on for any future relaunch

## 2026-03-24 22:49 PT — raised RL TPU host-RAM request to 300g for prod relaunch

Decision after the second post-fix train OOM:
- `128g` host RAM is no longer a reasonable request for the prod `v5p-8` train path under current checkpoint behavior
- instead of continuing to probe the same boundary, the next run should request materially more host RAM on both TPU jobs

Patch applied:
1. Shared RL deployment config now supports explicit TPU host-RAM overrides.
   - file: `lib/marin/src/marin/rl/rl_job.py`
   - added fields on `RunConfig`:
     - `train_ram: str | None`
     - `inference_ram: str | None`
2. RL orchestration now forwards those overrides into `ResourceConfig.with_tpu(...)` when present.
   - file: `lib/marin/src/marin/rl/orchestration.py`
3. Executor-backed experiment config can now carry the same override path for future E5-style probes.
   - file: `lib/marin/src/marin/rl/rl_experiment_utils.py`
   - added fields on `RLExperimentConfig`:
     - `train_ram: str | None`
     - `inference_ram: str | None`
4. The current direct prod probe script was updated to request `300g` for both trainer and rollout workers.
   - file: `experiments/exp_iris_rl_regression_direct_gcs_prod.py`
   - constant: `PROD_TPU_WORKER_RAM = "300g"`
   - wired into `RunConfig(train_ram=..., inference_ram=...)`

Why this is the chosen next experiment:
- current RL train attempt hit a measured sampled peak of ~`124 GB` before retrying
- the worker class is still `v5p-8`, whose host memory budget is substantially higher than the default `128g` request
- the immediate goal is to stop failing on the host-RAM request boundary and learn whether checkpoint/save logic still fails once headroom exists

Validation completed:
- `uv run pytest -q tests/rl/test_orchestration.py tests/rl/test_rl_experiment_utils.py`
  - result: `12 passed`
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/rl_job.py lib/marin/src/marin/rl/orchestration.py lib/marin/src/marin/rl/rl_experiment_utils.py experiments/exp_iris_rl_regression_direct_gcs_prod.py tests/rl/test_orchestration.py tests/rl/test_rl_experiment_utils.py`
  - result: `OK`

Tests added:
- `tests/rl/test_orchestration.py`
  - verifies train and rollout submissions honor `RunConfig.train_ram` / `RunConfig.inference_ram`
- `tests/rl/test_rl_experiment_utils.py`
  - verifies executor-backed RL config propagation preserves RAM overrides into `RunConfig`

Immediate next step after this entry:
- relaunch the direct GCS prod probe with the new `300g` train/rollout RAM requests
- keep the task-memory poller attached from the start

## 2026-03-24 22:51 PT — launching next direct prod probe with 300g TPU host RAM

Executing the next planned experiment after terminating `r2`:
- same direct + GCS + prod topology
- same 100-step envelope as the recent prod probe
- short job-name envelope preserved
- new change under test: `300g` host RAM request for both train and rollout TPU jobs

Launch intent:
- determine whether higher host-RAM headroom clears the repeated early train OOM / retry boundary
- keep memory polling available from the start for the train task

## 2026-03-24 23:43 PT — Overnight Babysit Armed For `r3`

Current live root job:
- `/ahmed/irl-e4p-100r3-0324-2245`

Current observed state before handoff/end-of-turn:
- root: `JOB_STATE_RUNNING`
- rollout: `JOB_STATE_RUNNING`
- train: `JOB_STATE_RUNNING`, `preemption_count=1`

Important new observations:
- `300g` TPU host RAM materially changed the failure envelope.
- The train retry on worker `marin-tpu_v5p_8-us-central1-a-20260325-0602-86a14bb9-worker-0` reached sampled `memory_peak_mb=224972` while staying alive.
- This is strong evidence that the prior `128 GiB` OOMs were a real host-RAM ceiling, not a misleading dashboard artifact.
- The run also survived a non-OOM train worker failure (`Request timed out`) via train-task retry.
- Retry still restarts from the base model because no earlier durable trainer checkpoint existed at that boundary.

Progress boundary cleared on `r3`:
- Logs show checkpoint save triggered at step `4`:
  - `Saving temporary checkpoint at step 4`
  - `Saving checkpoint at step 4 ...`
- Later logs show:
  - `Transferring weights at step 4`
  - `Training step 4 completed`
- Therefore `r3` has already crossed the old checkpoint-coupled failure boundary that killed the `128 GiB` runs.

Overnight babysitter:
- scratch script: `scratch/babysit_iris_job.sh`
- state file: `scratch/20260324-2338_monitoring_state.json`
- monitor log: `scratch/20260324-2338_babysit.log`
- behavior: 120s startup check, then 570s cadence, auto-resubmit on terminal root non-success using the saved resubmit command.

Memory pollers still relevant:
- user-owned JSONL: `scratch/20260325-055054Z_rl-e4p-20260325-054703-train_memory.jsonl`
- agent-owned JSONL: `scratch/e4p_r3_train_memory_agent.jsonl`

Operational note:
- The detached babysitter was smoke-tested locally with a short cadence before the real 8-hour launch.
- It is intentionally in `scratch/` to avoid polluting the repo with one-off operational tooling.

## 2026-03-25 09:44 PT — froze dead `r3` subtrees locally and wrote retry chronology

Root job under analysis:
- `/ahmed/irl-e4p-100r3-0324-2245`

Reason for this archival pass:
- the overnight babysitter itself expired after its 8-hour window, which initially made it look like the RL job had died
- after checking the live Iris tree, the root job was still `RUNNING`
- to avoid losing evidence from the already-dead child subtrees, I exported all currently retrievable logs for the dead `r3` branches into a stable local bundle

Local bundle index:
- [E4 r3 Retry Log Bundle](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/README.md)

What was archived:

### Subtree `e4p-20260325-054703` (initial long-running subtree)
Raw logs:
- [054703 train raw log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/054703_train.log)
- [054703 rollout raw log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/054703_rollout.log)
- [054703 train timeline extract](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/054703_train_timeline.log)
- [054703 train memory trace JSONL](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260325-055054Z_rl-e4p-20260325-054703-train_memory.jsonl)
- [054703 controller bug-report note](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/054703_train_bug_report.txt)

Important evidence from this subtree:
- this is the subtree that matches the memory-poller run id the user pasted (`e4p-20260325-054703`)
- it did not merely run a few minutes; it ran for hours and retried in-place multiple times inside the same child job
- the raw train log shows the subtree reaching at least step `37`
- the final visible terminal event in the train raw log is:
  - `Container was OOM killed by the kernel`
  during the step-37 checkpoint save path
- the controller no longer serves `job bug-report` for the exact `054703` child job, but `iris job logs` still returns the raw logs, so those were exported while still available

Condensed chronology from the exported train timeline:
- initial startup from base model
- no checkpoints available initially
- step `0` completed once
- early worker failure / retry inside the same child job
- resumed from scratch at first because no trainer checkpoint existed yet
- later checkpoints at steps `2,4,6,8,...`
- OOM at step `10`
- later retry resumes from checkpoint `8`
- OOM again at step `31`
- later retries continue, with progress through the 30s
- final visible OOM occurs during checkpoint save at step `37`

### Subtree `e4p-20260325-125232` (short-lived replacement subtree)
Raw logs:
- [125232 train raw log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/125232_train.log)
- [125232 rollout raw log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/125232_rollout.log)
- [125232 train timeline extract](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/125232_train_timeline.log)
- [125232 rollout timeline extract](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/125232_rollout_timeline.log)
- [125232 train bug report](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/125232_train_bug_report.txt)
- [125232 rollout bug report](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/125232_rollout_bug_report.txt)

Important evidence from this subtree:
- `125232` is a separate replacement subtree under the same root job
- train and rollout both started successfully
- train never got past `Waiting for initial rollouts...`
- rollout was still in the vLLM/TPU bring-up and ragged-paged-attention compile-warning phase
- both children were then killed with controller-level reason:
  - `Parent task preempted`
- exact controller durations:
  - train: `5m 54s`
  - rollout: `5m 45s`

Root-level controller context exported alongside the raw logs:
- [root retry story grep log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/root_retry_story.log)
- [root bug report](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/root_bug_report.txt)

Most precise retry story we can currently support from the exported evidence:
1. `r3` root launched initial subtree `e4p-20260325-054703`.
2. That subtree ran for hours and the train child retried in-place multiple times.
3. The `054703` train child repeatedly made real progress and checkpointed, eventually reaching at least step `37`.
4. The final visible terminal event in the `054703` train raw log is another checkpoint-coupled OOM (`Container was OOM killed by the kernel`).
5. Later, the root itself suffered a parent-level preemption / worker-loss event (documented in the current root bug report as a timed-out CPU worker and `preemption_count=1`).
6. After that parent-level retry, Iris launched a fresh subtree `e4p-20260325-125232`.
7. That new subtree was killed quickly with `Parent task preempted` before it got past initial rollout startup.
8. The root then launched the next replacement subtree `e4p-20260325-130820`, which is the currently live branch.

Clarification for future readers:
- the user-pasted JSONL from `scripts/iris/run_poll_task_memory.sh` is memory telemetry for one train task, not the raw Iris task logs
- both forms of evidence are now archived locally and linked above

## 2026-03-25 09:47 PT — clarified `r3` lineage: the 6-minute subtree was not the many-hour run

This needed to be stated explicitly because the dashboard/W&B view made the chronology look contradictory.

Correct interpretation:
- the many-hour `r3` training run was subtree `e4p-20260325-054703`
- the ~6 minute train/rollout pair visible in the dashboard was subtree `e4p-20260325-125232`
- the currently live replacement subtree is `e4p-20260325-130820`

What was confusing:
- W&B page `e4p-20260325-130820-train` only shows the duration of the current replacement train child
- the dashboard screenshot showing `rl-e4p-20260325-125232-{train,rollout-0}` with durations around `5m 45s` / `5m 54s` was easy to misread as “the original long run only trained for 6 minutes”

What actually happened:
1. Root `/ahmed/irl-e4p-100r3-0324-2245` first launched subtree `e4p-20260325-054703`.
2. `054703` train ran for hours and retried in-place multiple times.
3. Exported raw train logs for `054703` span:
   - first visible log line: `05:47:14 UTC`
   - last visible log line: `12:39:20 UTC`
4. During that span, `054703` did real training and checkpoint work; it was not idle.
5. Only later, after a parent/root preemption event, did Iris launch the short-lived replacement subtree `125232`.
6. `125232` train never got past `Waiting for initial rollouts...` before both children were killed with `Parent task preempted`.
7. The root then launched the next replacement subtree `130820`, which is the current live branch.

Plain-English takeaway:
- the hours of training belonged to `054703`
- the 6-minute durations belonged to a later replacement subtree `125232`
- they are different child jobs under the same root lineage and must not be compared as if they were the same run

Evidence links:
- [054703 train raw log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/054703_train.log)
- [054703 train timeline extract](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/054703_train_timeline.log)
- [125232 train bug report](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/125232_train_bug_report.txt)
- [root bug report](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r3_retry_logs/root_bug_report.txt)

## 2026-03-25 10:24 PT - Relaunch With 400 GiB TPU Host RAM Requests

- Updated [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py) to raise `PROD_TPU_WORKER_RAM` from `"300g"` to `"400g"`.
- This affects both `run_config.train_ram` and `run_config.inference_ram`, so both trainer and rollout now request `400 GiB` host RAM.
- Rationale: the successful on-demand `exp2039-20260318-040100-train` W&B run showed host RSS well above the prior Iris cap:
  - median RSS `~197.6 GB`
  - p95 RSS `~298.5 GB`
  - max RSS `~393.9 GB`
- Relaunched fresh 100-step direct+GCS+prod probe:
  - root job: `/ahmed/irl-e4p-100r4-0325-1023`
  - submit command: `uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --experiment-name-suffix e4p --num-train-steps 100`
- This run is intended to match the successful on-demand memory envelope more closely before claiming a deeper RL pipeline bug.

## 2026-03-25 12:29 PT - R4 Rollout "Crash" Was A Worker Heartbeat Timeout, Not An OOM

Observed on run:
- root: `/ahmed/irl-e4p-100r4-0325-1023`
- rollout child: `/ahmed/irl-e4p-100r4-0325-1023/rl-e4p-20260325-172718-rollout-0`
- trainer child: `/ahmed/irl-e4p-100r4-0325-1023/rl-e4p-20260325-172718-train`

What W&B made visible:
- the rollout W&B run briefly appeared "crashed", then resumed on the same run id.
- this looked like a rollout failure from the UI, but Iris state showed the rollout child itself never went terminal.

What Iris state shows:
- rollout child is still `RUNNING`
- rollout child has `preemption_count=1`, `failure_count=0`
- rollout bug report shows:
  - attempt `0`: `worker_failed`
  - error: `Worker marin-tpu-v5p-8-us-central1-a-20260325-1730-7efed3e6-worker-0 failed: Request timed out`
  - attempt `1`: `running` on `marin-tpu-v5p-8-us-central1-a-20260325-1845-d5289490-worker-0`
- trainer remained alive throughout this event:
  - train job still `RUNNING`
  - no train preemption attached to the rollout retry

Interpretation:
- this was **not** a rollout-model OOM
- this was **not** an RL application traceback
- this was a **worker/control-plane failure** on the first rollout TPU host attempt
- Iris retried the rollout child in place, and W&B continued on the same rollout run id after the retry

Exact controller/worker path for `Request timed out`:
- controller heartbeats workers through [worker_provider.py:131](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/worker_provider.py:131)
- worker RPC client timeout is hardcoded to `5.0s` in [worker_provider.py:41](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/worker_provider.py:41)
- controller heartbeat cadence is `5s` in [controller.py:620](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/controller.py:620)
- repeated heartbeat failures are counted in [transitions.py:1272](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/transitions.py:1272)
- once the failure threshold is crossed, Iris marks the worker dead and retries the task in [transitions.py:1278](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/transitions.py:1278) and [transitions.py:1320](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/transitions.py:1320)
- controller then processes the failed worker in [controller.py:1490](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/controller.py:1490)

Why this may be brittle even if the worker was not actually dead:
- worker heartbeat handling does real work under the same RPC envelope:
  - reconciliation, log draining, host metrics, health check
- see [worker.py:563](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/worker/worker.py:563) through [worker.py:667](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/worker/worker.py:667)
- worker also resets itself if controller heartbeats stop arriving long enough:
  - [worker.py:349](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/worker/worker.py:349)
- so the current system has a narrow control-plane margin for TPU workers under load

Most likely explanations for this specific rollout interruption:
1. TPU worker host stalled or became unhealthy briefly
2. network/control-plane hiccup between controller and worker
3. worker heartbeat handler exceeded the controller-side `5s` RPC timeout enough times in a row

What this incident does **not** support:
- not evidence of rollout OOM
- not evidence of trainer OOM
- not evidence of a deterministic RL rollout logic bug

What this incident **does** support:
- Iris rollout retry behavior is working: the child was retried in place and resumed
- Iris heartbeat envelope for TPU workers is probably too aggressive / too opaque for long-running RL

Hardening recommendations:
1. raise worker heartbeat RPC timeout above `5s` for TPU workers, likely `15-20s`
2. add controller-side heartbeat latency metrics per worker
3. add worker-side `handle_heartbeat` duration metrics and warnings that survive task retry
4. capture worker process status/logs automatically when a worker is declared failed so postmortems have real evidence instead of just `Request timed out`
5. keep rollout/train as independently retryable child jobs; that part behaved correctly here

## 2026-03-25 12:44 PT - Controller Confirmed The Rollout Worker Loss Exactly

This sharpens the earlier `r4` rollout diagnosis.

Earlier wording said the rollout interruption was the **most likely** result of repeated heartbeat failures. After checking controller-side evidence directly, that is now confirmed rather than inferred.

Controller transaction-log evidence:
- queried `txn_actions` for worker `marin-tpu-v5p-8-us-central1-a-20260325-1730-7efed3e6-worker-0`
- controller recorded repeated `worker_heartbeat_failed` actions for that exact worker
- the first visible failure was stronger than a generic timeout:
  - `Request failed: ... tcp connect error: Connection refused (os error 111)` to `10.128.0.97:10001`
- after that, controller recorded repeated heartbeat failures with:
  - `Request timed out`

Controller process-log evidence:
- controller log shows the worker registered normally at `10.128.0.97:10001`
- around `18:45 UTC`, repeated provider-sync failures were recorded for this exact worker, with failed sync durations in the `~5.1s` to `~7.5s` range
- controller then logged that the worker failed and the autoscaler terminated TPU slice `marin-tpu-v5p-8-us-central1-a-20260325-1730-7efed3e6`

So the Iris-observed event sequence is now clear:
1. rollout worker initially healthy
2. controller lost connectivity to the worker endpoint
3. controller recorded repeated heartbeat failures for that worker
4. autoscaler terminated the TPU slice
5. Iris retried rollout on a new worker attempt

This confirms:
- not a rollout OOM
- not a rollout Python exception
- not a trainer failure propagating downward
- it was a controller/worker connectivity failure on that TPU slice, followed by normal Iris retry behavior

Important precision:
- controller logs confirm **what Iris observed**
- they do **not** by themselves prove whether the underlying cause was worker-daemon crash vs TPU-host VM issue vs transient network/control-plane fault
- but the control-plane event is no longer speculative: repeated heartbeat failures really happened

Implication for hardening:
- the relevant robustness target is the Iris controller/worker heartbeat path, not RL application logic
- next platform-level fix still points to:
  1. larger controller-side worker heartbeat RPC timeout
  2. heartbeat latency metrics on both controller and worker
  3. better automatic worker-failure evidence capture

## 2026-03-25 12:51 PT - Same W&B Rollout Run Resumed; Trainer Is Still The Main Remaining Risk

User-facing conclusion from `r4`:
- the important robustness property held for rollout: the same logical rollout job resumed under the same W&B run id after the first worker-attempt died
- so from the RL operator perspective, rollout recovery is acceptable for this incident even though the dashboard hid the attempt boundary

Current `r4` live state:
- root `/ahmed/irl-e4p-100r4-0325-1023` -> `RUNNING`, `failure_count=0`, `preemption_count=0`
- rollout `/ahmed/irl-e4p-100r4-0325-1023/rl-e4p-20260325-172718-rollout-0` -> `RUNNING`, `failure_count=0`, `preemption_count=1`
- train `/ahmed/irl-e4p-100r4-0325-1023/rl-e4p-20260325-172718-train` -> `RUNNING`, `failure_count=0`, `preemption_count=0`

Important trainer status:
- train bug report still shows exactly one live attempt:
  - attempt `0` on `marin-tpu-v5p-8-us-central1-a-20260325-1727-aae988dc-worker-0`
- so trainer has not yet demonstrated retry behavior on this run

Important live memory status:
- train memory poller is still attached:
  - [e4p_r4_train_memory_agent.jsonl](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4p_r4_train_memory_agent.jsonl)
- recent samples show the trainer climbing back into the high host-RAM regime but still under the new `400 GiB` cap
- sampled numbers during the latest observed checkpoint-growth phase:
  - current memory rose through roughly `205 GB -> 289 GB`
  - sampled peak remains `333004 MB` (`~325 GiB`)
- this is consistent with the earlier finding that `400 GiB` is much closer to the successful on-demand envelope than `300 GiB`

Neutral interpretation:
- rollout robustness is now good enough to stop making it the primary focus
- the next thing that actually matters is whether trainer can finish the long run under the `400 GiB` cap without:
  1. checkpoint-coupled OOM
  2. trainer worker failure / retry
  3. lost progress on trainer retry

So the practical next concern is no longer "does rollout recover?"; it is "does trainer stay alive and durable through repeated checkpoint/save boundaries and complete the run cleanly?"

## 2026-03-25 16:01 PT - Verified Claude's Performance Diagnosis Against Raw W&B Histories

Checked source:
- [iris-rl-claude.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-claude.md)

Verified directly against W&B API histories for:
- Iris train: `marin-community/marin_iris_rl_debug/e4p-20260325-172718-train`
- Iris rollout: `marin-community/marin_iris_rl_debug/e4p-20260325-172718-rollout-0`
- on-demand fast baseline train: `marin-community/marin_post_training/exp2039-nb-inflight2-train`
- on-demand fast baseline rollout: `marin-community/marin_post_training/exp2039-nb-inflight2`
- on-demand finished baseline train: `marin-community/marin_post_training/exp2039nb-20260319-032500-train`
- on-demand finished baseline rollout: `marin-community/marin_post_training/exp2039nb-20260319-032500`

What is confirmed:
- Iris `r4` really is much slower wall-clock per train step than the fast on-demand baseline.
- But trainer-side compute is not the bottleneck.
- The current Iris rollout shape is definitely pathological for throughput.

Measured train medians from W&B history:
- Iris `r4` train:
  - wall-clock/step: `473.1s` (`7.9 min`)
  - train step duration: `61.0s`
  - batch prep: `31.8s`
  - fwd/bwd: `38.8s`
  - weight transfer: `14.2s`
  - derived inter-step idle: `401.5s`
- on-demand `exp2039-nb-inflight2` train:
  - wall-clock/step: `136.4s` (`2.3 min`)
  - train step duration: `67.1s`
  - batch prep: `9.45s`
  - fwd/bwd: `62.6s`
  - weight transfer: `14.1s`
  - derived inter-step idle: `67.0s`
- on-demand finished baseline `exp2039nb-20260319-032500` train:
  - wall-clock/step: `215.7s` (`3.6 min`)
  - train step duration: `66.7s`
  - batch prep: `21.3s`
  - fwd/bwd: `51.0s`
  - weight transfer: `14.4s`
  - derived inter-step idle: `138.1s`

Interpretation from those numbers:
- trainer compute on Iris is not slower; it is actually slightly faster on raw step time than the fast on-demand run
- the large slowdown is real and almost entirely outside trainer compute

What Claude got right:
- the full-eval trigger in [rollout_worker.py:841](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rollout_worker.py:841) is keyed to the rollout loop's local `step`, i.e. rollout-batch iteration count, not trainer step count
- current Iris config in [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py) uses:
  - `n_prompts=16`
  - `n_generations_per_prompt=16`
  - `train_batch_size=1024`
- therefore each rollout batch contributes only `256` samples, so trainer needs roughly `4` rollout batches to fill one training batch
- W&B rollout history confirms this: Iris rollout median `cumulative_batch_count` delta between visible eval rows is `4`, with most deltas `3` or `5`
- so the eval semantics are a real throughput footgun for this Iris shape

What Claude overstated or mixed together:
- the prose combines metrics from different on-demand baselines; the `2.3 min` wall-clock baseline is from the crashed `nb-inflight2` run, while the quoted trainer-side `batch_prep` / `fwd_bwd` numbers align better with the finished baseline
- more importantly, the on-demand W&B histories do not support the simplified statement "1 rollout batch per train step -> 1 eval per train step" as a complete explanation
- finished on-demand rollout `exp2039nb-20260319-032500` shows eval rows roughly every `8` trainer steps with `cumulative_batch_count` jumping by about `10`
- therefore the performance gap is not isolated to `n_prompts` and eval cadence alone

Critical missing difference vs on-demand:
- on-demand `exp2039` used `inflight_weight_updates=True` in [exp2039_rl_math500.py](/Users/ahmed/code/marin3/experiments/exp2039_rl_math500.py)
- current Iris `r4` is effectively using `inflight_weight_updates=False` (default in [rl_job.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_job.py))
- with non-inflight mode, rollout does a blocking weight sync before each rollout batch, so with the current small-batch shape the run pays that coordination overhead many more times per trainer step

Current verdict:
- Claude identified a real bug / footgun
- Claude did not prove that this one issue fully explains the slowdown
- the more defensible conclusion is:
  1. small rollout batch shape (`n_prompts=16`)
  2. rollout-step-based eval cadence (`eval_frequency=1`)
  3. non-inflight / blocking sync behavior
  together explain the current poor throughput much better than any single factor alone

Recommendations from this verification:
1. clean parity experiment:
   - rerun Iris with `n_prompts=64`
   - and `inflight_weight_updates=True`
   - keep everything else the same
   - this is the fairest apples-to-apples comparison to the fast on-demand baseline
2. smallest surgical experiment:
   - keep current config shape
   - change only `eval_frequency` from `1` to `4`
   - this tests Claude's primary hypothesis directly
3. longer-term code fix:
   - change full-eval cadence to key off trainer step count or another trainer-aligned progress signal rather than rollout loop iterations

Bottom line:
- there is enough information to say Claude found a real throughput problem
- there is not enough information to say Claude completely explained the slowdown without also accounting for inflight-vs-blocking weight sync differences

## 2026-03-25 16:06 PT - Preparing Parity Run Against On-Demand exp2039

User requested a new parity run on the Iris cluster.

Plan for the parity run:
- keep the stable `400g` TPU host-RAM envelope from `r4`
- keep `v5p-8` for both trainer and rollout
- pin RL workers to `us-central1` (best available equivalent to the current `us-central1-a` TPU group; RL worker config takes region, not explicit zone)
- change rollout shape to match on-demand `exp2039` more closely:
  - `n_prompts=64`
  - `n_generations_per_prompt=16`
  - `inflight_weight_updates=True`
- keep the rest of the current direct + GCS + prod probe unchanged unless the code requires otherwise
- launch as a fresh 100-step run so the comparison is fast enough to inspect while still exercising steady-state behavior

Rationale:
- W&B verification showed Claude found a real throughput issue, but not the full story
- the cleanest next experiment is a parity run that changes both of the largest known rollout-side deltas from on-demand:
  1. rollout batch shape
  2. inflight vs blocking weight sync mode

## 2026-03-25 16:47 PT - Launched Iris Parity Run Against On-Demand exp2039

Patched [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py) to support controlled parity flags:
- `--n-prompts`
- `--eval-frequency`
- `--region`
- `--inflight-weight-updates`

Validation before launch:
- `./infra/pre-commit.py --fix experiments/exp_iris_rl_regression_direct_gcs_prod.py` -> `OK`
- `uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --help` showed all new flags

Launched parity run:
- root job: `/ahmed/irl-e4par-100-0325-2147`
- root launch region: `us-central1`
- TPU worker type requested for both trainer and rollout: `v5p-8`
- host RAM requested for both trainer and rollout: `400g`
- root CPU launcher resources: `cpu=0.5`, `memory=1GB`, `disk=5GB`

Exact entrypoint:
- `uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --experiment-name-suffix e4par --num-train-steps 100 --n-prompts 64 --eval-frequency 1 --region us-central1 --inflight-weight-updates`

Purpose of this run:
- closest available Iris parity check against the faster on-demand `exp2039` rollout/training shape
- changes from `r4` are intentionally limited to the two largest rollout-side deltas:
  1. `n_prompts=64` instead of `16`
  2. `inflight_weight_updates=True` instead of blocking sync mode

Expected interpretation:
- if this run speeds up materially, that supports the combined rollout-side diagnosis
- if it remains slow, then Claude's explanation was incomplete even after controlling for the largest known differences

## 2026-03-25 16:49 PT - Exact Parity Launch Command And Immediate Next Steps

Exact Iris submit command used:
```bash
uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait \
  --user ahmed \
  --job-name irl-e4par-100-0325-2147 \
  --region us-central1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py \
       --experiment-name-suffix e4par \
       --num-train-steps 100 \
       --n-prompts 64 \
       --eval-frequency 1 \
       --region us-central1 \
       --inflight-weight-updates
```

Immediate post-submit state:
- root `/ahmed/irl-e4par-100-0325-2147` -> `JOB_STATE_RUNNING`
- root task state counts: `building=1`
- `failure_count=0`
- `preemption_count=0`

Next steps for this parity run:
1. wait for train and rollout child jobs to appear, then record their concrete run ids
2. attach the existing train memory poller to the parity train task once it exists
3. compare the first `10-20` completed trainer steps against `r4` on:
   - wall-clock step cadence
   - batch prep time
   - forward/backward time
   - weight-transfer time
   - rollout `cumulative_batch_count` growth
   - eval cadence in the rollout W&B run
4. decide whether the parity changes (`n_prompts=64` + `inflight_weight_updates=True`) materially reduce the `r4` rollout-side slowdown

Decision rule:
- if parity run step cadence moves substantially toward on-demand, Claude's diagnosis was directionally right and the remaining gap can be decomposed further
- if parity run remains near `r4` speed, then the current performance explanation is still incomplete

## 2026-03-25 16:54 PT - Started Babysitter And Train Memory Poller For Parity Run

Current parity run state at monitor start:
- root `/ahmed/irl-e4par-100-0325-2147` -> `RUNNING`
- train `/ahmed/irl-e4par-100-0325-2147/rl-e4par-20260326-044831-train` -> `RUNNING`
- rollout `/ahmed/irl-e4par-100-0325-2147/rl-e4par-20260326-044831-rollout-0` -> `RUNNING`
- all three with `failure_count=0`
- train `preemption_count=0`
- rollout `preemption_count=0`

Babysitter started with existing generic Iris monitor script:
- monitor script: [babysit_iris_job.sh](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/babysit_iris_job.sh)
- monitor state: [20260325-2154_monitoring_state.json](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260325-2154_monitoring_state.json)
- monitor log: [20260325-2154_babysit.log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260325-2154_babysit.log)
- monitor PID: `87455`
- configured duration: `28800s` (`8h`)

Train memory poller attached:
- task: `/ahmed/irl-e4par-100-0325-2147/rl-e4par-20260326-044831-train/0`
- JSONL: [e4par_train_memory_agent.jsonl](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4par_train_memory_agent.jsonl)
- poller log: [e4par_train_memory_agent.log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4par_train_memory_agent.log)
- poller PID: `87453`

Monitoring intent:
- preserve one active monitor owner for the root job
- preserve a live host-RAM trace for trainer so parity-run checkpoint behavior can be compared directly with `r4`

## 2026-03-25 22:56 PDT - Patched Rollout Eval Cadence To Deduplicate On Trainer Step

Problem addressed:
- Full curriculum eval in [rollout_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rollout_worker.py) was keyed off the rollout loop's local `step`, not trainer progress.
- With the slow `r4` shape (`n_prompts=16`, `eval_frequency=1`), that caused roughly one full eval per rollout batch instead of once per completed trainer step.

Code changes:
- [run_state.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/run_state.py)
  - added monotonic trainer-step tracking on `RLRunState`
  - added `RunStateSnapshot`
  - added `get_snapshot()` and `update_train_step()`
- [train_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/train_worker.py)
  - trainer step hook now calls `_record_train_step(info.step)`
  - `_record_train_step()` updates both the local replay buffer step and shared `run_state`
- [rollout_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rollout_worker.py)
  - rollout now refreshes a cached `current_train_step` from `run_state.get_snapshot()` inside `_check_run_state()`
  - full eval is now gated by `_should_run_curriculum_eval(current_train_step, last_eval_train_step, eval_frequency)`
  - deduplicates full evals so the same completed trainer step is evaluated at most once per rollout worker
  - full eval now logs against `current_train_step`, not rollout loop step

Semantics after patch:
- Full eval runs once per observed completed trainer step that matches `train_step % eval_frequency == 0`.
- This removes the rollout-batch footgun.
- Micro-eval semantics were left unchanged; they still key off rollout progression.
- This patch only affects newly launched jobs; already-running jobs keep their in-memory rollout worker code.

Validation:
- `uv run pytest -q tests/rl/test_run_state.py tests/rl/test_train_worker.py tests/rl/test_rollout_worker.py` -> `16 passed`
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/run_state.py lib/marin/src/marin/rl/train_worker.py lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_run_state.py tests/rl/test_train_worker.py tests/rl/test_rollout_worker.py` -> `OK`

Tests added:
- [test_run_state.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/tests/rl/test_run_state.py)
  - verifies monotonic trainer-step tracking and snapshot contents
- [test_train_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/tests/rl/test_train_worker.py)
  - verifies `_record_train_step()` updates both replay buffer and shared run-state actor
- [test_rollout_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/tests/rl/test_rollout_worker.py)
  - verifies full-eval gating/dedup logic and invalid-frequency rejection

Recommended next step:
- Let the current parity run finish unchanged for evidence continuity.
- For the next launch, pick whether to keep `n_prompts=64` for throughput parity or deliberately return to `n_prompts=16` to validate that the eval-cadence fix removes the original slowdown trigger.

## 2026-03-26 00:00 PDT - Cross-Repo Notes From `vllm_tpu_multi` No-Ray Multi-Host vLLM Logbook

Source reviewed:
- [/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md](/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md)

Why this matters for the Iris RL thread:
- this external logbook captures the no-Ray / on-demand vLLM work that informed the earlier fast RL baselines
- several findings there map directly onto the current Iris throughput and lifecycle questions

### Findings that matter directly here

1. Multi-host no-Ray cross-host PP for vLLM was not supported in that stack.
- `NRM-001` failed and the explicit conclusion was:
  - no-Ray cross-host PP not supported
  - the practical alternative for 8B was independent replicas, not PP-across-host inference
- Relevance here:
  - for 8B RL sampling, the winning pattern is still "more independent sampler replicas" rather than trying to force one giant multi-host inference server

2. For 8B throughput, independent small-host replicas won decisively.
- v6e findings in that logbook:
  - `v6e-8` single host: ~`3,973 tok/s`, ~`104.4s` per 1024-completion batch, ~`5.5h` epoch projection
  - `v6e-16` as `4x` independent replicas: ~`11,912 tok/s`, ~`35.5s` per batch, ~`1.85h` epoch projection
- v5p findings in that logbook:
  - `v5p-8` single host: ~`4,749 tok/s`, ~`89.2s` per batch, ~`4.7h` epoch projection
  - `v5p-16` as `2x` independent replicas: ~`8,681 tok/s`, ~`48.9s` per batch, ~`2.6h` epoch projection
- Relevance here:
  - this strongly supports replica-parallel sampling as the right throughput lever for RL on small models
  - if Iris parity/prod runs remain sampler-bound after the current fix set, adding rollout replicas is a high-confidence next optimization

3. Concurrency matters a lot; the external sweep found a clear saturation regime.
- The logbook's concurrency sweep found throughput ceiling near `concurrency=256/replica` on v6e-16.
- Relevance here:
  - the current Iris rollout shape and batching knobs are absolutely first-order performance parameters, not secondary tuning
  - this reinforces the importance of keeping `n_prompts` / batch shape aligned with the sampler hardware envelope

4. Queue-based `LLM.generate()` HTTP serving was confirmed to be the wrong API for high-throughput serving.
- The external logbook's critical finding was:
  - queue + sync `LLM.generate()` serialized requests to effective batch-size-1
  - this caused a roughly `2x` throughput regression versus the async vLLM serving path
- It concluded the right fix for HTTP serving is `AsyncLLMEngine` / `build_app()`, not sync `LLM.generate()` behind a queue.
- Relevance here:
  - this is highly relevant context for Harbor / serving paths, but less directly relevant to the current Iris RL rollout worker because RL is not using the same queue-based HTTP serving architecture
  - still useful as a warning: avoid concluding that "vLLM itself is slow" when the real issue may be the serving interface shape

5. On-demand RL had an explicit graceful sampler shutdown fix.
- The logbook records a tri-state coordinator status (`running` / `completed` / `failed`) added to Arrow Flight coordination so samplers exit cleanly and W&B runs close when the trainer finishes.
- Relevance here:
  - this is important lineage context for why on-demand runs stopped leaving orphaned sampler runs
  - it also aligns with our general RL lifecycle concerns around sampler durability and clean shutdown semantics

6. The external on-demand RL throughput bottleneck analysis blamed trainer starvation, not trainer compute.
- Logged breakdown there:
  - forward/backward ~`45s`
  - device-to-host weight copy ~`12.6s`
  - batch prep fetch ~`15-18s`
  - batch creation ~`3.7s`
  - starvation gap `0-60s`
  - total ~`195s`
- The proposed fixes were:
  1. add a second sampler replica
  2. add background batch preparation / prefetch for streaming RL batches
  3. fix inflight weight updates in manual mode
- Relevance here:
  - this is extremely consistent with the current Iris-side throughput diagnosis:
    - trainer compute is not the main bottleneck
    - rollout-side production and trainer starvation dominate wall-clock step time
  - it also provides a concrete prioritized optimization roadmap after current stability / executor-prod questions are closed

### Interpretation for the current Iris RL investigation

This external logbook strengthens several conclusions already emerging locally:
- the current Iris throughput problem was unlikely to be a trainer compute problem
- sampler topology and batch shape matter much more than raw TPU count for 8B RL sampling
- independent rollout replicas are likely the right scaling primitive if we need more sampling throughput later
- the fastest on-demand baselines were produced by a system already thinking in terms of:
  - sampler throughput
  - starvation elimination
  - inflight updates
  - clean sampler lifecycle

### Practical carry-forward into the Iris thread

Near-term:
- keep focusing the main experiment thread on:
  1. finishing the current direct parity run
  2. then re-testing executor-backed prod (`E5`)
- do not divert immediately into large sampling-topology changes until `E5` is settled

Medium-term, if throughput is still the next bottleneck after `E5`:
1. add more rollout replicas before trying more exotic topology changes
2. consider background-prefetch / batch-prep overlap in `StreamingRolloutLoader`
3. keep `inflight_weight_updates=True` in the high-throughput configuration family

Bottom line from the external logbook:
- the old fast no-Ray / on-demand results are consistent with a world where RL throughput is sampler-bound and replica-parallelism wins
- that does not contradict the current Iris findings; it actually supports them
- the executor-prod question remains separate from these throughput lessons and should still be answered explicitly

## 2026-03-26 01:20 PDT - Plan To Remove Sampling Bottleneck In Direct Mode

User direction:
- deprioritize `E5` for now
- focus on removing the sampling bottleneck in direct mode
- compare two candidate topologies:
  1. `2 x v5p-8` samplers + `1 x v5p-8` trainer
  2. `1 x v5p-16` sampler using multi-host vLLM data parallel + `1 x v5p-8` trainer

### Recommendation

I recommend doing the `2 x v5p-8` sampler plan first.

Reasoning:
- the direct RL coordinator already supports multiple rollout workers today in [orchestration.py:176](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/orchestration.py:176)
- the current direct prod probe only hardcodes one rollout worker in [exp_iris_rl_regression_direct_gcs_prod.py:206](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py:206)
- external no-Ray/vLLM work in [/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md](/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md) strongly favored independent replicas for 8B throughput
- the `v5p-16` multihost-DP sampler path is not just a config toggle in Marin RL today; it requires new inference/runtime plumbing

### Option A: `2 x v5p-8` samplers + `1 x v5p-8` trainer

This is the practical near-term path.

Current support already exists:
- `RunConfig.num_rollout_workers` is already part of the RL config surface in [rl_job.py:47](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_job.py:47)
- experiment config also carries `num_rollout_workers` in [rl_experiment_utils.py:123](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rl_experiment_utils.py:123)
- coordinator submits one rollout child per worker index in [orchestration.py:176](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/orchestration.py:176)
- rollout storage is already file/GCS-based and naturally aggregates multiple writers

Concrete implementation plan:
1. expose `--num-rollout-workers` in [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py)
2. keep the current fast parity settings:
   - trainer `v5p-8`
   - inference `v5p-8`
   - `n_prompts=64`
   - `inflight_weight_updates=True`
   - `400g` RAM on trainer and samplers
3. fix eval ownership before launching two samplers
4. launch `2 x v5p-8` rollout workers and re-measure trainer starvation and step cadence

Non-obvious correctness issue that must be fixed:
- current full-eval dedupe is only worker-local in [rollout_worker.py:337](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rollout_worker.py:337) and [rollout_worker.py:858](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rollout_worker.py:858)
- with two rollout workers, both workers would currently run full eval once per trainer step
- simplest fix:
  - make only `worker_index == 0` run full eval and micro-eval

Expected outcome:
- trainer should stop waiting on rollout production as often
- wall-clock step time should improve further
- if this works, the next bottleneck is likely trainer-side batch preparation in [train_worker.py:115](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/train_worker.py:115), not sampler generation

If trainer is no longer sampler-starved after this:
- next optimization would be background/prefetched batch preparation in `StreamingRolloutLoader`
- not a more exotic sampler topology

### Option B: `1 x v5p-16` sampler with multi-host vLLM DP + `1 x v5p-8` trainer

This is a separate feature spike, not the next direct experiment.

Why:
- current RL vLLM config in [vllm.py:84](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:84) only exposes TP-style settings
- current async/inflight wrapper in [worker.py:181](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py:181) builds a single async engine with TP settings only
- there is no Marin-RL-facing support today for:
  - `data_parallel_size`
  - `distributed_executor_backend`
  - external launcher / multihost rank setup

External evidence from the vLLM multihost repo:
- generic vLLM does have a DP surface, e.g. `data_parallel_size` plus `distributed_executor_backend=\"external_launcher\"` in [/Users/ahmed/code/vllm_tpu_multi/vllm/examples/offline_inference/torchrun_dp_example.py:96](/Users/ahmed/code/vllm_tpu_multi/vllm/examples/offline_inference/torchrun_dp_example.py:96)
- but the same repo explicitly warns that generic multihost support does not automatically imply polished TPU-plugin behavior in [/Users/ahmed/code/vllm_tpu_multi/CODEX_VLLM_MULTIHOST.md:64](/Users/ahmed/code/vllm_tpu_multi/CODEX_VLLM_MULTIHOST.md:64)

What Option B would actually require:
1. extend RL inference config to expose multihost DP knobs
2. add a new RL inference runtime path for multihost vLLM DP
3. validate standalone bring-up on Iris outside RL first
4. only then wire it into the rollout worker
5. then validate generation, weight sync, retries, and shutdown

My judgment:
- this is real engineering work, not an experiment-only change
- for an 8B model, it is also not obviously the right topology even if implemented
- the external evidence still points toward independent replicas as the better throughput scaling primitive

### Decision

Recommended order:
1. patch direct prod probe to support `num_rollout_workers`
2. patch eval ownership so only one rollout worker evaluates
3. launch `2 x v5p-8` samplers + `1 x v5p-8` trainer
4. only if that is still sampler-bound, consider background batch prep
5. treat the `v5p-16` multihost-DP sampler as a separate R&D thread after the simpler replica path

Bottom line:
- `2 x v5p-8` samplers is the next serious experiment
- `1 x v5p-16` multihost-DP sampler is not ruled out, but it is not the shortest path to removing the direct-mode sampling bottleneck

## 2026-03-26 01:24 PDT - MS1 Launch Prep: Direct Two-Sampler Run

Current state before launch:
- existing parity run is still live:
  - root `/ahmed/irl-e4par-100-0325-2147`
  - train `rl-e4par-20260326-044831-train`
  - rollout `rl-e4par-20260326-044831-rollout-0`
- user requested that the thread move on to the multi-sampler direction and stay under active babysitting for at least 8 hours

Implementation completed before launch:
1. direct prod experiment now accepts `--num-rollout-workers` in [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py)
2. rollout eval is now single-owner:
   - only `worker_index == 0` runs micro-eval / full eval
   - this avoids duplicated evaluation work when multiple sampler jobs are active
3. targeted validation passed:
   - `uv run pytest -q tests/rl/test_rollout_worker.py` -> `13 passed`
   - `./infra/pre-commit.py --fix experiments/exp_iris_rl_regression_direct_gcs_prod.py lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_rollout_worker.py` -> `OK`

Planned launch command (`MS1`):

```bash
uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py \
  --experiment-name-suffix e4ms2 \
  --num-train-steps 100 \
  --n-prompts 64 \
  --eval-frequency 1 \
  --region us-central1 \
  --inflight-weight-updates \
  --num-rollout-workers 2
```

Operational decision:
- first try to submit `MS1` without killing the live parity run
- if TPU capacity blocks `MS1`, stop the older parity run and retry
- once `MS1` is live, hand it to an 8-hour Iris babysitter with exact resubmit metadata and keep the train memory poller attached

Operational correction immediately after this plan:
- running `uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py ...` directly from the shell was incorrect for this thread
- that path fell back to Fray `LocalClient` and attempted to run the coordinator on the laptop
- visible symptom:
  - `fray.v2.client current_client: using LocalClient (fallback)`
  - followed by local rollout failures (`ModuleNotFoundError: No module named 'tpu_inference'`)
- this was not an Iris cluster failure and not a multi-sampler design failure; it was the wrong launch entrypoint
- the known-good pattern remains the Iris root-job submit wrapper used for the parity run:
  - `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait ... -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py ...`

## 2026-03-26 01:31 PDT - MS1 Correctly Launched Under Iris And Handed To Long-Running Monitor

Corrected launch command actually used:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait \
  --user ahmed \
  --job-name irl-e4ms2-100-0326-0125 \
  --region us-central1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py \
       --experiment-name-suffix e4ms2 \
       --num-train-steps 100 \
       --n-prompts 64 \
       --eval-frequency 1 \
       --region us-central1 \
       --inflight-weight-updates \
       --num-rollout-workers 2
```

Submitted root job:
- `/ahmed/irl-e4ms2-100-0326-0125`

Subtree created by the root coordinator:
- trainer:
  - `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-train`
- rollout workers:
  - `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-rollout-0`
  - `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-rollout-1`

Launch chronology:
1. first corrected Iris submit succeeded immediately and created the root job
2. initial scheduler state had:
   - trainer `PENDING`
   - rollout-0 `PENDING`
   - rollout-1 `PENDING`
   with:
   - `Insufficient memory (need 400.0GB, available 304.8GB)`
   - `Insufficient TPUs`
   - autoscaler backoff on `tpu_v5p_8-us-central1-a`
3. per the pre-recorded operational plan, the older single-sampler parity run was stopped to free TPU capacity:
   - stopped root:
     - `/ahmed/irl-e4par-100-0325-2147`
   - which also terminated:
     - `rl-e4par-20260326-044831-train`
     - `rl-e4par-20260326-044831-rollout-0`
4. after freeing that capacity, `MS1` converged to:
   - root `RUNNING`
   - train `RUNNING`
   - rollout-0 `RUNNING`
   - rollout-1 `RUNNING`

Current state at log time:
- `/ahmed/irl-e4ms2-100-0326-0125` -> `JOB_STATE_RUNNING`
- `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-train` -> `JOB_STATE_RUNNING`
- `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-rollout-0` -> `JOB_STATE_RUNNING`
- `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-rollout-1` -> `JOB_STATE_RUNNING`
- all visible jobs currently show:
  - `failure_count=0`
  - `preemption_count=0`

Train-memory polling:
- attached task:
  - `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-train/0`
- first live samples:
  - `2026-03-26T08:31:38.832222+00:00`
    - `attempt=1`
    - `mem_mb=34713`
    - `peak_mb=34723`
  - `2026-03-26T08:31:43.967062+00:00`
    - `attempt=1`
    - `mem_mb=34713`
    - `peak_mb=34723`

Long-running monitor ownership:
- Iris babysitter PTY session:
  - `43809`
- train-memory poller PTY session:
  - `85990`
- state file:
  - [20260326-0129_monitoring_state.json](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260326-0129_monitoring_state.json)
- monitor log:
  - [20260326-0129_babysit.log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260326-0129_babysit.log)
- train-memory poller log:
  - [e4ms2_train_memory_agent.log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4ms2_train_memory_agent.log)
- train-memory JSONL:
  - [e4ms2_train_memory_agent.jsonl](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4ms2_train_memory_agent.jsonl)

Monitoring contract:
- babysitter duration set to `28800s` (8 hours)
- root job will be resubmitted with the exact same `MS1` command if it hits a terminal non-success state
- this is now the single active monitor owner for the multi-sampler run

## 2026-03-26 05:13 PDT - MS1 Two-Sampler 100-Step Run Finished Cleanly; Extending To 200 Steps

Final controller state for `MS1`:
- root `/ahmed/irl-e4ms2-100-0326-0125` -> `JOB_STATE_SUCCEEDED`
- train `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-train` -> `JOB_STATE_SUCCEEDED`
- rollout-0 `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-rollout-0` -> `JOB_STATE_SUCCEEDED`
- rollout-1 `/ahmed/irl-e4ms2-100-0326-0125/rl-e4ms2-20260326-082712-rollout-1` -> `JOB_STATE_SUCCEEDED`
- all visible jobs finished with:
  - `failure_count=0`
  - `preemption_count=0`

Important interpretation:
- the two-sampler direct shape completed the full 100-step run without retries
- this is the strongest direct evidence so far that:
  - the multi-sampler topology is operationally stable on Iris
  - the sampler bottleneck can be attacked without destabilizing the run

Immediate next step requested by user:
- keep the same topology and continue to a longer run:
  - `2 x v5p-8` rollout workers
  - `1 x v5p-8` trainer
  - `400g` RAM on trainer and rollout
  - `n_prompts=64`
  - `inflight_weight_updates=True`
  - `200` train steps

Planned continuation command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait \
  --user ahmed \
  --job-name irl-e4ms2-200-<timestamp> \
  --region us-central1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py \
       --experiment-name-suffix e4ms2 \
       --num-train-steps 200 \
       --n-prompts 64 \
       --eval-frequency 1 \
       --region us-central1 \
       --inflight-weight-updates \
       --num-rollout-workers 2
```

## 2026-03-26 05:17 PDT - Launched MS2 200-Step Two-Sampler Continuation And Reattached 8-Hour Monitoring

Executed continuation command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait \
  --user ahmed \
  --job-name irl-e4ms2-200-0326-0514 \
  --region us-central1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py \
       --experiment-name-suffix e4ms2 \
       --num-train-steps 200 \
       --n-prompts 64 \
       --eval-frequency 1 \
       --region us-central1 \
       --inflight-weight-updates \
       --num-rollout-workers 2
```

New root job:
- `/ahmed/irl-e4ms2-200-0326-0514`

Immediate post-launch state:
- root job entered `JOB_STATE_RUNNING`
- no child train/rollout jobs were visible yet at first controller check
- no failures or preemptions recorded on the root job

Current monitoring ownership for the 200-step continuation:
- babysitter PTY session:
  - `1018`
- auto-attach train-memory poller PTY session:
  - `62393`
- state file:
  - [20260326-0515_monitoring_state.json](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260326-0515_monitoring_state.json)
- monitor log:
  - [20260326-0515_babysit.log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260326-0515_babysit.log)
- auto-attach helper:
  - [attach_e4ms2_200_poller.sh](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/attach_e4ms2_200_poller.sh)
- intended train-memory JSONL once the train child appears:
  - [e4ms2_200_train_memory_agent.jsonl](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/e4ms2_200_train_memory_agent.jsonl)

Notes:
- this extends the exact `MS1` topology that already completed 100 steps cleanly
- no resubmission/retry has happened for `MS2`
- the same 8-hour babysitting contract is active for the longer run

## 2026-03-26 09:17 PDT - W&B Links For Two-Sampler Direct Run (`e4ms2-20260326-121919`)

Separate reference section for the direct RL run with:
- `1 x v5p-8` trainer
- `2 x v5p-8` sampler/rollout workers
- larger rollout batch shape (`n_prompts=64`)
- `inflight_weight_updates=True`

W&B runs:
- trainer:
  - https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-train?nw=nwuserahmedah
- rollout worker 0:
  - https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-0
- rollout worker 1:
  - https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-1

## 2026-03-26 09:22 PDT - Exact Throughput Calculation For `r4` vs `e4par` vs `e4ms2`

Recomputed directly from raw W&B trainer histories, not from dashboard eyeballing.

Method:
- use one row per logged `global_step`
- compute wall-clock per step as delta of W&B `_runtime`
- compute trainer compute time from `throughput/step_duration_seconds`
- derive trainer idle/wait as:
  - `wall_clock_per_step - step_duration`

Runs compared:
- `r4`:
  - `e4p-20260325-172718-train`
- single-sampler parity:
  - `e4par-20260326-044831-train`
- two-sampler direct:
  - `e4ms2-20260326-121919-train`

Median metrics:
- `r4`
  - wall-clock/step: `496.1041092910018s` = `8.2684 min`
  - step duration: `61.00027213749854s`
  - derived idle: `435.10383715350326s`
  - batch prep: `30.71908414363861s`
- `e4par`
  - wall-clock/step: `182.20604731799995s` = `3.0368 min`
  - step duration: `60.76345235000008s`
  - derived idle: `121.44259496799987s`
  - batch prep: `30.078736543655396s`
- `e4ms2`
  - wall-clock/step: `101.08752536799875s` = `1.6848 min`
  - step duration: `61.082273318999796s`
  - derived idle: `40.005252048998955s`
  - batch prep: `3.959851026535034s`

Exact speedup calculations:
- total speedup, `r4 -> e4ms2`:
  - `496.1041092910018 / 101.08752536799875 = 4.907675673486169x`
- parity speedup, `r4 -> e4par`:
  - `496.1041092910018 / 182.20604731799995 = 2.722684872137125x`
- two-sampler speedup on top of parity, `e4par -> e4ms2`:
  - `182.20604731799995 / 101.08752536799875 = 1.8024590165453354x`

Interpretation:
- the total improvement is real and large:
  - about `4.9x`
- the two-sampler configuration materially improves throughput on top of the single-sampler parity run:
  - about `1.8x`
- trainer compute is basically unchanged across all three runs:
  - about `61s` median
- the gain is therefore almost entirely from reduced trainer waiting
  - idle dropped:
    - `435.1s -> 121.4s -> 40.0s`
- the most visible train-side symptom of the second sampler is batch prep collapse:
  - `30.7s -> 30.1s -> 4.0s`

Important caveat:
- `e4ms2` was still in progress when these medians were computed
- the exact final medians may move slightly by run completion
- but the direction and magnitude of the speedup are already clear

## 2026-03-26 09:33 PDT - Bottleneck Analysis For `e4ms2` Trainer And Remaining Headroom

Analyzed from:
- trainer W&B history:
  - `e4ms2-20260326-121919-train`
- rollout W&B histories:
  - `e4ms2-20260326-121919-rollout-0`
  - `e4ms2-20260326-121919-rollout-1`
- live trainer logs in:
  - [20260326-0515_babysit.log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260326-0515_babysit.log)

Trainer-side medians:
- wall-clock/step:
  - `101.08752536799875s`
- trainer step duration:
  - `61.08199151200097s`
- hook time:
  - `15.379230644997733s`
- batch prep:
  - `3.957648992538452s`
- MFU:
  - `90.26134442184804%`

Critical interpretation:
- the trainer TPU is already highly utilized during the actual train step
  - MFU is about `90%`
- the trainer is therefore compute-efficient when it is busy
- the remaining end-to-end loss is mostly not “slow math”

More precise decomposition:
- steady-state floor implied by trainer metrics:
  - `step_duration + hook_time ~= 61.08s + 15.38s = 76.46s`
- current observed median wall-clock/step:
  - `101.09s`
- median unexplained residual after compute + hooks:
  - about `4.03s`
- but this residual has a fat tail:
  - `38.5%` of steps have residual `>10s`
  - `20.0%` of steps have residual `>40s`
  - `13.1%` of steps have residual `>60s`

Meaning:
- on median steps, the trainer is close to saturated
- across the full run, the pipeline is not fully saturated because the tail is still large

Evidence that rollout supply is still a bottleneck in the tail:
- rollout-0 cumulative batches at comparable runtime:
  - `77`
- rollout-1 cumulative batches at comparable runtime:
  - `110`
- rollout-0 average seconds per completed batch over run:
  - `213.55s`
- rollout-1 average seconds per completed batch over run:
  - `149.12s`

Interpretation:
- rollout-0 is materially slower because it is carrying the eval burden
- with only one sampler doing eval, this is much better than before, but it still creates imbalance
- if rollout-0 were freed from eval and matched rollout-1 throughput, combined sampler supply would be about:
  - `74.56s` per rollout batch
- that is almost exactly the current trainer floor of about `76.46s`

So the most important conclusion is:
- `2` samplers are close, but not fully sufficient in the current setup because one of them is partially consumed by eval
- `2` pure rollout samplers with eval moved off the critical path would likely be enough to saturate the trainer

Checkpoint overhead is the other major remaining bottleneck:
- checkpoint saves show up repeatedly in live logs at steps:
  - `83`, `89`, `95`, `101`, `107`, `114`, `120`, `126`
- these coincide with many of the worst wall-clock spikes
- examples:
  - step `89`: batch prep `56.21s`, fwd/bwd `4.77s`
  - step `101`: batch prep `45.20s`, fwd/bwd `36.88s`
  - step `107`: batch prep `50.70s`, fwd/bwd `10.33s`
- wall-clock p90 is:
  - `205.76s`
- wall-clock max is:
  - `273.77s`

Practical recommendation order:
1. Move eval off the critical sampler path.
   - best version: dedicate a separate eval worker / eval process
   - cheaper version: reduce eval frequency further
   - expected effect: make the existing 2-sampler topology much closer to trainer saturation
2. Reduce checkpoint overhead if throughput matters more than rapid recovery.
   - easiest knob: checkpoint less frequently
   - stronger change: make checkpoint commit more asynchronous
3. Only after that, consider a third sampler.
   - a third rollout worker would likely saturate the trainer even with current eval placement
   - but the upside is bounded by the trainer floor

Estimated remaining headroom:
- current median wall-clock/step:
  - `101.09s`
- current trainer floor:
  - `76.46s`
- best-case additional speedup from sampler-side saturation alone is therefore only about:
  - `101.09 / 76.46 = 1.322x`

Neutral conclusion:
- no, the pipeline has not hit its absolute theoretical limit
- yes, it is close to the trainer-side floor on typical steps
- the next gains are now incremental, not another 4-5x jump
- the cheapest remaining win is probably:
  - move eval off the critical sampler path
- the next biggest throughput win after that is probably:
  - reduce checkpoint frequency / checkpoint cost

## 2026-03-26 09:46 PDT - Replaced Fake Micro-Eval Disable With Explicit `None`

Problem:
- multiple experiments were “disabling” micro-eval with:
  - `micro_eval_frequency=9999999`
- this is a footgun because the config reads as enabled, just with a very large cadence

Change:
- `CurriculumConfig.micro_eval_frequency` now accepts:
  - `int | None`
- explicit semantics:
  - `None` means micro-eval is disabled
  - positive integer means enabled every N rollout-worker steps
- rollout gating now goes through `_should_run_micro_eval(...)`
  - this returns `False` when `micro_eval_frequency is None`
  - and rejects nonpositive frequencies when micro-eval is enabled

Files changed:
- [curriculum.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/curriculum.py)
- [rollout_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/rollout_worker.py)
- [test_rollout_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/tests/rl/test_rollout_worker.py)

Experiment callsites updated from fake disable to explicit disable:
- [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py)
- [exp_iris_rl_regression_executor_gcs_small.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_executor_gcs_small.py)
- [exp_iris_rl_direct.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_direct.py)
- [exp_iris_rl_debug.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_debug.py)
- [exp_iris_rl_regression_direct_hf_small.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_hf_small.py)
- [exp_iris_rl_regression_direct_gcs_small.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_small.py)
- [exp_iris_rl_oom_isolation.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_oom_isolation.py)
- [exp2039_rl_math500.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp2039_rl_math500.py)

Validation:
- `uv run pytest -q tests/rl/test_rollout_worker.py` -> `20 passed`
- `./infra/pre-commit.py --fix ...` on touched files -> `OK`

## 2026-03-26 10:10 PDT - Quantified Rollout Overproduction / Discard In The 2-Sampler Run

Question:
- with `2` samplers, `n_prompts=64`, and group size `16`, how much rollout data are we actually throwing away?

Relevant config for the current `e4ms2-20260326-121919` run:
- each rollout batch contains:
  - `64 * 16 = 1024` completions
- trainer batch size is:
  - `1024`
- replay buffer policy in [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py):
  - `max_samples=1`
  - `max_rollout_step_delay=1`
  - `capacity=4096`

Replay semantics that matter:
- trainer does **not** consume whole producer batches serially
- replay buffer samples `1024` **individual** rollouts from the combined pool
- each rollout can be used at most once (`max_samples=1`)
- rollouts can survive for at most one trainer-step of lag (`max_rollout_step_delay=1`)

Observed W&B counters at measurement time:
- trainer `global_step` rows:
  - `154` completed steps (`0..153`)
- rollout-0 cumulative rollout batches:
  - `91`
- rollout-1 cumulative rollout batches:
  - `129`
- total rollout batches produced:
  - `220`

Produced vs consumed:
- trainer consumption:
  - `154 * 1024 = 157,696` rollouts
- sampler production:
  - `220 * 1024 = 225,280` rollouts
- excess produced over consumed:
  - `225,280 - 157,696 = 67,584` rollouts
  - equivalently:
    - `66` extra full minibatches

Bound on actual discarded data:
- replay capacity is only:
  - `4096` rollouts
  - `4` full minibatches
- so the true discarded amount is bounded by:
  - minimum discarded:
    - `67,584 - 4,096 = 63,488`
  - maximum discarded:
    - `67,584`

Interpretation:
- at this point in the run, we have thrown away roughly:
  - `63k-68k` rollouts
- that is about:
  - `28%-30%` of produced rollout data
- this is the concrete cost of using extra sampler capacity to keep the trainer much closer to saturation

Important nuance:
- this is not “waste caused by a bug in replay”
- it is the expected consequence of:
  - overproducing relative to one trainer step
  - allowing only `1` use per rollout
  - allowing only `1` step of rollout staleness

Practical implication:
- the current 2-sampler topology is buying speed with real inference overproduction
- if TPU budget matters more than absolute throughput, the next tuning knob is probably:
  - reduce per-sampler rollout batch size
- if absolute throughput matters more, the current behavior is acceptable and expected

## 2026-03-26 11:55 PDT - Added Explicit vLLM KV-Cache Metrics Flag

Change:
- added `kv_cache_metrics: bool = False` to `vLLMInferenceContextConfig`
- threaded that flag through both sync `LLM(...)` and async/inflight `AsyncEngineArgs(...)`
- added `--kv-cache-metrics` to the direct prod experiment launcher

Files changed:
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`
- `experiments/exp_iris_rl_regression_direct_gcs_prod.py`
- `tests/rl/test_inference_ctx.py`
- `tests/rl/test_rollout_worker.py`

Validation:
- `uv run pytest -q tests/rl/test_inference_ctx.py tests/rl/test_rollout_worker.py` -> `33 passed`
- `./infra/pre-commit.py --fix ...` on touched files -> `OK`

Important caveat:
- this change only passes the vLLM observability flag through to rollout-worker engine creation
- it does not by itself add a custom W&B exporter for those metrics
- new launches can now opt in with `--kv-cache-metrics`, after which we can inspect whether the underlying vLLM runtime surfaces the cache metrics in logs or metric endpoints in a way that is useful to us

## 2026-03-26 12:04 PDT - DP-vs-TP Critique and Validation Plan

Question under discussion:
- should rollout inference stay on `TP=4` per sampler, or is there a better packing / sharing configuration such as DP-style replicas on the same `v5p-8` host?

What is actually established:
- current sampler HBM is dominated by reserved KV cache, not weights
- archived rollout logs show:
  - model-loaded HBM per chip: about `3.74 GiB`
  - KV init HBM per chip: about `86.17 / 95.74 GiB`
  - `GPU KV cache size: 2,700,928 tokens`
- current live sampler W&B system metrics show:
  - median HBM per chip: about `86.36 GiB`
  - max HBM per chip: about `90.77 GiB`
- from the KV init shape, reserved KV cache is about:
  - `82.43 GiB/chip`

Critique of the TP-vs-DP discussion:
- the claim that `DP=4` does not fit under the current cache reservation is likely correct
- the stronger claim that `TP=4` is therefore forced is too strong
- the more plausible alternative is not pure `DP=4`; it is `TP=2 / DP=2` style packing on one `v5p-8`

Memory arithmetic:
- current `TP=4` sampler:
  - weights/chip: about `3.74 GiB`
  - median observed total/chip: about `86.36 GiB`
  - peak observed total/chip: about `90.77 GiB`
- projected `DP=4` on the same host:
  - add about `11.22 GiB/chip` for full weights instead of TP-sharded weights
  - projected median: about `97.58 GiB/chip`
  - projected peak: about `101.99 GiB/chip`
  - conclusion: does not fit under the current cache budget
- projected `TP=2 / DP=2`:
  - add about `3.74 GiB/chip` relative to current TP=4
  - projected median: about `90.10 GiB/chip`
  - projected peak: about `94.51 GiB/chip`
  - conclusion: might fit, but with very little headroom

Important caveat:
- this is still only memory arithmetic
- it does not prove throughput wins or losses
- external vLLM TPU work in `/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md` suggests small independent replicas can beat one bigger TP server for throughput on 8B, but that is not yet an apples-to-apples proof for this exact `v5p-8` RL sampler setup

Role of `kv_cache_metrics`:
- the new `kv_cache_metrics` flag is useful for answering whether current cache reservation is oversized and whether lower-cache packing is safe
- it is not by itself enough to tell us whether DP-style packing is faster than TP
- to answer throughput cleanly, we still need an A/B benchmark
- to answer memory cleanly, we need the cache metrics to actually be surfaced somewhere we can inspect, plus prompt-token counts on rollout batches

Plan:
1. Launch one rollout-enabled run with `--kv-cache-metrics`
2. Verify whether vLLM actually exposes usable cache metrics in logs or metrics endpoints in our embedded rollout-worker setup
3. Add minimal rollout logging for prompt-token counts so live working-set size is bounded more tightly than response-only lower bounds
4. Use those metrics to estimate whether KV cache can be reduced enough for `TP=2 / DP=2` packing on one `v5p-8`
5. If the memory envelope looks viable, run an explicit A/B:
   - baseline: current `TP=4` sampler layout
   - candidate: `TP=2 / DP=2` style packing
6. Compare:
   - rollout batch time
   - trainer batch-prep time
   - end-to-end train step time
   - HBM headroom and stability

Decision rule:
- if `TP=2 / DP=2` fits comfortably and improves end-to-end step time, it is the better inference topology
- if it fits but hurts throughput, stay on `TP=4`
- if it does not fit with enough safety margin, stop and keep `TP=4`

## 2026-03-26 12:12 PDT - Recommended Pure-Inference Experiment Ladder

Conclusion:
- for the TP-vs-packed-DP question, the next experiments should be pure inference first, not full RL
- reason: RL end-to-end step time is materially noisier than sampler throughput because it mixes in trainer hooks, checkpoints, replay timing, and eval cadence
- current e4ms2 evidence:
  - trainer wall-clock median: about `101.3s`
  - trainer wall-clock p90: about `207.0s`
  - trainer wall-clock CV: about `0.42`
  - by contrast, rollout-storage write time is only about `0.31s` median per batch and is negligible next to `163-228s` rollout batch times

Recommended experiment ladder:

### INF-001: TP=4 baseline, current sampler shape
Purpose:
- establish the real single-engine inference baseline under current RL-like prompt/response shape

Config:
- `TP=4`
- current `max_model_len=2048`
- current batch shape equivalent to RL sampler load
- enable `kv_cache_metrics`
- log prompt-token counts and response-token counts
- run `20-30` rollout-equivalent batches after warmup

Measure:
- batch wall time
- tokens/sec
- HBM per chip
- KV cache metrics
- prompt tokens, response tokens, total live-sequence pressure

### INF-002: TP=4 cache-budget sweep
Purpose:
- determine whether current KV cache reservation is oversized and how much it can be reduced before throughput degrades

Config sweep:
- hold topology fixed at `TP=4`
- vary `gpu_memory_utilization`, e.g. `0.90`, `0.80`, `0.70`
- optionally lower `max_model_len` only if the cache metrics imply large unused slack at 2048

Measure:
- same metrics as INF-001
- especially whether throughput stays flat while cache reservation falls

Decision point:
- if reducing cache budget does not hurt throughput much, that is evidence current TP=4 reservation is oversized and packed layouts become more plausible

### INF-003: TP=2 single-engine feasibility run
Purpose:
- measure the actual cost of moving from TP=4 to TP=2 before attempting any co-location

Config:
- one engine only
- `TP=2`
- same prompt/response shape
- `kv_cache_metrics` enabled

Measure:
- batch wall time
- tokens/sec
- HBM per chip
- cache behavior

Decision point:
- if single-engine TP=2 already looks bad on throughput or HBM margin, stop; there is no reason to attempt packed TP=2x2

### INF-004: Packed TP=2 / DP=2-style co-location
Purpose:
- test the real hypothesis: two smaller independent inference replicas on one `v5p-8` host may beat one TP=4 engine

Config:
- two independent `TP=2` engines on one `v5p-8`
- same total request load as INF-001
- route requests across both replicas

Measure:
- aggregate tokens/sec
- per-replica tokens/sec
- HBM headroom per chip
- cache behavior per replica
- stability / OOM margin

Decision point:
- if aggregate throughput improves and HBM remains safe, packed TP=2x2 is the better inference topology
- if not, keep TP=4

### INF-005: Short RL validation only after an inference winner exists
Purpose:
- confirm the inference winner actually improves trainer feed rate in the real loop

Config:
- short RL run only, about `30-40` train steps
- push checkpoints out of the measurement window
- minimize eval noise

Measure:
- trainer batch-prep time
- train step wall time
- end-to-end throughput

Recommended order:
1. INF-001
2. INF-002
3. INF-003
4. INF-004 only if INF-003 is promising
5. INF-005 only after an inference winner exists

Smallest high-information run:
- `20-30` rollout-equivalent batches per inference condition is enough for the first pass
- do not start with a 100-200 step RL run for this question

## 2026-03-26 - Imported Claude pure-inference benchmark results

Source:
- `.agents/logbooks/iris-rl-claude.md`

Purpose:
- carry the newer Claude-side inference benchmarking results into this Codex logbook without duplicating the earlier eval-cadence and `e4par`/`e4ms2` RL findings that were already recorded here

### INF-001: TP=4 pure-inference baseline completed

Artifacts:
- Iris root job:
  - `/ahmed/inf-001-tp4-0326-r4`
- child job:
  - `/ahmed/inf-001-tp4-0326-r4/inf-001-inf001-tp4-20260326-195258`
- W&B run:
  - `inf001-tp4-20260326-195258`
- script:
  - `experiments/exp_inf_001_tp4_baseline.py`

Config:
- model:
  - `meta-llama/Llama-3.1-8B-Instruct`
- TPU:
  - `v5p-8`
- topology:
  - `TP=4`
- `max_model_len=2048`
- `gpu_memory_utilization=0.90`
- batch shape:
  - `64` prompts x `16` generations = `1024` completions per batch
- batches:
  - `25` total, `3` warmup + `22` measured

Results:
- median output throughput:
  - `3,345 tok/s`
- mean output throughput:
  - `3,340 tok/s`
- p10-p90 throughput:
  - `3,292-3,405 tok/s`
- median batch time:
  - `108.4s`
- HBM after model load:
  - `86.2 GiB/chip`
- throughput CV:
  - about `3.4%`

Interpretation:
- pure inference matches the `e4ms2` rollout-worker throughput almost exactly, so rollout-side grading/serialization overhead is not the dominant limit
- the `TP=4` baseline leaves only about `9 GiB/chip` headroom at `gpu_memory_utilization=0.90`

### INF-002: TP=4 cache-budget sweep completed

Purpose:
- test whether the current KV cache reservation is oversized by sweeping `gpu_memory_utilization` while holding the rest of the sampler shape fixed

Sweep results:

| `gpu_memory_utilization` | HBM per chip (GiB) | Median tok/s | Median batch time (s) |
|---|---|---|---|
| `0.90` | `86.2` | `3,345` | `108.4` |
| `0.80` | `76.6` | `3,381` | `107.4` |
| `0.70` | `67.0` | `3,357` | `110.4` |
| `0.60` | `57.4` | `3,385` | `107.0` |

Important preserved engine details from Claude log:
- all runs used vLLM `0.13.2.post6`, V1 engine, `dtype=bfloat16`, `TP=4`, `max_seq_len=2048`, `enable_prefix_caching=True`, `enable_chunked_prefill=True`, `enforce_eager=True`, `kv_cache_metrics=True`
- KV cache capacity by budget:
  - `0.90` -> `2,700,928` tokens
  - `0.80` -> `2,387,200` tokens
  - `0.70` -> `2,073,472` tokens
  - `0.60` -> `1,759,744` tokens

Key findings:
- throughput stayed flat across the entire `0.90 -> 0.60` sweep
  - spread was under about `1.2%`
- dropping from `0.90` to `0.60` freed about `28.8 GiB/chip` with no measured throughput penalty
- the KV cache at `0.90` was materially oversized for this RL-like prompt/response shape
- HBM is therefore not the blocker for packed `TP=2` experiments

Decision imported from Claude:
- proceed to `INF-003`
- for feasibility work, use `gpu_memory_utilization=0.70` or `0.60` only if extra headroom is useful operationally; throughput evidence did not require the larger cache budget

### INF-003: TP=2 single-engine feasibility launched, with an important correction

Jobs launched:
- `INF-003a`
  - `TP=2`
  - `gpu_memory_utilization=0.90`
  - root job: `/ahmed/inf-003-tp2-0326`
- `INF-003b`
  - `TP=2`
  - `gpu_memory_utilization=0.45`
  - root job: `/ahmed/inf-003-tp2-gpu45-0326`

Important correction from Claude:
- the original rationale for `INF-003b` was wrong
- packed `TP=2x2` on one `v5p-8` does not split chip memory in half per engine
- the intended packed design uses disjoint chip partitions via `TPU_VISIBLE_CHIPS`
  - engine 1 on chips `0,1`
  - engine 2 on chips `2,3`
- each engine therefore still gets full per-chip HBM on its own 2-chip slice
- so the real packed-topology comparison is:
  - `2 * throughput(INF-003a)` vs `INF-001`
- `INF-003b` is still useful only as additional TP=2 cache-sensitivity data

Status at Claude handoff:
- `INF-003` results were still pending in the Claude logbook

### INF-004 direction imported from Claude

Proposed packed benchmark:
- one `v5p-8` host
- two subprocesses, each running independent `TP=2` inference
- process layout:
  - subprocess 1: `TPU_VISIBLE_CHIPS=0,1`
  - subprocess 2: `TPU_VISIBLE_CHIPS=2,3`
- use offline `LLM.generate()` rather than `vllm serve` so the benchmark stays aligned with the RL rollout-worker API

Decision gate:
- if aggregate `2 x TP=2` throughput beats the `INF-001` `TP=4` baseline (`3,345 tok/s`) and remains stable, packed `TP=2x2` wins
- otherwise keep `TP=4`

### Benchmark observability improvement carried over

Claude also noted that the pure-inference benchmark scripts now log KV-cache init details to W&B under `kv_cache/`, including:
- `kv_cache/num_gpu_blocks`
- `kv_cache/block_size`
- `kv_cache/total_tokens`
- `kv_cache/max_concurrency_2048`
- `kv_cache/gpu_memory_utilization`

Operational value:
- this removes dependence on ephemeral Iris logs for future cache-sizing comparisons

## 2026-03-26 - INF-003a live readout and immediate next step

Source:
- W&B run:
  - `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/inf003-tp2-20260326-225800`

Status at read time:
- run state:
  - `running`
- final `summary/*` fields were not populated yet because the run had not finished
- however, the run already had `16` measured batches in W&B history, which is enough to answer the launch-readiness question for `INF-004a`

Measured `INF-003a` throughput from live history:
- measured batches observed:
  - `16`
- median output throughput:
  - `2985.3 tok/s`
- mean output throughput:
  - `2973.6 tok/s`
- p10-p90 throughput:
  - `2880.6 - 3040.7 tok/s`
- median batch time:
  - `122.4s`
- mean batch time:
  - `124.2s`
- p10-p90 batch time:
  - `113.6 - 135.4s`

Interpretation:
- single-engine `TP=2` is clearly viable
- it is materially slower than the `TP=4` baseline per engine, but not remotely bad enough to stop the packed experiment
- using the live median as the planning number:
  - projected packed throughput for `2 x TP=2` is about:
    - `2 * 2985.3 = 5970.6 tok/s`
- compare against `INF-001` TP=4 baseline:
  - `3345 tok/s`
- projected packed gain:
  - about `1.78x`

Decision:
- we have enough information to launch `INF-004a`
- no need to wait for `INF-003a` to fully finish just to decide whether the packed benchmark is justified

Immediate next steps:
1. Launch `INF-004a` using the new script:
   - `experiments/exp_inf_004_tp2x2_packed.py`
2. Keep the benchmark apples-to-apples with `INF-001` / `INF-003a`:
   - total prompts `64`
   - generations `16`
   - `gpu_memory_utilization=0.90`
   - `TP=2` per replica
   - chip groups:
     - `0,1`
     - `2,3`
3. Evaluate packed success on:
   - aggregate makespan throughput
   - aggregate index-aligned throughput
   - per-replica stability
   - HBM headroom
   - absence of cross-process TPU initialization/pathology
4. If `INF-004a` lands near the projected `~6.0k tok/s` and remains stable, the next benchmark should be a short RL confirmation run rather than another pure-inference topology sweep

## 2026-03-26 - INF-004a packed `TP=2x2` completed successfully

Launch command:
- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name inf-004a-tp2x2-0326-1646 --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_inf_004_tp2x2_packed.py --experiment-name-suffix inf004-tp2x2 --n-prompts 64 --n-generations 16 --num-batches 25 --warmup-batches 3 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --chip-groups '0,1;2,3' --region us-central1`

Artifacts:
- root job:
  - `/ahmed/inf-004a-tp2x2-0326-1646`
- child benchmark job:
  - `/ahmed/inf-004a-tp2x2-0326-1646/inf-004-inf004-tp2x2-20260326-234814`
- W&B run:
  - `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/inf004-tp2x2-20260326-234814`

Final status:
- root job:
  - `JOB_STATE_SUCCEEDED`
- child job:
  - `JOB_STATE_SUCCEEDED`
- child `failure_count`:
  - `0`
- child `preemption_count`:
  - `0`
- babysitting outcome:
  - no code fixes, restarts, or TPU recovery actions were needed

Packed benchmark results from W&B summary:
- measured batches:
  - `22`
- aggregate makespan throughput:
  - `5672.2 tok/s`
- aggregate index-aligned median throughput:
  - `5462.1 tok/s`
- aggregate index-aligned p10-p90:
  - `5109.4 - 5619.3 tok/s`
- aggregate makespan measured elapsed time:
  - `1430.2s`
- aggregate total measured output tokens:
  - `8,112,425`
- worker 0 median throughput:
  - `2875.2 tok/s`
- worker 1 median throughput:
  - `2840.6 tok/s`
- worker 0 median batch time:
  - `65.8s`
- worker 1 median batch time:
  - `64.7s`
- per-worker HBM after init:
  - `86.16 GiB/chip`
- per-worker KV cache capacity:
  - `1,289,088` tokens

Observed end-of-run summaries from job logs:
- worker 0:
  - `2875.2 tok/s` median (`2751.1-2947.8` p10-p90), `65.8s/batch` median
- worker 1:
  - `2840.6 tok/s` median (`2726.3-2941.2` p10-p90), `64.7s/batch` median
- packed aggregate:
  - makespan `5672.2 tok/s`
  - index-aligned median `5462.1 tok/s`

Comparison against earlier baselines:
- vs `INF-001` `TP=4` baseline (`3345 tok/s` median):
  - packed makespan throughput is about `1.70x`
  - packed index-aligned median is about `1.63x`
- vs the `INF-003a` live planning projection (`~5970.6 tok/s` from `2 * 2985.3`):
  - realized packed makespan throughput reached about `95%` of the simple linear projection

Interpretation:
- packed `2 x TP=2` on one `v5p-8` clearly beats single-engine `TP=4` for this RL-like offline inference shape
- cross-process interference exists but is modest; aggregate makespan throughput remained close to the sum of the two worker medians
- the benchmark ran cleanly end-to-end, so the main question is no longer inference feasibility
- for pure inference on this workload, packed `TP=2x2` is the winning topology

Updated next step:
- stop spending cycles on more topology-only pure-inference experiments unless a new deployment constraint appears
- move to a short RL confirmation run that uses the winning packed topology, then compare wall-clock step time and sampler/trainer overlap against `e4ms2`

## 2026-03-26 - Overall next steps after `INF-004a`

Bottom line:
- `INF-004a` is strong evidence that the rollout side can probably be reduced from:
  - `2 x v5p-8` rollout workers
  - to `1 x v5p-8` rollout host running packed `2 x TP=2`
- but this is only proven for pure inference so far
- it does **not** yet prove that the full RL system can replace the `e4ms2` sampler topology with one packed rollout job at the same trainer-facing cadence

Current best overall plan:
1. run a short RL confirmation that preserves the `e4ms2` trainer/curriculum/replay/eval shape as closely as possible while targeting the reduced rollout footprint
2. compare that run directly against `e4ms2` on:
   - wall-clock per train step
   - inter-step idle time
   - `batch_prep`
   - rollout throughput
   - eval interference
   - IS ratio / freshness behavior
3. if the proxy run still leaves a meaningful gap to `e4ms2`, implement true packed rollout support inside the RL path
   - one rollout Iris job
   - two independent `TP=2` vLLM engines
   - chip groups `0,1` and `2,3`
4. once that packed RL path exists, rerun the same short confirmation and then promote:
   - `1 x v5p-8` trainer
   - `1 x v5p-8` packed rollout
   as the new default Llama 3.1 8B Iris RL shape

Separate experiment file created for this next stage:
- `experiments/exp_iris_rl_regression_direct_gcs_packed_candidate.py`

What that file does:
- preserves the known-good `e4ms2` RL envelope:
  - `train_batch_size=1024`
  - `n_prompts=64`
  - `n_generations_per_prompt=16`
  - `eval_frequency=1`
  - `inflight_weight_updates=True`
  - replay buffer `max_rollout_step_delay=1`
  - `1 x v5p-8` trainer
- but fixes the currently launchable rollout footprint to:
  - `1 x v5p-8` rollout worker
- and records the true packed rollout target in the experiment metadata/logging:
  - `2 x TP=2`
  - chip groups `0,1` and `2,3`

Important caveat recorded here:
- the current RL orchestration/config surface still cannot express:
  - two independent vLLM engines inside one rollout job
  - each with its own `TPU_VISIBLE_CHIPS`
- so the new file is a **packed-rollout proxy**, not the final packed rollout implementation

## 2026-03-26 - Implementation plan for RL-side packed rollout support

Goal:
- replace the current `e4ms2` rollout topology:
  - `2 x v5p-8` rollout workers
- with one rollout job on one `v5p-8` that internally runs:
  - replica 0: `TP=2` on chips `0,1`
  - replica 1: `TP=2` on chips `2,3`
- while keeping the trainer-facing RL behavior as close to `e4ms2` as possible

What is easy vs hard:
- easy:
  - split prompt batches across two replicas
  - merge completions back in original prompt order
  - keep the parent rollout worker API unchanged for environments
- hard:
  - weight sync
- reason:
  - `INF-004a` proved the working packed topology uses separate subprocesses with `TPU_VISIBLE_CHIPS` set before importing JAX/vLLM
  - that means we cannot just instantiate two engines in the current rollout-worker process
  - and we should **not** try to serialize the full 8B state dict from the parent rollout worker into two child subprocesses every step

Main design decision:
- packed rollout should use persistent inference subprocesses, one per replica
- each packed child process should own:
  - its own vLLM engine
  - its own `WeightTransferClient`
  - its own weight-sync loop / current weight-step tracking
- the parent rollout worker should act as:
  - curriculum + eval coordinator
  - prompt sharder / completion merger
  - rollout writer / metric logger

Concrete implementation plan:

### Phase 1 - Add a real packed vLLM config type

Create a separate config type rather than another boolean flag.

Suggested shape:

```python
@dataclass(frozen=True)
class PackedvLLMReplicaConfig:
    visible_chips: str
    tensor_parallel_size: int


@dataclass(frozen=True)
class PackedvLLMInferenceContextConfig:
    model_name: str
    canonical_model_name: str | None
    max_model_len: int
    gpu_memory_utilization: float
    replicas: list[PackedvLLMReplicaConfig]
    sampling_params: VLLMSamplingConfig
    load_format: str = "auto"
    enforce_eager: bool = True
    kv_cache_metrics: bool = False
```

Code touchpoints:
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
- `lib/marin/src/marin/rl/rollout_worker.py`
- `lib/marin/src/marin/rl/rl_job.py`

Required dispatch updates:
- extend the inference-config unions to include the packed config
- update `create_inference_context(...)` to construct a packed inference context when this config is provided

### Phase 2 - Build a packed inference subprocess worker

Create a new module for the persistent child process entrypoint, using the exact env pattern that worked in `INF-004a`:
- set before import:
  - `TPU_PROCESS_BOUNDS`
  - `TPU_CHIPS_PER_PROCESS_BOUNDS`
  - `TPU_VISIBLE_CHIPS`
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`

The child process should:
- build one `vLLMInferenceContext`
- open one `WeightTransferClient`
- wait for first weights
- serve RPC-style commands from the parent:
  - `generate`
  - `sync_once` or `current_weight_step`
  - `metrics`
  - `shutdown`

Likely file:
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_worker.py`

### Phase 3 - Implement `PackedvLLMInferenceContext`

Parent responsibilities:
- spawn one child per replica
- own the request routing / result gathering
- split prompts by prompt index
- preserve original ordering when merging completions
- aggregate per-replica metrics

Important behavioral contract:
- `batch_completions(...)` must still return the same `list[ChatCompletion]` shape expected by `MathEnv.sample(...)`
- environments should not need to know that the context is packed

Likely file:
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`

Split/merge rule:
- partition the prompt list by contiguous slices for stable ordering
- assign slice `0` to chips `0,1`
- assign slice `1` to chips `2,3`
- merge child results back by original prompt position

### Phase 4 - Refactor weight sync for packed mode

This is the real blocker.

Current code path:
- parent rollout worker owns one `_transfer_client`
- parent calls `receive_weights(...)`
- parent passes the resulting state dict into `_policy_ctx.reload_model(...)`

Why that does not scale to packed mode:
- the state dict is too large to relay into two subprocesses every sync step
- the packed subprocesses need to fetch weights directly

Packed-mode change:
- when using `PackedvLLMInferenceContextConfig`, the packed context should own weight-sync orchestration
- the parent rollout worker should no longer create the normal `_transfer_client` in this mode
- instead it should ask the packed context for:
  - `wait_for_initial_weights()`
  - `sync_weights_if_needed()`
  - `current_weight_step`
  - packed weight-transfer metrics

Pragmatic implementation note:
- this can be done as a packed-specific branch in `RolloutWorker` first
- it does not need a repo-wide abstraction cleanup in the first pass

### Phase 5 - Keep RL semantics unchanged above the inference layer

Things that should remain unchanged:
- curriculum lesson sampling
- eval and micro-eval gating
- rollout metadata construction
- replay buffer behavior
- rollout writing
- trainer code

Important observation:
- eval currently runs only when `worker_index == 0`
- packed rollout will still have exactly one rollout worker job
- so eval semantics should stay unchanged automatically if the parent rollout worker remains the evaluation owner

### Phase 6 - Wire the real packed config into the experiment launcher

After the packed context exists, update:
- `experiments/exp_iris_rl_regression_direct_gcs_packed_candidate.py`

so it becomes a real packed rollout launch rather than a proxy:
- one rollout worker
- packed replicas:
  - `0,1`
  - `2,3`
- `TP=2` per replica
- same `e4ms2` trainer/curriculum/replay/eval shape:
  - `train_batch_size=1024`
  - `n_prompts=64`
  - `n_generations_per_prompt=16`
  - `eval_frequency=1`
  - `inflight_weight_updates=True`

### Phase 7 - Test plan

Add tests at three levels:

1. config + dispatch tests
- packed config selects the packed inference context
- invalid chip-group / TP combinations fail fast

2. split/merge correctness tests
- a fake packed context should prove:
  - prompts are partitioned correctly
  - merged completions preserve original order
  - per-prompt completion multiplicity stays correct

3. rollout-worker integration tests
- packed rollout path still logs and writes batches correctly
- eval still triggers only once per completed trainer step
- packed metrics are surfaced without crashing the existing logging path

Likely test files:
- `tests/rl/test_inference_ctx.py`
- `tests/rl/test_rollout_worker.py`

### Phase 8 - Validation ladder

`PKR-001` startup smoke:
- `5` train steps
- `n_prompts=8` or `16`
- validate:
  - both packed child replicas start
  - both receive weights
  - training advances
  - no TPU init/pathology

`PKR-002` short parity confirmation:
- `30-40` train steps
- production-like shape:
  - `n_prompts=64`
  - `16` generations
- compare against:
  - `e4par`
  - `e4ms2`

Primary success metrics:
- wall-clock/step
- trainer inter-step idle
- `batch_prep`
- rollout throughput
- eval behavior
- stability / zero retries

Success criterion for promotion:
- packed RL should land close enough to `e4ms2` that replacing:
  - `2 x v5p-8` rollout workers
with:
  - `1 x v5p-8` packed rollout
is operationally worthwhile

Recommended implementation order:
1. packed config + subprocess worker
2. packed inference context split/merge
3. packed-specific weight-sync path in rollout worker
4. tests
5. `PKR-001`
6. `PKR-002`

## 2026-03-26 - Revised packed rollout plan: full inflight parity only

This supersedes the earlier branching plan.

Decision:
- do **not** build a non-inflight packed mode
- packed rollout must preserve the `e4ms2` asynchronous weight-update behavior closely enough to replace it, not just approximate the topology
- the earlier packed-rollout proxy is no longer the main next step; the next real step is the packed RL implementation itself

### Hard requirements

Packed rollout is only considered successful if all of these are true:

1. one rollout Iris job runs on one `v5p-8`
2. that rollout job owns two persistent inference replicas:
   - replica 0: `TP=2` on chips `0,1`
   - replica 1: `TP=2` on chips `2,3`
3. `inflight_weight_updates=True` remains enabled
4. the trainer does not block on rollout-side weight sync
5. every logical rollout batch and eval batch still has one unambiguous `weight_step` in RL metadata
6. any single child failure kills the whole packed rollout worker and lets Iris retry the rollout job

### Revised architecture

Parent rollout worker responsibilities:
- curriculum interaction
- eval scheduling
- prompt construction
- prompt sharding across replicas
- merge completions back into original prompt order
- rollout writing
- aggregate metrics logging
- supervising child process health

Packed child replica responsibilities:
- set TPU env before JAX/vLLM import
- create one async/inflight-capable vLLM engine
- create one `WeightTransferClient`
- poll for new weights continuously in the background
- stage fetched weights as pending
- serve generate requests from the parent

Important design rule:
- use fresh subprocesses launched with `subprocess.Popen(..., exec)` semantics
- do **not** use Python multiprocessing/fork inheritance for packed replicas
- do **not** try to push the 8B state dict from parent to children every sync step

### Stronger inflight synchronization model

The previous plan was underspecified here. This is the actual model we should implement.

Each packed child tracks:
- `active_weight_id`
- `pending_weight_id`
- `busy` flag
- latest packed-health heartbeat / metrics

Weight updates flow like this:
1. child polls Arrow Flight in the background
2. when a newer weight arrives, child decodes it and stores it as `pending`
3. child does **not** immediately swap `pending -> active` while a request is running
4. parent coordinates activation so both replicas transition to the same weight id between requests

Generation correctness rule:
- a logical RL batch must be generated under one shared dispatch weight id `W`
- parent may only dispatch batch shards once both replicas report:
  - idle
  - healthy
  - same `active_weight_id == W`

Activation rule:
- if replicas have staged the same newer `pending_weight_id == W_next`, parent sends an explicit activation command to both replicas while both are idle
- only after both acknowledge activation does parent allow the next batch to start

Why this is the right compromise:
- keeps asynchronous background fetch/decode overlap
- avoids trainer-blocking main-thread weight sync
- preserves one clean batch weight id for RL metadata
- is stricter than today's single-worker inflight semantics, but necessary once one logical batch is split across two local replicas

### Explicit protocol to implement

Parent <-> child protocol should not stay vague. Minimum RPC surface:

- `status() -> {healthy, busy, active_weight_id, pending_weight_id, metrics}`
- `activate_pending(expected_weight_id) -> ack`
- `generate(request_id, prompts, sampling_params, expected_weight_id) -> completions`
- `shutdown()`

Child-side generate contract:
- reject generation if `active_weight_id != expected_weight_id`
- mark replica `busy=True` for the full request lifetime
- only allow `activate_pending(...)` when idle

Parent-side dispatch contract:
1. poll both replica statuses
2. if both idle and both have the same newer pending weight id, activate that weight on both
3. choose `dispatch_weight_id = common active weight id`
4. shard prompts by contiguous slices
5. send one `generate(...)` request to each replica with that exact `dispatch_weight_id`
6. merge completions in original prompt order
7. write rollout batch metadata using `dispatch_weight_id`

### Concrete code changes

1. Add a packed inflight config type

Use a separate config type, not a boolean:

```python
@dataclass(frozen=True)
class PackedvLLMReplicaConfig:
    visible_chips: str
    tensor_parallel_size: int


@dataclass(frozen=True)
class PackedAsyncvLLMInferenceContextConfig:
    model_name: str
    canonical_model_name: str | None
    max_model_len: int
    gpu_memory_utilization: float
    replicas: list[PackedvLLMReplicaConfig]
    sampling_params: VLLMSamplingConfig
    load_format: str = "auto"
    enforce_eager: bool = True
    kv_cache_metrics: bool = False
```

2. Add a packed child runtime module

Suggested new file:
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_worker.py`

Responsibilities:
- env setup (`TPU_VISIBLE_CHIPS`, etc.)
- async vLLM engine bring-up
- local weight-transfer polling thread
- pending/active weight state machine
- request loop

3. Add a parent packed inference context

Suggested new file:
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`

Responsibilities:
- launch children
- supervise health
- activation orchestration
- prompt sharding / merge
- aggregate metrics
- shutdown on any child fault

4. Update inference-context dispatch

Files:
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
- `lib/marin/src/marin/rl/rollout_worker.py`
- `lib/marin/src/marin/rl/rl_job.py`

Required changes:
- extend config unions
- teach `create_inference_context(...)` to build packed mode
- packed mode should expose the same `BaseInferenceContext.batch_completions(...)` contract to environments

5. Update rollout worker for packed-mode ownership

Packed mode should bypass the current single-context transfer path:
- parent rollout worker should **not** own the ordinary `_transfer_client`
- parent rollout worker should obtain packed metrics and current batch weight ids from the packed context
- packed context becomes the weight-sync owner in this mode

Files:
- `lib/marin/src/marin/rl/rollout_worker.py`

### Things the plan was previously missing

These are now explicitly in-scope:

- **Batch-level weight consistency**
  - one logical RL batch cannot mix replica weight ids
- **Child failure semantics**
  - any child crash, stuck request, or health-check failure kills both children and crashes the rollout worker
- **No orphan processes**
  - shutdown path must terminate children even on parent exception
- **Request-level atomicity**
  - a child may fetch/decode future weights while busy, but may not change `active_weight_id` during an in-flight request
- **Eval correctness**
  - full eval and micro-eval must use the same packed dispatch path and the same weight-id rules as training batches
- **Metric shape**
  - log:
    - packed aggregate throughput
    - per-replica throughput
    - per-replica active/pending weight ids
    - activation lag / skew

### Test plan, upgraded

1. Packed child state-machine tests
- pending weights do not activate while busy
- activation only succeeds when idle
- generation rejects wrong expected weight id

2. Parent packed-context tests
- prompt shards preserve ordering on merge
- batch dispatch only occurs when replicas share the same active weight id
- parent activates matching pending weights on both replicas before next batch

3. Rollout-worker integration tests
- packed mode still produces valid `RolloutBatch`
- metadata `weight_step` equals dispatch weight id
- eval still fires exactly once per completed trainer step
- one child failure propagates as rollout-worker failure

Primary test files:
- `tests/rl/test_inference_ctx.py`
- `tests/rl/test_rollout_worker.py`

### Validation ladder, upgraded

`PKR-001` packed startup smoke
- `5` train steps
- small prompt count
- must prove:
  - both children start
  - both children receive and stage weights
  - at least one synchronized activation happens
  - at least one packed rollout batch is written

`PKR-002` packed inflight correctness run
- `10-20` train steps
- inspect:
  - child active/pending weight traces
  - dispatch weight ids
  - absence of mixed-weight batches

`PKR-003` packed parity run
- `30-40` train steps
- production-like shape:
  - `n_prompts=64`
  - `16` generations
  - `eval_frequency=1`
  - `inflight_weight_updates=True`
- compare against `e4ms2`:
  - wall-clock/step
  - trainer idle
  - `batch_prep`
  - rollout throughput
  - eval behavior
  - stability / retries

Promotion criterion:
- packed RL should be close enough to `e4ms2` that replacing:
  - `2 x v5p-8` rollout workers
with:
  - `1 x v5p-8` packed rollout
is the new default operational recommendation

### Revised implementation order

1. packed inflight config type
2. packed child runtime with active/pending weight state machine
3. parent packed inference context with synchronized activation protocol
4. packed-mode rollout-worker branch
5. tests for child state machine, parent dispatch, and rollout integration
6. `PKR-001`
7. `PKR-002`
8. `PKR-003`

## 2026-03-26 - Packed rollout implementation landed locally

### What changed

Implemented the first real RL-side packed rollout path for Iris:

- added `PackedvLLMInferenceContextConfig`
- added parent packed inference context:
  - `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
- added child worker process + IPC protocol:
  - `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_worker.py`
  - `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_protocol.py`
- added shared remote-metadata staging helper:
  - `lib/marin/src/marin/rl/environments/inference_ctx/staging.py`
- wired packed config into exports and RL job / rollout worker dispatch:
  - `lib/marin/src/marin/rl/environments/inference_ctx/__init__.py`
  - `lib/marin/src/marin/rl/environments/inference_ctx/base.py`
  - `lib/marin/src/marin/rl/rl_job.py`
  - `lib/marin/src/marin/rl/rollout_worker.py`

### Implementation shape

Current design is the planned full-inflight-parity shape:

- one rollout Iris job still owns the environment, curriculum, rollout writing, and eval logic
- inside that rollout job, the packed parent context spawns `2` fresh subprocesses
- each subprocess sets:
  - `TPU_VISIBLE_CHIPS`
  - `TPU_PROCESS_BOUNDS`
  - `TPU_CHIPS_PER_PROCESS_BOUNDS`
- each child owns:
  - one `AsyncvLLMInferenceContext`
  - one weight-transfer client
  - background inflight weight polling
  - `active_weight_id` / `pending_weight_id`
- the parent:
  - polls child status
  - only dispatches a logical batch at one shared weight id
  - activates matching pending weights on both replicas before dispatch
  - shards prompts contiguously and merges completions back in original order
  - treats any child RPC/process failure as fatal

Important rollout-worker change:

- packed mode now owns weight sync in the inference context
- the rollout worker skips the ordinary parent `_transfer_client` in that mode
- rollout metadata `weight_step` is taken from the packed context’s dispatch weight id

### Launcher

Added the real packed RL experiment launcher:

- `experiments/exp_iris_rl_regression_direct_gcs_packed.py`

Shape:

- `1 x v5p-8` trainer
- `1 x v5p-8` rollout
- rollout packed as:
  - replica `0`: chips `0,1`, `TP=2`
  - replica `1`: chips `2,3`, `TP=2`
- `n_prompts=64`
- `16` generations / prompt
- `eval_frequency=1`
- `inflight_weight_updates=True`

### Local validation

Formatting / typecheck:

- `./infra/pre-commit.py --fix ...` on all touched files: passed

Focused unit coverage:

- `uv run --package marin pytest tests/rl/test_rollout_worker.py tests/rl/test_packed_vllm_inference_ctx.py tests/rl/test_inference_ctx.py`
- result: `40 passed`

Broader RL package coverage with default repo markers:

- `uv run --package marin pytest tests/rl`
- result: `144 passed, 2 skipped, 12 deselected`

Slow-inclusive local integration pass:

- `uv run --package marin pytest tests/rl -m 'not tpu_ci'`
- result: **not clean locally**
- failure mode was timeout in existing RL integration tests waiting for initial rollouts / local threaded harness progress
- this did **not** surface a packed-rollout unit failure; the new packed path is not covered by those local levanter integration cases yet

### Next action

Run the real Iris smoke:

- `PKR-001`
- command target:
  - `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-packed-pkr001-<timestamp> --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_gcs_packed.py --experiment-name-suffix e4pk2 --num-train-steps 5 --n-prompts 64 --eval-frequency 1 --region us-central1`

## 2026-03-26 19:05 PDT - `PKR-001` packed RL smoke succeeded, but throughput landed near `e4par`, not `e4ms2`

### Launch / artifacts

- root job:
  - `/ahmed/irl-e4pk2-5-20260326-182126`
- child jobs:
  - `/ahmed/irl-e4pk2-5-20260326-182126/rl-pkr001-20260327-012712-train`
  - `/ahmed/irl-e4pk2-5-20260326-182126/rl-pkr001-20260327-012712-rollout-0`
- W&B:
  - trainer: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr001-20260327-012712-train
  - rollout: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr001-20260327-012712-rollout-0

### Final status

- root, trainer, and rollout all ended `JOB_STATE_SUCCEEDED`
- `exit_code=0`
- `failure_count=0`
- `preemption_count=0`

This was a real end-to-end RL success for the new packed rollout path:

- one trainer `v5p-8`
- one rollout `v5p-8`
- rollout pod running `2 x TP=2` local packed replicas
- inflight weight updates active

### What worked

- packed rollout was functionally correct in RL:
  - both local replicas started, fetched weights, activated weights, and generated batches
  - both replicas ended on `active_weight_id=4`
  - both replicas reported `6` successful weight fetches / activations
  - both replicas reported `0` failed receives
- rollout storage writes were cheap:
  - typical serialize + GCS write cost was about `0.21-0.28s`
- trainer-side compute was healthy once data existed:
  - steady-state `throughput/step_duration_seconds` was about `61s`
  - steady-state `throughput/tokens_per_second` was about `34.4k`
  - steady-state `throughput/mfu` was about `90%`

### What did not work

This did **not** match `e4ms2` throughput.

The short-smoke wall-clock picture was:

- trainer waited `275s` for initial rollouts
- trainer progress reached `5/5` at about `18m10s`
- that is about `218s/step` over the 5-step train loop
- excluding the startup-heavy first step, the remaining 4 steps were about `171s/step`

That is much slower than:

- `e4ms2`: about `101s/step`
- and only modestly better than the older single-rollout parity regime (`e4par`: about `182s/step`)

So packed RL currently lands much closer to `e4par` class than `e4ms2` class.

### Why it missed

The bottleneck was **not** trainer math. The bottleneck was rollout-side availability.

Evidence:

- trainer-side compute stayed near `61s/step`
- but trainer `batch_prep` totals after startup were:
  - step 2: `60.854s`
  - step 3: `39.208s`
  - step 4: `31.538s`
- those delays were mostly fetch wait, not batch construction:
  - step 2 fetch: `57.645s`
  - step 3 fetch: `36.126s`
  - step 4 fetch: `28.320s`

Rollout logs explain the wait:

- the single packed rollout worker was doing both:
  - train rollout generation (`32 + 32` prompt shards)
  - eval generation (`250 + 250` prompt shards)
- eval used both local packed replicas on the same worker
- that means the packed rollout topology reproduced sampler throughput, but it did **not** reproduce the scheduling benefit of two independent rollout workers

Concretely:

- in `e4ms2`, eval interference is diluted across two rollout jobs
- in packed `PKR-001`, one rollout job still owns all rollout and eval work
- so eval continues to contaminate trainer-facing cadence

### Secondary notes

- rollout W&B ended with:
  - `inference.packed/replica_count = 2`
  - `inference.packed/replica_0/total_generate_requests = 15`
  - `inference.packed/replica_1/total_generate_requests = 15`
  - `inference.rollout_storage/cumulative_batch_count = 10`
- rollout W&B also emitted a minor step-order warning around step `-1`
  - this did not affect job success
- the TPU scoped-vmem `ragged_paged_attention` messages observed during startup were compile-time warnings, not fatal errors

### Conclusion

`PKR-001` is a **correctness success** and a **throughput miss**.

It proves:

- one `v5p-8` can host the packed sampler path correctly in real RL

It does **not** prove:

- that one packed rollout worker can replace the `e4ms2` two-rollout topology at the same trainer-facing cadence

Current best interpretation:

- packed rollout fixed the hardware-topology problem
- the remaining throughput problem is eval scheduling / ownership inside the RL stack

### Next action

The next discriminating experiment should isolate pure rollout parity from eval contamination:

- rerun packed RL with eval disabled or made sparse
- compare resulting wall-clock / `batch_prep` directly against `e4ms2`
- if packed cadence moves toward `~100-120s/step`, the remaining blocker is eval scheduling, not packed inference

## 2026-03-26 19:28 PDT - Historical postmortem on older `e4par` and `e4p` runs

I checked the older W&B links that looked like unexplained deaths and wrote down the actual Iris-side cause so we do not misclassify them later.

### `e4par-20260326-044831` did not crash; it was manually terminated

- W&B:
  - trainer: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4par-20260326-044831-train
  - rollout: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4par-20260326-044831-rollout-0
- Iris subtree:
  - root: `/ahmed/irl-e4par-100-0325-2147`
  - trainer: `/ahmed/irl-e4par-100-0325-2147/rl-e4par-20260326-044831-train`
  - rollout: `/ahmed/irl-e4par-100-0325-2147/rl-e4par-20260326-044831-rollout-0`
- Root, trainer, and rollout all ended `JOB_STATE_KILLED` with `error="Terminated by user"`.
- `failure_count=0` and `preemption_count=0` on the killed tasks.
- The run was still healthy when it was stopped:
  - trainer had already completed step `70`
  - rollout had just finished eval at step `70` and started the next generation wave
  - there was no traceback or runtime failure signal in the last logs

Conclusion:

- `e4par-20260326-044831` was a coordinated manual stop, not an Iris failure, OOM, or rollout crash
- W&B showing an unclean terminal state is just the result of external termination before a graceful final flush

### `e4p-20260325-130820` failed on the trainer, not the rollout

- W&B:
  - trainer: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4p-20260325-130820-train
  - rollout: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4p-20260325-130820-rollout-0
- Iris lineage:
  - root: `/ahmed/irl-e4p-100r3-0324-2245`
  - trainer child: `/ahmed/irl-e4p-100r3-0324-2245/rl-e4p-20260325-130820-train`
  - rollout child: `/ahmed/irl-e4p-100r3-0324-2245/rl-e4p-20260325-130820-rollout-0`
- The trainer child exhausted its retry budget and Iris killed the subtree.
- Archived bug report evidence for the trainer child showed:
  - `failure_count=2`
  - `preemption_count=1`
  - attempt `0`: TPU worker timeout / worker failure
  - attempt `1`: exit `137` OOM kill shortly after replay-buffer activity
  - attempt `2`: exit `137` OOM kill during the JAX array-serialization / commit path
- The rollout child itself looked healthy up to termination:
  - W&B rollout summary ended around `_runtime ~ 13215s`
  - `_step=20`
  - `inference.rollout_storage/cumulative_batch_count=115`
  - `inference.throughput/tokens_per_second ~3300`
  - `inference.eval/math_full/avg_at_1=0.47`
- Once the trainer child exceeded max failures, Iris killed the rollout child as collateral and the root later failed because the `130820` trainer child was no longer present.

Conclusion:

- `e4p-20260325-130820` was a trainer-instability failure, not a rollout failure
- the relevant failure modes were worker timeout plus repeated trainer OOMs
- the rollout W&B run looking superficially healthy is expected because rollout was not the first component to fail

## 2026-03-26 23:18 PDT - Concrete plan for packed `v5p-8`: split train across both replicas, pin eval to one replica

Terminology for this plan:

- one Iris rollout job owns one packed rollout worker
- that packed rollout worker owns two local vLLM replicas on the same `v5p-8`
- each local replica is `TP=2`
- use `replica 0` and `replica 1` for the local vLLM instances

Goal:

- preserve packed rollout for train sampling by using both local replicas when possible
- route eval onto only one local replica so eval no longer consumes all packed sampler capacity
- keep `LLM.generate()` / in-process vLLM hot weight reloads; do **not** switch to `vllm serve`

Important caveat:

- routing eval to one replica is not sufficient by itself
- today eval runs inline in the rollout loop, so the rollout worker still blocks while eval is running
- to get trainer-facing benefit, we need both:
  - replica-aware request routing in the packed inference context
  - background eval execution in the rollout worker so train sampling can continue on the other replica

### Desired behavior

- `train` request, no eval active:
  - shard prompts across `replica 0` and `replica 1`
- `eval` or `micro_eval` request:
  - send all prompts only to `replica 1`
- `train` request while eval is active on `replica 1`:
  - send all train prompts only to `replica 0`
- no request should ever dispatch to a replica that is already reserved by another request

This will make one packed rollout worker behave much more like two local samplers:

- `replica 0`: train-priority path
- `replica 1`: shared train/eval path, but eval-preferred when an eval is scheduled

### Code changes

1. Extend the inference API with request kind.

- file: `lib/marin/src/marin/rl/environments/inference_ctx/base.py`
- add `request_kind: Literal["train", "eval", "micro_eval"]` to `batch_completions(...)`
- update all inference-context implementations to accept the new parameter
- default behavior for non-packed contexts can ignore the value after type acceptance

2. Pass mode through environments.

- file: `lib/marin/src/marin/rl/environments/math_env.py`
- `MathEnv.sample(...)` already has `mode`
- forward that mode into `inference_ctx.batch_completions(..., request_kind=mode)`
- preserve existing `mode in ("train", "eval")` validation

3. Add packed routing policy to the packed inference context.

- file: `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
- add a small routing helper that maps request kind to preferred replica set:
  - `train` -> `{0,1}` when both free, else `{0}`
  - `eval` -> `{1}`
  - `micro_eval` -> `{1}`
- reject invalid states explicitly rather than silently falling back
- keep prompt ordering stable by splitting only across the chosen subset and then merging back into original index order

4. Add per-replica reservation / concurrency control.

- file: `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
- current implementation assumes one packed batch at a time across all replicas
- replace that with per-replica reservation state:
  - lock for `replica 0`
  - lock for `replica 1`
  - request-scoped reservation objects
- requirements:
  - train and eval may run concurrently if they target disjoint replicas
  - two evals may not overlap on `replica 1`
  - train may use both replicas only when both are available
  - train must degrade cleanly to `replica 0` only when `replica 1` is reserved by eval

5. Make dispatch-weight resolution subset-aware.

- file: `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
- today `_resolve_dispatch_weight(...)` checks both replicas together
- change it to resolve activation for a chosen replica subset
- rules:
  - train on `{0,1}` requires a shared dispatch weight across both replicas
  - eval on `{1}` requires only `replica 1` to be activated to the chosen dispatch weight
  - train metadata continues to use the train request's dispatch weight only
- keep full inflight parity:
  - child replicas still own their own weight-transfer clients
  - no non-inflight fallback

6. Move eval off the synchronous rollout path.

- file: `lib/marin/src/marin/rl/rollout_worker.py`
- current `_evaluate_lesson(...)` / `_evaluate_curriculum(...)` run inline in the main rollout loop
- add a dedicated background executor or thread for eval work
- behavior:
  - when eval is triggered, submit eval work asynchronously
  - main rollout loop continues generating train batches immediately
  - if another eval trigger arrives while one is still running, coalesce or skip the duplicate instead of queuing an unbounded backlog
- this is the key scheduling fix; without it, `replica 1` isolation helps less than it should

7. Track train and eval dispatch state separately in metrics.

- files:
  - `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
  - `lib/marin/src/marin/rl/rollout_worker.py`
- add packed metrics for:
  - `packed/replica_0/busy`
  - `packed/replica_1/busy`
  - `packed/eval_active`
  - `packed/last_train_dispatch_weight_id`
  - `packed/last_eval_dispatch_weight_id`
  - per-replica generate counts split by request kind if easy
- this is needed so the next RL run can prove whether eval is still contaminating train throughput

8. Keep failure handling strict.

- files:
  - `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
  - `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_worker.py`
- if either child replica dies, hangs, or returns a weight-consistency error:
  - kill the packed inference context
  - crash the rollout worker
  - let Iris retry the rollout job
- do not add a degraded one-replica survival mode in v1

### Test plan

1. Routing unit tests.

- file: `tests/rl/test_packed_vllm_inference_ctx.py`
- cases:
  - `train` with no eval active uses both replicas
  - `eval` uses only `replica 1`
  - `micro_eval` uses only `replica 1`
  - `train` while eval is active uses only `replica 0`
  - merged outputs preserve original prompt ordering

2. Weight-resolution unit tests.

- file: `tests/rl/test_packed_vllm_inference_ctx.py`
- cases:
  - train subset `{0,1}` requires shared dispatch weight
  - eval subset `{1}` resolves independently
  - stale / mismatched status causes deterministic failure, not silent serving

3. Rollout-worker integration tests.

- files:
  - `tests/rl/test_rollout_worker.py`
- cases:
  - eval submission does not block the main train rollout loop
  - duplicate eval triggers coalesce while one eval job is already running
  - packed metrics expose eval-active and per-replica busy state

### Validation ladder

1. Local tests.

- `uv run --package marin pytest tests/rl/test_packed_vllm_inference_ctx.py`
- `uv run --package marin pytest tests/rl/test_rollout_worker.py`
- `./infra/pre-commit.py --fix` on touched files

2. Short Iris smoke.

- rerun a 5-step packed RL smoke with the new eval-routing behavior
- success criteria:
  - job still succeeds cleanly
  - train rollouts continue while eval is active
  - metrics show eval bound to `replica 1`

3. Throughput confirmation run.

- run packed RL with the same shape as `PKR-001`
- compare against prior `PKR-001`:
  - overall step time
  - trainer `batch_prep`
  - rollout-side generate counts per replica
- if cadence improves materially toward `~100-120s/step`, then eval scheduling was the main remaining bottleneck

### Expected outcome

Best realistic outcome:

- one packed rollout `v5p-8` becomes much closer to an `e4ms2` replacement because eval no longer monopolizes both local replicas

What this still does **not** promise:

- exact `e4ms2` parity under every eval regime
- during eval windows, train still temporarily loses access to `replica 1`
- the point of this change is to preserve one train-serving replica during eval, not to make eval free

## 2026-03-26 23:31 PDT - Eval-isolated packed rollout implementation landed locally

I implemented the concrete packed `v5p-8` follow-up from the section above.

### What changed

- added explicit inference request kinds (`train`, `eval`, `micro_eval`) to the RL inference-context API
- threaded request kind through RL environments so packed routing can distinguish train from eval while still using the eval dataset for both full eval and micro-eval
- updated packed vLLM parent routing so:
  - `train` uses both local replicas when eval is idle
  - `eval` and `micro_eval` pin to `replica 1`
  - `train` falls back to `replica 0` while eval is active or waiting on `replica 1`
- made packed dispatch-weight resolution subset-aware instead of globally assuming both replicas serve every request
- added parent-side packed metrics for:
  - `packed/eval_active`
  - per-replica reservation flags
  - `packed/last_train_dispatch_weight_id`
  - `packed/last_eval_dispatch_weight_id`
- added child-side per-request-kind generate counters:
  - `total_train_generate_requests`
  - `total_eval_generate_requests`
  - `total_micro_eval_generate_requests`
- reworked rollout-worker eval execution:
  - packed rollout now computes eval work on a background executor
  - main train rollout loop continues running while async eval is in flight
  - duplicate eval triggers are coalesced instead of building an unbounded queue
  - eval computation happens off-thread, but tracker / curriculum side effects are consumed on the main thread after the future completes

### Local validation

- `uv run python -m py_compile ...` on all touched RL / packed files: passed
- focused tests:
  - `uv run --package marin pytest tests/rl/test_packed_vllm_inference_ctx.py tests/rl/test_rollout_worker.py tests/rl/environments/test_math_env.py tests/rl/environments/test_mock_env.py`
  - result: `39 passed`
- repo checks on touched files:
  - `./infra/pre-commit.py --fix ...`
  - result: passed, including `pyrefly`
- broader RL suite:
  - `uv run --package marin pytest tests/rl`
  - result: `150 passed, 2 skipped, 12 deselected`

### Current state

This is ready for the next Iris validation run.

The important unanswered question is now purely empirical:

- does pinning eval to `replica 1` and letting train continue on `replica 0` materially improve packed RL cadence versus `PKR-001`?

### Next step

Run a short packed RL smoke on Iris with the new implementation and check:

- rollout-side metrics show eval requests only on `replica 1`
- train generate requests continue on `replica 0` during eval windows
- trainer `batch_prep` improves materially versus `PKR-001`

## 2026-03-26 23:43 PDT - Launch plan for `PKR-002`

Next experiment:

- `PKR-002`
- goal: rerun the short 5-step packed RL smoke after the eval-isolation changes
- success criteria:
  - root / trainer / rollout all succeed cleanly
  - rollout metrics show eval requests pinned to `replica 1`
  - train requests continue on `replica 0` during eval windows
  - trainer `batch_prep` improves versus `PKR-001`

Launch command:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-packed-pkr002-20260326-2343 --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_gcs_packed.py --experiment-name-suffix pkr002 --num-train-steps 5 --n-prompts 64 --eval-frequency 1 --region us-central1`

## 2026-03-27 00:21 PDT - Launch plan for `PKR-003`

Next experiment:

- `PKR-003`
- goal: isolate packed-rollout throughput with eval effectively out of the way
- shape: same packed rollout topology as `PKR-002`, but extend to `10` train steps and make eval sparse enough that it should not materially affect the run
- assumption: setting `eval_frequency=100` on a `10`-step run is sufficient to remove eval as the main source of rollout contention

Success criteria:

- root / trainer / rollout all succeed cleanly
- first packed rollout batch arrives without a hard startup failure
- trainer steady-state cadence is substantially better than the eval-heavy packed runs
- `batch_prep` stays near the low-single-digit baseline instead of repeatedly spiking into the `40-60s` range

Launch command:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-packed-pkr003-20260327-0021 --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_gcs_packed.py --experiment-name-suffix pkr003 --num-train-steps 10 --n-prompts 64 --eval-frequency 100 --region us-central1`

## 2026-03-27 00:55 PDT - `PKR-003` capacity status

Current Iris state:

- root job `/ahmed/iris-rl-packed-pkr003-20260327-0021`: `JOB_STATE_RUNNING`
- trainer child `/ahmed/iris-rl-packed-pkr003-20260327-0021/rl-pkr003-20260327-072722-train`: `JOB_STATE_RUNNING`
- rollout child `/ahmed/iris-rl-packed-pkr003-20260327-0021/rl-pkr003-20260327-072722-rollout-0`: `JOB_STATE_PENDING`
- `failure_count=0` and `preemption_count=0` across the whole tree

Current blocker is scheduler capacity, not a code/runtime failure:

- `Insufficient TPUs (need 4, available 0)`
- `Insufficient memory (need 400.0GB, available 304.8GB)`
- autoscaler backoff on `tpu_v5p_8-us-central1-a`

Implication:

- trainer is alive but still waiting for initial rollouts
- rollout has not started yet, so there is no rollout-side correctness or throughput signal from `PKR-003` yet
- no traceback, OOM, or packed-vLLM startup error has appeared so far; this is purely a cluster-capacity stall

## 2026-03-27 00:58 PDT - 8-hour babysit queue

User requested continuous monitoring for the next `8` hours and asked that follow-up jobs launch automatically as earlier runs complete.

Queue for this monitoring window:

1. keep babysitting `PKR-003` until it reaches a terminal state
2. if `PKR-003` succeeds cleanly, launch `PKR-004`
3. if `PKR-004` also succeeds cleanly and the monitoring window still has time left, launch `PKR-005`

Planned follow-up experiments:

- `PKR-004`
  - goal: measure packed RL with normal eval cadence after the eval-isolation changes
  - shape: same packed topology, `10` train steps, `eval_frequency=1`
- `PKR-005`
  - goal: longer packed confirmation run if `PKR-004` looks healthy enough to justify extending the run
  - shape: same packed topology, `30` train steps, `eval_frequency=1`

Assumptions for the automated queue:

- use the same packed experiment file: `experiments/exp_iris_rl_regression_direct_gcs_packed.py`
- keep `n_prompts=64`
- treat scheduler capacity waits as non-failures and continue waiting
- do not mutate the cluster; only submit/monitor Iris jobs

## 2026-03-27 02:08 PDT - `PKR-003` capacity cleared and run started

`PKR-003` is no longer blocked on rollout capacity.

Observed transition:

- at `02:06 PDT`, Iris status changed from rollout `PENDING` to rollout `RUNNING`
- root and trainer both remained `RUNNING`
- no failures or preemptions were recorded at the moment capacity cleared

Early runtime signal after rollout startup:

- trainer already completed `training step 1` with `duration=88.39s` and `batch_prep=4.54s`
- packed rollout logs show both local children active:
  - child `0` activated weight `0` and started a `64`-prompt generate
  - child `1` activated weight `0`, fetched weight `1`, and started a `500`-prompt generate

Interpretation:

- the job has moved from scheduler wait into real packed-RL execution
- the immediate next thing to watch is whether steady-state cadence stays near the low `batch_prep` seen at step `1`, and whether sparse eval actually keeps rollout contention low

## 2026-03-27 02:37 PDT - `PKR-003` succeeded; continuing to `PKR-004`

`PKR-003` finished successfully on Iris:

- root `/ahmed/iris-rl-packed-pkr003-20260327-0021`: `JOB_STATE_SUCCEEDED`
- trainer child: `JOB_STATE_SUCCEEDED`
- rollout child: `JOB_STATE_SUCCEEDED`
- `failure_count=0`, `preemption_count=0`

Operational note:

- the queue monitor hit a wrapper-side post-success status-resolution bug after `PKR-003` completed
- this was a monitor bug, not an Iris or RL job failure
- fix applied locally: add retries around post-monitor root-state resolution before deciding whether to advance the queue

Next action:

- launch `PKR-004` directly and reattach the fixed queue monitor on top of the new active job

Planned launch command:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-packed-pkr004-20260327-0237 --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_gcs_packed.py --experiment-name-suffix pkr004 --num-train-steps 10 --n-prompts 64 --eval-frequency 1 --region us-central1`

## 2026-03-27 03:19 PDT - `PKR-004` launched and is running after one preemption

Launch result:

- root job `/ahmed/iris-rl-packed-pkr004-20260327-0237`
- trainer child `/ahmed/iris-rl-packed-pkr004-20260327-0237/rl-pkr004-20260327-094215-train`
- rollout child `/ahmed/iris-rl-packed-pkr004-20260327-0237/rl-pkr004-20260327-094215-rollout-0`

Current state at this checkpoint:

- root: `JOB_STATE_RUNNING`
- trainer: `JOB_STATE_RUNNING`, `preemption_count=1`
- rollout: `JOB_STATE_RUNNING`, `preemption_count=1`
- no child has a recorded failure yet

Observed runtime pattern so far:

- startup again emitted the scoped-Vmem `INVALID_ARGUMENT` spam seen in earlier packed runs, but recovered
- early steps matched the packed/eval-heavy expectation:
  - step `0`: `111.33s`, `batch_prep=4.71s`
  - step `1`: `91.12s`, `batch_prep=4.50s`
  - step `2`: `60.85s`, `batch_prep=46.43s`
  - step `3`: `60.92s`, `batch_prep=44.23s`
- eval metrics advanced:
  - step `-1`: `avg_at_1=0.292`
  - step `0`: `avg_at_1=0.344`

Operational note:

- later logs showed a restart-like sequence and Iris now reports `preemption_count=1` on both trainer and rollout
- the job recovered and remains `RUNNING`, so the babysit loop stayed attached instead of resubmitting
- `PKR-005` remains queued to launch automatically if `PKR-004` reaches a clean terminal success

## 2026-03-27 09:12 PDT - `PKR-004` succeeded; overall verdict unchanged

Final Iris state:

- root `/ahmed/iris-rl-packed-pkr004-20260327-0237`: `JOB_STATE_SUCCEEDED`
- trainer child: `JOB_STATE_SUCCEEDED`, `preemption_count=2`
- rollout child: `JOB_STATE_SUCCEEDED`, `preemption_count=1`
- `failure_count=0` everywhere

What `PKR-004` showed:

- the packed direct Iris RL path is operational under normal eval cadence too, not just sparse-eval mode
- the same qualitative throughput pattern reappeared:
  - very low `batch_prep` on the first two steps
  - then large `batch_prep` inflation once eval work overlaps rollout supply
- startup again emitted the scoped-Vmem `INVALID_ARGUMENT` spam seen in earlier packed runs, but the run recovered

Current verdict:

- correctness verdict: **yes**
  - direct packed RL on Iris is validated
  - one rollout `v5p-8` can host the packed `2 x TP=2` path end-to-end in real RL
- production-throughput verdict: **not yet**
  - sparse-eval packed RL (`PKR-003`) looked materially healthier
  - normal-eval packed RL (`PKR-004`) still exhibits the eval-heavy contention pattern that keeps it from being an obvious `e4ms2` replacement

Decision at this point:

- packed rollout is a real and working topology
- but a single packed rollout worker is still not a proven drop-in replacement for the old two-rollout-worker shape under `eval_frequency=1`
- the next promotion step should be gated on whether we want to invest in better eval isolation / ownership, not on sampler correctness anymore

## 2026-03-27 09:12 PDT - Bottom-line status

Bottom line after `PKR-003` and `PKR-004`:

- direct packed RL on Iris: **validated**
- packed rollout as the new default RL topology: **not yet promoted**

Recommended interpretation:

- use `PKR-003` as evidence that the packed sampler path itself is healthy when eval pressure is mostly removed
- use `PKR-004` as evidence that normal eval cadence still reintroduces enough rollout contention to block promotion over `e4ms2`

Recommended next decision:

- either invest in stronger eval isolation / scheduling changes
- or keep the packed path as a validated capability without making it the production default yet

## 2026-03-28 - Claude handoff findings imported

Source: `.agents/logbooks/iris-rl-claude.md`

This section imports the main conclusions from the separate Claude session so the Codex logbook keeps the migration, performance, and reliability story in one place.

### 1. Root cause of the early Iris performance gap

Claude's main performance finding was that the slow `r4` Iris run was not trainer-bound. The trainer-side compute was already competitive with on-demand; the gap was rollout-side.

With `n_prompts=16`, Iris needed `4` rollout batches to fill one `1024`-sample trainer batch. Because rollout eval was keyed off the rollout worker's internal batch counter, `eval_frequency=1` accidentally meant `4` full MATH-500 evals per trainer step instead of `1`. That is the dominant explanation for the `~7.9 min/step` behavior in `r4`.

See `.agents/logbooks/iris-rl-claude.md`:
- `Root Cause: eval_frequency interacts badly with small n_prompts`
- `Performance Comparison: Iris r4 vs On-Demand nb-inflight2`

### 2. Validation that the simple config fix worked

Claude then validated the obvious corrective action: match the on-demand rollout shape by setting `n_prompts=64`, so one rollout batch naturally feeds one trainer step.

That `e4par` Iris run recovered the expected cadence almost exactly:
- `r4`: `~7.9 min/step`
- `e4par`: `~2.3 min/step`
- on-demand `nb-inflight2`: `~2.3 min/step`

This is important context for the later packed-rollout work in this logbook: the first major Iris slowdown was a rollout/eval-cadence bug, not evidence that Iris itself was intrinsically slower.

See `.agents/logbooks/iris-rl-claude.md`:
- `Validation: Iris e4par run (n_prompts=64, inflight updates)`

### 3. Best-performing non-packed Iris topology observed by Claude

Claude's strongest throughput result before the packed-rollout branch here was `e4ms2`: `1` trainer plus `2` rollout workers, each with `n_prompts=64`.

Recorded outcome:
- median wall clock `~1.7 min/step`
- materially faster than both `r4` and the best on-demand reference
- trainer `batch_prep` dropped sharply because the trainer was almost always fed

Claude also quantified the cost of that speedup:
- rollout production averaged `~1.43` batches per trainer step
- roughly `29-30%` of produced rollout data was discarded
- this was not a replay-buffer bug; it was the expected price of overproducing to keep the trainer saturated

Claude's analysis also concluded that `max_samples=2` would not help under the current `max_rollout_step_delay=1` setting. The better levers were:
- move eval off the critical sampler path
- reduce checkpoint cost/frequency
- reduce per-sampler prompt count if we want a different throughput/waste tradeoff

This provides the baseline that the later packed `PKR-*` work in this logbook was trying to match or beat.

See `.agents/logbooks/iris-rl-claude.md`:
- `Multi-Sampler Run: e4ms2 (1 trainer + 2 rollout workers)`
- `Replay Buffer Mechanics and Rollout Overproduction Analysis`

### 4. Pure-inference findings that motivated the packed-rollout branch

Claude ran the pure-inference ladder to answer whether packed `TP=2x2` on one `v5p-8` was even worth attempting.

The high-signal result was that `TP=4` throughput stayed effectively flat while `gpu_memory_utilization` dropped from `0.90` to `0.60`. That meant the KV cache reservation was oversized for this workload and that packed `TP=2x2` looked memory-feasible on `v5p-8`.

That inference-side conclusion is the direct precursor to the packed-rollout implementation and `PKR-001` through `PKR-004` validation recorded later in this Codex logbook.

See `.agents/logbooks/iris-rl-claude.md`:
- `INF-001 Results: TP=4 Baseline (2026-03-26)`
- `INF-002: TP=4 Cache-Budget Sweep (2026-03-26)`
- `INF-004 Plan: Data-Parallel TP=2x2 on Single v5p-8`

### 5. Reliability finding: retry did not resume until naming was split

Claude found a separate high-impact reliability bug during the long `e4ms2` thread: preempted Iris RL runs were not actually resuming.

Root cause:
- the experiment script generated a fresh timestamp-based name on every invocation
- Iris retry re-ran the script
- checkpoint paths, W&B run IDs, rollout storage paths, and actor names all changed on retry
- Levanter then correctly saw an empty checkpoint directory and restarted from the base model

The required fix was to split:
- stable run identity for checkpoints, W&B, and rollout storage
- per-attempt instance identity for child-job names and actor names

Claude's logbook records this fix as implemented and production-validated on the long `iris-rl-e4ms2-500` run. That work is now part of the carry-forward state for this thread.

See `.agents/logbooks/iris-rl-claude.md`:
- `Critical Bug: RL runs do not resume after preemption (2026-03-27)`
- `Fix implemented and validated (2026-03-28)`

### 6. Remaining carry-forward issue from the Claude session

The main unresolved issue at Claude handoff was not checkpoint/W&B resume anymore; that part was working. The remaining issue was trainer liveness after certain failures.

Observed behavior:
- trainer could fail or hang mid-step
- Arrow Flight / background threads kept the process alive long enough that Iris still showed the job as `RUNNING`
- W&B stopped advancing
- the run eventually recovered through Iris retry, but only after a delayed failure-detection window

This is consistent with the coordinator-liveness / zombie-process concern already noted elsewhere in this logbook. The practical consequence is that resume works, but failure surfacing and fast teardown are still not fully solved.

See `.agents/logbooks/iris-rl-claude.md`:
- `Trainer stuck/zombie issue at step 143 - detailed crash log`
- `1-hour status check (2026-03-28 ~10:30 UTC)`

## 2026-03-28 18:45 PDT - e4ms2-500 final trainer failure signature narrowed; diagnostics patch landed

Context:
- root job: `/ahmed/iris-rl-e4ms2-500-0327-v2`
- trainer run: `iris-rl-e4ms2-500-train`
- local debug note: `docs/debug-log-iris-rl-e4ms2-500-final-failure.md`

This follow-up was specifically to answer:
1. what exact signature the repeated trainer failures share
2. whether the late failure should still be treated as a simple host-RAM / OOM issue
3. how to improve logging so the next failure is materially easier to diagnose

### 1. Repeated failure signature is now much narrower than "generic coordination error"

Trainer attempts `0`, `2`, `3`, and `4` all show the same sequence:

- training continues normally up to a checkpoint boundary
- Levanter logs:
  - `Saving temporary checkpoint at step ...`
  - `Saving checkpoint at step ...`
- JAX checkpoint logs:
  - `Waiting for previous serialization to finish`
  - `Thread joined successfully`
  - `Error check finished successfully`
- then the main trainer thread goes silent
- replay-buffer / rollout-ingest background threads continue logging briefly
- the task finally dies with:
  - `UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats): /job:jax_worker/replica:0/task:0`
  - `RPC: /tensorflow.CoordinationService/PollForError`

Crucial narrowing observation:
- the failing checkpoints never emit:
  - `Starting commit to storage layer by process: 0`
- but successful earlier checkpoints in the same attempts do emit:
  - `Starting commit to storage layer by process: 0`
  - `Finished committing to storage layer by process: 0`
  - `on_commit_callback successfully ran!`

Interpretation:
- the coordination error is still the outer death notice
- the shared inner signature is now checkpoint-path failure before async commit starts
- more precisely: the trainer appears to wedge or die in JAX/TensorStore pre-commit serialization, not in the later commit/on-commit callback stage

The local code path supports that reading:
- `levanter.checkpoint.save_checkpoint(...)` enters `tree_serialize_leaves_tensorstore(...)`
- that calls `GlobalAsyncCheckpointManager.serialize(...)`
- JAX runs `asyncio.run(_run_serializer())`
- only after that returns does it start the async commit thread that logs `Starting commit to storage layer`

So the best current phrase for the failure family is:
- checkpoint-triggered trainer death during pre-commit TensorStore serialization

### 2. Revised interpretation: this is not strong evidence for a simple `400g`-budget OOM

An earlier hypothesis was that this might still be "basically checkpoint OOM" in a broader sense. That should now be weakened.

What changed the interpretation:
- this run requested `400g` TPU host RAM
- decoded Iris task resource usage for the trainer showed:
  - `memory_peak_mb = 319590` (~`312 GiB`)
- that is very large, but it is still below the requested `400g` envelope

Most careful interpretation now:
- checkpointing is still the trigger boundary
- but the available evidence does **not** justify calling this a straightforward cgroup-limit OOM
- a better current guess is:
  - native JAX / TensorStore / TPU-host-transfer wedge or abort during serialization
  - possibly worker-specific flakiness
  - possibly some form of resource pressure / fragmentation that does not surface cleanly as `exit 137`

This is a more defensible statement than "it ran out of RAM again."

### 3. Logging / diagnostics improvements implemented immediately

Because the main blocker is now observability, a diagnostics-oriented patch was landed rather than continuing to speculate from the same weak signal.

Changes made:

1. Unbuffered worker logs
- RL worker environments now set:
  - `PYTHONUNBUFFERED=1`
- goal: reduce the chance that the true last logs are lost behind Python buffering

2. Train-worker manual thread dumps
- train worker now mirrors rollout worker behavior:
  - `faulthandler.enable()`
  - register `SIGUSR2` to dump all Python thread stacks to stderr
- goal: if a trainer is visibly wedged, we can get an immediate multi-thread stack snapshot without waiting for process death

3. Existing TensorStore metrics hook now enabled for RL trainer
- `install_tensorstore_metrics_hook_if_enabled(trainer)` is now wired into the RL train worker
- when `LEVANTER_LOG_TENSORSTORE_METRICS_EVERY` is set, RL runs will log TensorStore counters per step just like `train_lm`

4. Explicit checkpoint phase markers in Levanter checkpoint save path
- checkpoint saves now emit:
  - `PHASE: CHECKPOINT step=... phase=starting`
  - `filesystem_ready`
  - `tensorstore_serialize`
  - `async_commit_in_flight`
  - `metadata_write`
  - terminal `completed` / `failed`
- goal: make the exact checkpoint subphase visible in logs instead of inferring it indirectly from JAX internal messages

5. Periodic "checkpoint still running" progress logs
- long-running checkpoints now periodically log:
  - step
  - current checkpoint phase
  - total elapsed
  - phase elapsed
  - current process RSS when available via `psutil`
- goal: distinguish "slow checkpoint" from "silent hang" and provide lightweight host-memory context

6. Optional automatic Python thread dump on long checkpoint stall
- new env knob:
  - `LEVANTER_CHECKPOINT_DUMP_STACKS_AFTER_SECONDS`
- if set, a long-running checkpoint will automatically dump Python thread stacks once the timeout is exceeded
- goal: catch where Python-visible threads are blocked when the trainer never makes it to commit logs

7. TensorStore serialization entry/exit markers
- added logs around `manager.serialize(...)` with:
  - checkpoint dir
  - number of arrays
  - estimated total serialized array size
  - largest leaf path + size
  - explicit enter / return markers around `manager.serialize(...)`
- goal: prove whether the failure occurs before or after JAX returns from the serializer call and capture the checkpoint scale at the failing step

Validation:
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/orchestration.py lib/marin/src/marin/rl/train_worker.py lib/levanter/src/levanter/checkpoint.py lib/levanter/src/levanter/tensorstore_serialization.py`
- result: `OK`

### 4. Practical next-run guidance

For the next long Iris RL run, the highest-value debug configuration is:

- `LEVANTER_LOG_TENSORSTORE_METRICS_EVERY=1`
- `LEVANTER_CHECKPOINT_DUMP_STACKS_AFTER_SECONDS=180`

And if the trainer appears hung while the process is still alive:
- send `SIGUSR2` to the trainer process to dump all Python thread stacks

Expected outcome next time:
- we should know which checkpoint phase the run died in
- whether `manager.serialize(...)` returned
- whether async commit began
- whether metadata write began
- whether Python threads were stuck in a visible blocking point

Bottom line:
- the run still failed because the trainer exhausted ordinary failure retries
- but the deeper failure family is now best described as:
  - repeated checkpoint-path trainer death in pre-commit serialization
- and the repository now has substantially better instrumentation to identify the exact failing boundary on the next reproduction

## 2026-03-28 19:09 PDT - fixed resumed `weight_transfer/*` trainer metrics

Follow-up on the trainer W&B run `iris-rl-e4ms2-500-train`:

- user noticed that `weight_transfer/total_transfers` kept restarting near zero inside a single resumed trainer run
- question was whether W&B resume had failed, or whether this was a metric-definition problem on our side

### 1. W&B resume was not the bug

Direct history inspection of:
- `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-train`

showed:
- `weight_transfer/total_transfers` dropped at trainer steps:
  - `67`
  - `143`
  - `281`
  - `405`
- `weight_transfer/successful_transfers` dropped at those same steps
- `weight_transfer/failed_transfers` did not show the same reset pattern

Cross-checks:
- `throughput/total_tokens` stayed monotonic
- `throughput/total_gflops` stayed monotonic

Interpretation:
- the trainer W&B run id resumed correctly
- only the weight-transfer server counters were being reset

### 2. Root cause: process-local transfer-server counters were being logged as if they were run-global counters

The code path before the fix was:

- `TrainWorker.__init__` recreated a fresh transfer server on every trainer child retry
- `ArrowFlightServer.__init__` created a fresh in-memory `WeightTransferServerMetrics()` dataclass with zeros
- `weight_transfer_hook(...)` logged `dataclasses.asdict(self.transfer_server.get_metrics())` directly to W&B

So after a trainer retry:
- the W&B run resumed under the same stable trainer id
- but the source counters were back at `0`
- therefore the W&B chart showed a reset even though the run itself had resumed correctly

This was a metric-semantics bug, not a W&B resume bug.

### 3. Simplest fix implemented: split attempt-local vs resume-safe global counters

Instead of adding persistence plumbing through `RLRunState`, the fix implemented the smallest correct change:

1. Keep the existing server counters, but rename them explicitly as attempt-local metrics:
- `weight_transfer/attempt_total_transfers`
- `weight_transfer/attempt_successful_transfers`
- `weight_transfer/attempt_failed_transfers`

2. Publish run-global cumulative counters from restored trainer step state:
- `weight_transfer/total_transfers`
- `weight_transfer/successful_transfers`

Reason this works:
- there is exactly one bootstrap transfer before training starts
- then one transfer hook every `sync_interval_steps`, starting at trainer step `0`

So for a logged transfer step:
- global cumulative transfers = `2 + step // sync_interval_steps`

For the observed run with `sync_interval_steps=1`, that gives:
- step `0` -> `2`
- step `67` -> `69`
- step `143` -> `145`

which matches the intended run-global semantics and removes the retry-induced resets.

### 4. Code / test updates

Code:
- added `_resume_safe_weight_transfer_metrics(...)` in `lib/marin/src/marin/rl/train_worker.py`
- updated `weight_transfer_hook(...)` to log:
  - attempt-local counters from the transfer server dataclass
  - run-global monotonic counters derived from trainer step

Tests:
- extended `tests/rl/test_train_worker.py` with:
  - direct coverage for `_resume_safe_weight_transfer_metrics(...)`
  - a focused hook test verifying that a restarted trainer can log:
    - attempt-local counters such as `8`
    - run-global counters such as `69`
    at the same trainer step

Validation:
- `uv run pytest -q tests/rl/test_train_worker.py tests/rl/test_orchestration.py`
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/train_worker.py tests/rl/test_train_worker.py .agents/logbooks/iris-rl-codex.md docs/debug-log-weight-transfer-metric-reset.md`

Outcome:
- resumed trainer runs should now keep the main weight-transfer charts monotonic
- and the attempt-local values remain available for debugging per-process behavior after retries

## 2026-03-28 19:32 PDT - fixed rollout W&B naming and rollout receive counter semantics

Follow-up on the two rollout runs from `iris-rl-e4ms2-500`:

- `iris-rl-e4ms2-500-rollout-0`
- `iris-rl-e4ms2-500-rollout-1`

The user noticed two related but distinct issues:

1. W&B sidebar confusion
- the two rollout runs had distinct URLs and ids
- but both displayed the same visible name:
  - `iris-rl-e4ms2-500-20260328-031315-rollout`

2. Resume-unfriendly receive counters
- rollout-side counters such as `inference.successful_receives` could reset inside a resumed run
- this looked like the rollout analogue of the trainer-side `weight_transfer/total_transfers` issue

### 1. W&B rollout display-name fix

Diagnosis:
- this was a display-name collision, not a run-id collision
- both rollout runs were distinct in W&B:
  - ids / URLs were `...-rollout-0` and `...-rollout-1`
- but the visible name had been shared

Important design constraint:
- rollout W&B labels must be robust to preemption
- therefore they cannot key off the volatile Iris instance name

Fix implemented:
- rollout W&B naming now uses the stable per-worker identity
- coordinator stamps each rollout worker with:
  - `run_id = <stable-run-id>-rollout-<worker_index>`
  - `tracker_config.name = <stable-run-id>-rollout-<worker_index>`
- rollout tracker now uses:
  - `name = config.name or run_id`
  - `id = run_id`

Implication:
- worker `0` now stays visibly `...-rollout-0`
- worker `1` now stays visibly `...-rollout-1`
- retries of the same logical worker resume into the same W&B run and keep the same visible label

### 2. Rollout receive counters now split into attempt-local vs resume-safe global metrics

Diagnosis:
- rollout workers were logging `WeightTransferClientMetrics` directly
- those counters live in an in-memory client object
- after a rollout retry, the W&B run resumed but the counter source restarted

That meant the main W&B series:
- `inference.total_polls`
- `inference.successful_receives`
- `inference.failed_receives`

were not truly run-global counters; they were attempt-local counters masquerading as global ones.

Fix implemented:

1. Added coordinator-hosted cumulative storage in `RLRunState`
- per `worker_index`, track cumulative:
  - `total_polls`
  - `successful_receives`
  - `failed_receives`

2. Rollout worker now keeps the last locally observed attempt counters
- at each log step it computes non-negative deltas
- if a local counter decreases, that is treated as a fresh attempt-local counter after retry

3. Logging semantics now match trainer-side conventions
- attempt-local metrics:
  - `inference.attempt_total_polls`
  - `inference.attempt_successful_receives`
  - `inference.attempt_failed_receives`
- resume-safe cumulative metrics:
  - `inference.total_polls`
  - `inference.successful_receives`
  - `inference.failed_receives`

This makes the main rollout receive charts monotonic across retries while preserving per-attempt debug visibility.

### 3. Tests / validation

Tests added / updated:
- `tests/rl/test_rollout_worker.py`
  - rollout tracker uses explicit stable name
  - rollout receive metrics log attempt-local and cumulative values correctly after a counter reset
- `tests/rl/test_run_state.py`
  - `RLRunState` accumulates rollout transfer counters per worker
- `tests/rl/test_orchestration.py`
  - coordinator assigns stable per-worker rollout W&B names

Planned validation commands for this patch:
- `uv run pytest -q tests/rl/test_rollout_worker.py tests/rl/test_run_state.py tests/rl/test_orchestration.py`
- `make-fix`

Outcome:
- rollout W&B sidebar labels should now be unambiguous and preemption-safe
- rollout receive counters should now have the same resume-safe semantics as the trainer-side transfer counters

## 2026-03-28 19:40 PDT - launch plan for checkpoint-debug 500-step rerun

Next experiment:

- relaunch the `e4ms2` 500-step Iris RL job with the new checkpoint diagnostics enabled
- goal: if the trainer dies again in the checkpoint path, capture explicit checkpoint phase logs and automatic Python thread dumps instead of only the outer JAX coordination failure

Fresh stable run identity for this debug reproduction:

- stable run name: `iris-rl-e4ms2-500-ckptdbg`
- root Iris job name: `iris-rl-e4ms2-500-0328-ckptdbg`

Why a fresh stable run name:

- keep debug artifacts separate from the earlier crashed run
- avoid mixing old checkpoint / rollout / W&B state with the new reproduction
- still preserve preemption-safe resume inside this new run via `--run-name`

Config for this relaunch:

- trainer steps: `500`
- rollout workers: `2`
- `n_prompts=64`
- `eval_frequency=1`
- region: `us-central1`
- checkpoint diagnostics:
  - `--debug-checkpointer`
  - `--debug-checkpointer-dump-stacks-after 180`

Launch command:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-e4ms2-500-0328-ckptdbg --region us-central1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-e4ms2-500-ckptdbg --num-train-steps 500 --n-prompts 64 --eval-frequency 1 --num-rollout-workers 2 --region us-central1 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`

Submission result:

- root Iris job: `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg`

Monitoring plan for this run:

- keep detailed babysit state and poll-by-poll signals out of this main logbook
- instead, use dedicated tracking files under `scratch/`
- main logbook should only carry pointers and major conclusions

Tracking files:

- state JSON:
  - `scratch/20260328-1955_monitoring_state.json`
- event JSONL:
  - `scratch/20260328-1955_iris-rl-e4ms2-500-0328-ckptdbg_events.jsonl`
- babysit text log:
  - `scratch/20260328-1955_iris-rl-e4ms2-500-0328-ckptdbg_babysit.log`

Babysit cadence:

- immediate snapshot on startup
- one follow-up poll after `120s`
- then steady-state polls every `1800s` (`30 min`)

Monitoring mode:

- monitor-only, no automatic resubmit
- reason: if this run fails again, preserve the exact failed state and checkpoint-debug evidence for manual diagnosis

## 2026-03-28 20:19 PDT - parallel east1-d v6e-8 debug run plan

While the `us-central1` checkpoint-debug rerun is still active, launch a parallel
reproduction on `v6e-8` in `us-east1-d`.

Purpose:

- check whether the late checkpoint-path trainer failure reproduces on a different TPU generation and zone
- keep checkpoint diagnostics enabled so a second failure yields the same richer evidence

Small launcher cleanup before submit:

- `experiments/exp_iris_rl_regression_direct_gcs_prod.py` no longer hardcodes `v5p-8`
- added `--tpu-type` so the same experiment can target `v6e-8` without a one-off edit

Placement note:

- Fray `ResourceConfig` only exposes `regions`, not `zones`
- Iris child jobs inherit the parent job's zone constraint, so the root submit should carry:
  - `--region us-east1`
  - `--zone us-east1-d`
- the experiment itself should set `--region us-east1` and `--tpu-type v6e-8`

Planned identity and command:

- stable run name: `iris-rl-e4ms2-500-ckptdbg-v6e8-e1d`
- root Iris job name: `iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d`
- command:
  - `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d --region us-east1 --zone us-east1-d -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-e4ms2-500-ckptdbg-v6e8-e1d --num-train-steps 500 --n-prompts 64 --eval-frequency 1 --num-rollout-workers 2 --region us-east1 --tpu-type v6e-8 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`

Submission result:

- submitted root Iris job:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d`

Immediate scheduler result:

- root job is `JOB_STATE_PENDING`
- pending reason:
  - `Autoscaler: Unsatisfied autoscaler demand: no_matching_group: no groups in zone us-east1-d (did you mean us-east1-b?)`

Interpretation:

- the launcher patch worked and the job submit itself is valid
- the pending diagnostic is about the root CPU coordinator job, not TPU `v6e-8` availability
- `lib/iris/examples/marin.yaml` has `v6e` TPU groups in `us-east1-d`, but the non-preemptible CPU executor group only exists in:
  - `us-central1-a`
  - `us-east1-b`
  - `us-west1-a`
  - `europe-west4-a`
- because the root submit used `--zone us-east1-d`, Iris tried to place the small CPU coordinator in that same zone first, and the executor heuristic auto-tagged it non-preemptible
- the autoscaler therefore compared against non-preemptible CPU groups, saw none in `us-east1-d`, and suggested the closest CPU zone `us-east1-b`

Correct takeaway:

- this does **not** prove `v6e-8` TPU capacity is unavailable in `us-east1-d`
- it proves the current RL launch topology cannot zone-pin children to `us-east1-d` through the root job unless there is also a schedulable CPU coordinator path in `us-east1-d`

## 2026-03-28 20:33 PDT - child-zone hack for east1-d v6e-8

User does not have Iris admin access, so the operational fix of adding a small
non-preemptible CPU executor group in `us-east1-d` is not available.

Pragmatic code-side workaround:

- let the root RL coordinator stay only region-pinned
- add explicit child TPU job zone support so trainer and rollout workers can pin to `us-east1-d`

Minimal patch shape:

- Fray `ResourceConfig` now has an optional `zone`
- Fray Iris backend now converts `ResourceConfig.zone` into an Iris `zone` constraint
- RL `RunConfig` now carries an optional child-worker `zone`
- RL orchestration forwards that zone into trainer and rollout `ResourceConfig.with_tpu(...)`
- `experiments/exp_iris_rl_regression_direct_gcs_prod.py` now accepts `--zone` and passes it through to `RunConfig`

Validation:

- `uv run pytest -q lib/fray/tests/test_v2_iris.py tests/rl/test_orchestration.py`
- result: `19 passed`
- experiment CLI smoke check confirms `--zone` is exposed and parsed

Corrected east1-d launch shape:

- root submit should carry `--region us-east1` and **no** `--zone`
- experiment should carry:
  - `--region us-east1`
  - `--zone us-east1-d`
  - `--tpu-type v6e-8`

This should allow:

- root CPU coordinator to land on any schedulable east1 CPU executor, likely `us-east1-b`
- child TPU jobs to pin explicitly to `us-east1-d`

Submission result:

- terminated the earlier bad root-zoned pending job:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d`
- submitted corrected root job:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d-childzone`

Corrected command:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d-childzone --region us-east1 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-e4ms2-500-ckptdbg-v6e8-e1d-childzone --num-train-steps 500 --n-prompts 64 --eval-frequency 1 --num-rollout-workers 2 --region us-east1 --zone us-east1-d --tpu-type v6e-8 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`

Immediate scheduler result:

- root job is `JOB_STATE_PENDING`
- pending reason now says:
  - `Autoscaler: (scaling up) Waiting for worker scale-up in scale group 'cpu_vm_e2_highmem_2_ondemand-us-east1-b'`

Interpretation:

- this is the expected coordinator placement path
- the earlier `us-east1-d` mismatch is gone
- next thing to watch is whether the root coordinator becomes running and then launches child TPU jobs pinned to `us-east1-d`

## 2026-03-28 20:46 PDT - complete east1-d v6e-8 launch chronology, constraints, and blocker

This entry consolidates the full `us-east1-d` / `v6e-8` launch thread. The
earlier entries above capture each step as it happened; this section records the
final precise interpretation.

Goal:

- run the same 500-step checkpoint-debug RL reproduction on `v6e-8`
- keep trainer + rollout TPU jobs in `us-east1-d`
- allow the small CPU coordinator to run wherever Iris can place it

### Stage 1 - direct root zone pin failed for the wrong reason

Initial launch:

- root job:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d`
- command shape:
  - root submit used:
    - `--region us-east1`
    - `--zone us-east1-d`
  - experiment used:
    - `--region us-east1`
    - `--tpu-type v6e-8`

Observed scheduler result:

- root job stayed `JOB_STATE_PENDING`
- pending reason said:
  - `no groups in zone us-east1-d (did you mean us-east1-b?)`

What this actually meant:

- it was **not** saying `v6e-8` TPU capacity was missing in `us-east1-d`
- it was complaining about the root CPU coordinator placement
- Iris CLI auto-tagged the tiny root CPU job as non-preemptible
- the relevant root-job constraints at this stage were:
  - `region`
  - `zone`
  - `preemptible`
  - `reservation-job`
- the only non-preemptible CPU executor groups in `lib/iris/examples/marin.yaml` are in:
  - `us-central1-a`
  - `us-east1-b`
  - `us-west1-a`
  - `europe-west4-a`
- there is no such CPU executor group in `us-east1-d`

Important correction:

- `lib/iris/examples/marin.yaml` really does define `v6e` TPU groups in:
  - `europe-west4-a`
  - `us-east1-d`
  - `us-east5-b`
- so the root-zoned failure was a CPU-coordinator mismatch, not a TPU-capacity proof

### Stage 2 - code-side child-zone support hack

Because the user does not have Iris admin access, the operational fix
(adding a small CPU executor in `us-east1-d`) was not available.

The minimal code workaround implemented here was:

- Fray `ResourceConfig` gained optional `zone`
- Fray Iris backend now converts `ResourceConfig.zone` into an Iris `zone` constraint
- RL `RunConfig` gained optional `zone`
- RL orchestration now forwards `RunConfig.zone` into trainer + rollout TPU `ResourceConfig.with_tpu(...)`
- `experiments/exp_iris_rl_regression_direct_gcs_prod.py` gained:
  - `--tpu-type`
  - `--zone`

Validation run after the patch:

- `uv run pytest -q lib/fray/tests/test_v2_iris.py tests/rl/test_orchestration.py`
- result:
  - `19 passed`

Semantics of the hack:

- root submit may be left unzoned, or even unpinned entirely
- child TPU jobs can still carry explicit:
  - `region=us-east1`
  - `zone=us-east1-d`
- this works because child jobs now carry their own explicit region + zone instead
  of relying only on parent inherited placement

### Stage 3 - root region pinned, child zone pinned

Second launch after the hack:

- terminated bad root-zoned job:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d`
- submitted:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d-childzone`

Command shape:

- root submit used:
  - `--region us-east1`
  - no root `--zone`
- experiment used:
  - `--region us-east1`
  - `--zone us-east1-d`
  - `--tpu-type v6e-8`

Observed scheduler result:

- root job stayed `JOB_STATE_PENDING`
- pending reason first showed:
  - waiting for scale-up in `cpu_vm_e2_highmem_2_ondemand-us-east1-b`
- later showed:
  - `no_capacity: cpu_vm_e2_highmem_2_ondemand-us-east1-b=backoff`

Interpretation:

- this confirmed the stage-1 bug was fixed
- the root coordinator no longer tried to land in `us-east1-d`
- however, pinning the root to `region=us-east1` still restricted it to east1 CPU
  executor capacity
- so this version still did not satisfy the stronger requirement of
  “CPU job can run wherever”

### Stage 4 - final successful placement shape: any CPU region for root, east1-d for children

Third launch after relaxing the root placement:

- terminated region-pinned root job:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d-childzone`
- submitted final placement variant:
  - `/ahmed/iris-rl-e4ms2-500-0328-ckptdbg-v6e8-e1d-anycpu`

Final command shape:

- root submit used:
  - **no** `--region`
  - **no** `--zone`
- experiment used:
  - `--region us-east1`
  - `--zone us-east1-d`
  - `--tpu-type v6e-8`
  - `--debug-checkpointer`
  - `--debug-checkpointer-dump-stacks-after 180`

This is the first launch shape that actually matches the user request:

- CPU coordinator may run anywhere Iris can place it
- RL TPU children are explicitly pinned to `us-east1-d`

Observed states:

- root coordinator:
  - `JOB_STATE_RUNNING`
- trainer child:
  - `JOB_STATE_PENDING` initially, then later `JOB_STATE_FAILED`
- rollout children:
  - initially `JOB_STATE_PENDING`
  - final state `JOB_STATE_SUCCEEDED`

Most important scheduler proof:

- trainer child resources showed:
  - TPU variant `v6e-8`
- rollout child resources also showed:
  - TPU variant `v6e-8`
- child pending reasons explicitly referenced:
  - scale group `tpu_v6e_8-us-east1-d`
- child-job scheduler constraints at that point were:
  - `device-type`
  - `device-variant`
  - `zone`
  - `region`
  - `reservation-job`

That is the key success criterion for the placement hack:

- the root coordinator no longer blocks on east1 CPU placement
- the expensive RL children really do target the correct TPU scale group in `us-east1-d`

### Stage 5 - actual remaining blocker: cross-region artifact paths, not placement

The final `anycpu` run did **not** fail on placement. It failed later in the
trainer bootstrap path with:

- `iris.marin_fs.TransferBudgetExceeded`
- specific path:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/model-00001-of-00004.safetensors`
- message:
  - transfer would exceed the shared 10 GB cross-region budget

Root cause:

- the experiment still hardcodes:
  - `MODEL_PATH = "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"`
  - `MARIN_PREFIX = "gs://marin-us-central1"`
- so even though the trainer and rollout workers are running in `us-east1-d`,
  they are still trying to:
  - read model weights from `us-central1`
  - write checkpoints / rollouts / other outputs under `us-central1`

This means the current east1 launch is only partially regionalized:

- compute placement is east1-d
- artifact paths are still us-central1

That mismatch is what triggered the transfer-budget failure.

### Verified storage fact: east1 copy already exists

To rule out “maybe east1 doesn’t have the model,” I explicitly checked GCS.

Verified present:

- `gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/...`
- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/...`

So the problem is **not** missing east1 model availability.
The problem is that this experiment is still pointed at the central1 copy.

### Current conclusion

What is proven:

- the child-zone placement hack works
- root CPU coordinator can be unpinned and run elsewhere
- trainer + rollout children can be explicitly pinned to `us-east1-d`
- Iris does in fact route those children to the `tpu_v6e_8-us-east1-d` scale group

What is still broken:

- `exp_iris_rl_regression_direct_gcs_prod.py` is not region-local for artifacts
- model bootstrap still points at `gs://marin-us-central1/...`
- output/checkpoint prefix still points at `gs://marin-us-central1/...`

What must change next for a truly valid east1-d run:

- derive `MODEL_PATH` from the selected region, e.g. `gs://marin-us-east1/...`
- derive `MARIN_PREFIX` from the selected region, e.g. `gs://marin-us-east1`
- relaunch only after both compute placement and artifact paths are east1-local

## 2026-03-28 23:13 PDT - rollout workers exiting cleanly on trainer retry is a control-plane bug

Observed on Claude's `v6e` run:

- root job:
  - `/ahmed/iris-rl-v6e-e5b-0328-v3`
- current Iris state:
  - root `RUNNING`
  - trainer child `RUNNING` with `failure_count=8`
  - rollout-0 `SUCCEEDED`
  - rollout-1 `SUCCEEDED`

That state is wrong for a 500-step RL run. The rollout workers should not
conclude the run is terminal while the trainer child is still retrying under
the same coordinator.

Root cause:

- `_train_worker_entry()` in `lib/marin/src/marin/rl/orchestration.py` was
  marking the shared lifecycle actor failed on any trainer *attempt* exception:
  - `runtime.run_state.mark_failed.remote("trainer crashed").result()`
- but the coordinator and `RLRunState` actor survive across Iris child-job
  retries
- rollout workers poll `RLRunState` and exit normally when they see:
  - `snapshot.status in ("completed", "failed")`
  - in `lib/marin/src/marin/rl/rollout_worker.py`

Observed live log evidence on `/ahmed/iris-rl-v6e-e5b-0328-v3`:

- `RL run marked as failed: trainer crashed`
- followed by:
  - `Run state is 'failed', stopping rollout worker`
  - for both rollout workers

This exactly explains why Iris showed the rollout children as `SUCCEEDED`:

- they were not being killed by the coordinator
- they were exiting normally after a false terminal signal from shared run state

Fix applied:

- removed the attempt-local `mark_failed()` call from `_train_worker_entry()`
- rationale:
  - an attempt-local trainer crash is not equivalent to whole-run terminal
    failure when Iris is still retrying the same trainer child job
  - rollout workers must remain alive across those retries

Validation:

- added regression test in `tests/rl/test_orchestration.py`
- command:
  - `uv run pytest -q tests/rl/test_orchestration.py`
- result:
  - `6 passed`

Debug note:

- `docs/debug-log-rollout-workers-exit-on-trainer-retry.md`

Operational consequence:

- already-submitted runs like `/ahmed/iris-rl-v6e-e5b-0328-v3` still carry the old bug
- they need relaunch to benefit from this fix

## 2026-03-28 23:36 PDT - v6e relaunch blocked by a second multi-host Arrow Flight bug

While preparing to relaunch the `v3` east jobs, I checked the terminal states of:

- `/ahmed/iris-rl-v6e-e1d-0328-v3`
- `/ahmed/iris-rl-v6e-e5b-0328-v3`

Both were already terminal, so there was nothing left to kill. The immediate reason
to avoid a blind relaunch was that the east5 trainer still had a second bootstrap
weight-transfer failure:

- `RuntimeError: Fetching value for jax.Array that spans non-addressable (non process local) devices is not possible...`

This narrowed the state of the Arrow Flight fix:

- Claude's earlier patch correctly fixed the collective deadlock by moving
  `copy_and_flatten(...)` outside the `process_index() == 0` block
- but `serve_weights()` still did a raw:
  - `jax.device_get(flat_dict)`
- on multi-host `v6e-16`, those flattened leaves can still be global arrays that are
  not fully addressable from one process

So the previous fix was necessary but incomplete.

### Patch applied

In `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`:

- added `_materialize_flat_leaf_for_transfer(...)`
- behavior:
  - fully-addressable leaves:
    - `np.asarray(leaf)`
  - non-fully-addressable leaves:
    - `jax.experimental.multihost_utils.process_allgather(leaf, tiled=True)`
- replaced:
  - `flat_dict = jax.device_get(flat_dict)`
- with:
  - `flat_dict = jax.tree.map(_materialize_flat_leaf_for_transfer, flat_dict)`

Why this is the correct narrower fix:

- it preserves the earlier "all processes must participate" invariant
- it stops trying to `device_get` a non-addressable global array directly
- it materializes each flattened tensor leaf to host memory in a way JAX explicitly
  supports for multi-process arrays

Debug note:

- [debug-log-v6e-multihost-weight-transfer-materialization.md](../../docs/debug-log-v6e-multihost-weight-transfer-materialization.md)

### Relaunch plan

With both `v3` jobs already terminal and both code fixes now local:

- rollout workers no longer exit permanently on trainer retry
- Arrow Flight no longer tries to raw-`device_get` non-addressable flattened
  arrays during bootstrap serve

the next step is to submit fresh `v4` runs in:

- `us-east1-d`
- `us-east5-b`

using the region-aware `experiments/xp_iris_rl_regression_direct_gcs_prod.py`
launcher so artifact paths remain region-local.

### Relaunch executed

There was nothing left to kill because both `v3` jobs were already terminal, so I
submitted fresh `v4` roots instead.

Submitted:

- `/ahmed/iris-rl-v6e-e1d-0328-v4`
- `/ahmed/iris-rl-v6e-e5b-0328-v4`

Stable run names:

- `iris-rl-v6e-e1d-v4`
- `iris-rl-v6e-e5b-v4`

Launch shape:

- root coordinator:
  - no explicit region/zone pin
- child TPU jobs:
  - `--region us-east1 --zone us-east1-d` for east1 run
  - `--region us-east5 --zone us-east5-b` for east5 run
- rollout TPU:
  - `v6e-8`
- trainer TPU:
  - `v6e-16`
- checkpoint debugger:
  - enabled
  - stack dumps after `180` seconds in one checkpoint phase

Exact submit commands:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-v6e-e1d-0328-v4 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/xp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-v6e-e1d-v4 --num-train-steps 500 --n-prompts 64 --eval-frequency 1 --num-rollout-workers 2 --region us-east1 --zone us-east1-d --tpu-type v6e-8 --train-tpu-type v6e-16 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`
- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-v6e-e5b-0328-v4 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/xp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-v6e-e5b-v4 --num-train-steps 500 --n-prompts 64 --eval-frequency 1 --num-rollout-workers 2 --region us-east5 --zone us-east5-b --tpu-type v6e-8 --train-tpu-type v6e-16 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`

Immediate post-submit state check:

- `/ahmed/iris-rl-v6e-e1d-0328-v4`
  - root `JOB_STATE_RUNNING`
  - `failure_count=0`
  - `preemption_count=0`
- `/ahmed/iris-rl-v6e-e5b-0328-v4`
  - root `JOB_STATE_RUNNING`
  - `failure_count=0`
  - `preemption_count=0`

## 2026-03-28 23:34 PDT - v6e-8 rollout workers now fail in hot-reload, not generic inference

The new `v4` runs answered the earlier question cleanly:

- the trainer-side multi-host Arrow Flight fix works
- the rollout "false `SUCCEEDED` while trainer retries" control-plane fix works
- but rollout workers on `v6e-8` are still not viable because they now fail
  during bootstrap weight application

This is a **different** failure from the old non-addressable trainer-side issue.

### What is now proven

On both:

- `/ahmed/iris-rl-v6e-e1d-0328-v4`
- `/ahmed/iris-rl-v6e-e5b-0328-v4`

the logs show the following sequence:

1. trainer successfully serves bootstrap weights for `weight_id -1`
2. rollout successfully receives all `291` params via Arrow Flight
3. rollout enters:
   - `reload_model: starting prefix cache reset`
   - `reload_model: converting state dict`
   - `reload_model: calling sync_weights`
4. rollout crashes inside TPU inference weight reload:
   - `jax.device_put(...)`
   - `x._multi_slice(...)`
   - `RESOURCE_EXHAUSTED`

So the failure is **not**:

- generic `v6e-8` inability to run Llama 8B inference
- Arrow Flight transport
- trainer-side weight serving

It is specifically rollout-side live weight reload on top of a running vLLM TPU
engine.

Detailed debug note:

- [debug-log-v6e-rollout-sync-oom.md](../../docs/debug-log-v6e-rollout-sync-oom.md)

### Important memory facts from the live logs

The high-signal evidence:

- rollout job type:
  - `v6e-8`
- current rollout inference config:
  - `tensor_parallel_size=4`
  - `gpu_memory_utilization=0.90`
- live vLLM cache init log shows:
  - mesh `model: 4`
  - about `28.12 / 31.25 GiB` HBM reserved per active chip
- right before the crash TPU inference logs:
  - processing `lm_head`
  - shape `(128256, 4096)`
  - dtype `bfloat16`
- the failing allocation is about:
  - `1002 MiB`
- allocator summary shows:
  - only about `1.17 GiB` free
  - largest contiguous block only about `448 MiB`

This strongly suggests fragmentation / temporary-buffer pressure during
`sync_weights()`, not a simple steady-state model-fit problem.

### Interpretation

The current rollout worker is using only 4 of the 8 available chips on a
`v6e-8` slice because TP is still fixed at `4`.

That means:

- half the chips are not helping absorb model state
- the active 4 chips are also carrying an aggressive KV-cache reservation at
  `gpu_memory_utilization=0.90`
- hot-reloading the bootstrap weights then needs a large temporary device
  allocation during `jax.device_put(...)->_multi_slice(...)`

That is why this does **not** conflict with "Llama 8B inference can run on
`v6e-8`":

- ordinary inference fit is a steady-state question
- RL hot-reload fit is a temporary-buffer + fragmentation question

### Not the first lever: rollout batch size

Reducing rollout batch size is not the first experiment here.

Reason:

- the crash happens at `step=0`
- it happens before the first generation batch is produced
- so `n_prompts` / `n_generations_per_prompt` do not explain this bootstrap OOM

### Secondary hypothesis: redundant bootstrap reload

There is also a plausible optimization opportunity:

- the rollout vLLM engine already boots from the checkpoint path on disk
- the trainer then immediately sends bootstrap weights for `weight_id -1`
- those are likely the same initial weights on a fresh run

So the system may be paying a full live `sync_weights()` cost before step `0`
for no actual model change.

This is **not** the primary hypothesis for long-term viability, because later
trainer steps would still require the same reload path. But if the cache/TP
experiments get close and still fail, skipping the redundant bootstrap reload is
a reasonable follow-up experiment.

### Launcher support added for the next probes

To avoid one-off edits for each sweep, `experiments/xp_iris_rl_regression_direct_gcs_prod.py`
now accepts:

- `--rollout-tensor-parallel-size`
- `--rollout-gpu-memory-utilization`
- `--rollout-max-model-len`

These feed directly into the rollout `vLLMInferenceContextConfig`.

### Next experiment ladder

Run short feasibility probes first:

- one rollout worker
- `20` train steps
- very infrequent eval, e.g. `--eval-frequency 1000`

Success criterion for each probe:

- rollout completes bootstrap weight sync
- rollout survives through first real rollout generation
- trainer receives initial rollouts instead of waiting forever at step `-1`

Proposed order:

1. `ROL-V6E8-001`: hold TP=`4`, lower `gpu_memory_utilization` to `0.60`
   - rationale:
     - directly targets the oversized cache reservation / fragmentation story
     - prior inference work showed cache budget can often be reduced without
       throughput loss
2. `ROL-V6E8-002`: set rollout TP=`8`, keep `gpu_memory_utilization=0.90`
   - rationale:
     - use all 8 chips on the `v6e-8` worker instead of only 4
     - test whether TP alone fixes the hot-reload memory profile
3. `ROL-V6E8-003`: rollout TP=`8` and `gpu_memory_utilization=0.60`
   - rationale:
     - most likely "make it work" configuration if either single knob is
       insufficient
4. `ROL-V6E8-004`: if needed, keep TP=`8`, lower `gpu_memory_utilization`, and
   reduce `rollout_max_model_len` below `2048`
   - rationale:
     - last-resort cache reduction knob if the live reload path still needs more
       contiguous HBM headroom

Candidate commands for the first three probes:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-v6e-e5b-rol-v6e8-001 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/xp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-v6e-e5b-rol-v6e8-001 --num-train-steps 20 --n-prompts 64 --eval-frequency 1000 --num-rollout-workers 1 --region us-east5 --zone us-east5-b --tpu-type v6e-8 --train-tpu-type v6e-16 --rollout-tensor-parallel-size 4 --rollout-gpu-memory-utilization 0.60 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`
- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-v6e-e5b-rol-v6e8-002 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/xp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-v6e-e5b-rol-v6e8-002 --num-train-steps 20 --n-prompts 64 --eval-frequency 1000 --num-rollout-workers 1 --region us-east5 --zone us-east5-b --tpu-type v6e-8 --train-tpu-type v6e-16 --rollout-tensor-parallel-size 8 --rollout-gpu-memory-utilization 0.90 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`
- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-v6e-e5b-rol-v6e8-003 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/xp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-v6e-e5b-rol-v6e8-003 --num-train-steps 20 --n-prompts 64 --eval-frequency 1000 --num-rollout-workers 1 --region us-east5 --zone us-east5-b --tpu-type v6e-8 --train-tpu-type v6e-16 --rollout-tensor-parallel-size 8 --rollout-gpu-memory-utilization 0.60 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`

## 2026-03-28 23:40 PDT - launching the first TP=8 rollout feasibility probe

User selected the TP=`8` branch first, so the next run is `ROL-V6E8-002`.

Purpose:

- test whether using all 8 chips on the `v6e-8` rollout worker is enough to
  avoid the bootstrap hot-reload OOM
- keep all other major knobs fixed so this isolates TP as the changed variable

Probe shape:

- region / zone:
  - `us-east5`
  - `us-east5-b`
- trainer:
  - `v6e-16`
- rollout:
  - `v6e-8`
  - `tensor_parallel_size=8`
  - `gpu_memory_utilization=0.90`
  - `max_model_len=2048`
- only `1` rollout worker
- only `20` train steps
- `eval_frequency=1000` to keep eval out of the way

Exact submit command:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-v6e-e5b-rol-v6e8-002 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/xp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-v6e-e5b-rol-v6e8-002 --num-train-steps 20 --n-prompts 64 --eval-frequency 1000 --num-rollout-workers 1 --region us-east5 --zone us-east5-b --tpu-type v6e-8 --train-tpu-type v6e-16 --rollout-tensor-parallel-size 8 --rollout-gpu-memory-utilization 0.90 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`

Submission result:

- root job:
  - `/ahmed/iris-rl-v6e-e5b-rol-v6e8-002`
- immediate state check:
  - root `JOB_STATE_RUNNING`
  - `failure_count=0`
  - `preemption_count=0`
  - task state counts:
    - `building=1`

Outcome:

- root job:
  - `JOB_STATE_FAILED`
- rollout child:
  - `JOB_STATE_FAILED`
  - `failure_count=4`
- trainer child:
  - `JOB_STATE_KILLED`
  - collateral shutdown after rollout failure

What happened:

- trainer bootstrap serve still succeeded
- rollout still received bootstrap weights for `weight_id -1`
- rollout still died in:
  - `reload_model -> sync_weights -> jax.device_put -> _multi_slice`
- failure signature was materially the same as TP=`4`:
  - `RESOURCE_EXHAUSTED`
  - attempted allocation about `1002 MiB`

Important negative result:

- changing rollout TP from `4` to `8` **did not** by itself fix the bootstrap
  hot-reload OOM on `v6e-8`

Allocator detail from the TP=`8` run:

- free memory still about `1.17 GiB`
- largest contiguous free block improved only modestly:
  - about `541 MiB` vs about `448 MiB` before
- still not enough to satisfy the ~`1002 MiB` temporary allocation

Interpretation:

- TP=`8` alone is not the primary lever
- the peak allocation seems dominated by the live reload path itself, not just
  by how the final model state is sharded
- the next stronger probe should pair TP=`8` with lower cache reservation

Next concrete move:

- run `ROL-V6E8-003`:
  - TP=`8`
  - `gpu_memory_utilization=0.60`

## 2026-03-28 23:58 PDT - launching `ROL-V6E8-003` (`TP=8`, `gpu_memory_utilization=0.60`)

`ROL-V6E8-002` established that TP=`8` alone is not sufficient.

So the next probe keeps the same rollout TP and changes only the cache-pressure
knob:

- rollout TP:
  - `8`
- rollout `gpu_memory_utilization`:
  - `0.60`

Everything else stays in the short feasibility shape:

- region / zone:
  - `us-east5`
  - `us-east5-b`
- trainer:
  - `v6e-16`
- rollout:
  - `v6e-8`
- one rollout worker
- `20` train steps
- `eval_frequency=1000`

Exact submit command:

- `uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait --user ahmed --job-name iris-rl-v6e-e5b-rol-v6e8-003 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- uv run python experiments/xp_iris_rl_regression_direct_gcs_prod.py --run-name iris-rl-v6e-e5b-rol-v6e8-003 --num-train-steps 20 --n-prompts 64 --eval-frequency 1000 --num-rollout-workers 1 --region us-east5 --zone us-east5-b --tpu-type v6e-8 --train-tpu-type v6e-16 --rollout-tensor-parallel-size 8 --rollout-gpu-memory-utilization 0.60 --debug-checkpointer --debug-checkpointer-dump-stacks-after 180`

## 2026-03-29 00:21 PDT - `ROL-V6E8-003` is alive; rollout bootstrap is fixed; trainer weight export is on the HBM cliff

Current live controller state for `/ahmed/iris-rl-v6e-e5b-rol-v6e8-003`:

- root job:
  - `JOB_STATE_RUNNING`
  - `failure_count=0`
- trainer child:
  - `JOB_STATE_RUNNING`
  - `failure_count=0`
- rollout child:
  - `JOB_STATE_RUNNING`
  - `failure_count=0`

So this probe did **not** reproduce the earlier fatal rollout bootstrap failure.

### What is now clearly fixed

Lowering rollout `gpu_memory_utilization` to `0.60` while keeping TP=`8`
appears to have fixed the rollout-side bootstrap hot-reload OOM.

Evidence:

- rollout received bootstrap weights for `weight_id -1`
- rollout finished:
  - `reload_model: sync_weights done in 7.3s`
  - `First weight transfer complete, inference can proceed`
- rollout advanced into real generation:
  - `PHASE: GENERATE step=0 lesson=math_full`

### Real RL progress observed

This run is now past bootstrap and into real RL activity.

Observed trainer progress:

- initial rollouts arrived after about `275s`
- trainer step `0` completed with real work:
  - `fwd_bwd=56.36s`
- trainer step `1` completed with real work:
  - `fwd_bwd=17.14s`
- trainer step `3` also completed with real work:
  - `fwd_bwd=32.83s`

Observed rollout progress:

- generated `500` groups from `math_full` at trainer step `0`
- later generated additional `64`-group rollout batches for later trainer steps
- latest visible phase was:
  - `PHASE: GENERATE step=6 lesson=math_full`

### The new issue: trainer-side allocator pressure during weight export

The run is still logging repeated TPU allocator failures during the trainer's
per-step weight-transfer hook.

These happen in the Arrow Flight export path:

- trainer hook:
  - [train_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/train_worker.py#L479)
- export path:
  - [arrow_flight.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/weight_transfer/arrow_flight.py#L344)
  - [arrow_flight.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/weight_transfer/arrow_flight.py#L455)

What that code does:

- casts the current training model to bf16 on TPU
- flattens each parameter
- materializes flattened leaves to host for Arrow Flight serving

Because `sync_interval_steps=1`, this export path runs **every training step**.

### Exact HBM math

The allocator warnings line up exactly with specific Llama 3.1 8B bf16 tensor
sizes.

Exact bf16 sizes:

- MLP `up_proj` / `gate_proj` / `down_proj`:
  - `14336 x 4096 x 2 bytes = 117,440,512 bytes = 112.00 MiB`
- attention `q_proj` / `o_proj`:
  - `32 x 128 x 4096 x 2 bytes = 33,554,432 bytes = 32.00 MiB`
- `lm_head`:
  - `128256 x 4096 x 2 bytes = 1,050,673,152 bytes = 1002.00 MiB`

These match the live TPU allocator errors exactly:

- `Attempting to allocate 112.00M`
- `Attempting to allocate 32.00M`

Approximate total bf16 parameter footprint for Llama 3.1 8B:

- one transformer layer:
  - about `416 MiB`
- `32` layers:
  - about `13.0 GiB`
- embeddings + `lm_head`:
  - about `1.96 GiB`
- total:
  - about `14.96 GiB`

Interpretation:

- the model clearly fits well enough for steady-state training and rollout
- the trainer's **live export path** is what is operating at the memory cliff
- this is compatible with "8B runs on v6e"; the problem is not static model fit,
  it is temporary/export-buffer pressure during per-step weight transfer

### Important nuance: these allocator failures are real, but not currently fatal

The trainer still successfully served weights after the allocator failures.

Confirmed successful transfers:

- bootstrap:
  - `weight_id -1`
- later per-step transfers:
  - `weight_id 0`
  - `weight_id 1`
  - `weight_id 2`
  - `weight_id 3`
  - `weight_id 4`
  - `weight_id 5`

So this is **not** currently "the job died from OOM."

The more precise statement is:

- TPU allocator warnings are real
- trainer export is right on the HBM edge
- the job is still alive and continuing

### More concerning than the allocator warnings: some "completed" trainer steps do not look real

Not every logged completed step looks trustworthy.

Trainer step breakdown so far:

- step `0`:
  - `fwd_bwd=56.36s`
  - looks like a normal real step
- step `1`:
  - `fwd_bwd=17.14s`
  - looks like a normal real step
- step `2`:
  - `fwd_bwd=0.00s`
  - suspicious
- step `3`:
  - `fwd_bwd=32.83s`
  - looks real again
- step `4`:
  - `fwd_bwd=0.00s`
  - suspicious
- step `5`:
  - `fwd_bwd=0.00s`
  - suspicious

So the current state is:

- rollout bootstrap on `v6e-8` is fixed
- trainer-side per-step weight export is memory-tight
- the run is alive
- but some later "completed" trainer steps are not yet credible as full
  forward/backward optimizer steps

### Updated interpretation

The earlier question was "did we hit OOM?"

Refined answer:

- yes, we are seeing **real TPU allocator exhaustion events**
- no, they are **not** currently killing the job
- the more important unresolved bug is why some steps still increment while
  reporting `fwd_bwd=0.00s`

That step-integrity issue now looks like the next thing to debug.

## 2026-03-29 00:29 PDT - MAJOR: RL trainer `fwd_bwd` timing metric is invalid

The suspicious `fwd_bwd=0.00s` values do **not** currently look like proof that
the trainer skipped forward/backward work.

After tracing the code and the live run, the stronger conclusion is:

- this is a **MAJOR instrumentation bug**
- the current `fwd_bwd` metric is computed from incompatible timing windows
- the zeroes are therefore misleading

### Root cause

Current implementation:

- [trainer.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/levanter/src/levanter/trainer.py#L524)
  measures data loading in `training_steps()` via:
  - `with capture_time() as loading_time: example = next(iter_data)`
- [trainer.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/levanter/src/levanter/trainer.py#L484)
  measures `StepInfo.step_duration` **inside** `train_step()`
- [train_worker.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/marin/src/marin/rl/train_worker.py#L417)
  then does:
  - `forward_backward_duration = max(0.0, info.step_duration - batch_prep_time)`

Why this is wrong:

- `batch_prep_time` is measured **before** `train_step()` begins
- `info.step_duration` is measured **inside** `train_step()`
- these are disjoint sequential intervals, not a parent interval and child
  interval
- subtracting them is mathematically invalid

There is a second bug hiding inside the same metric:

- the so-called `batch_prep_time` is not just "batch prep"
- it includes the blocking wait for new rollouts from the replay buffer:
  - fetch
  - batch creation
  - sharding
- so it is partly rollout wait time, not purely data transformation time

### Why the live logs looked so strange

The suspicious steps are exactly the ones where the data-loader wait time
exceeded the train-step compute time.

Examples from the live `ROL-V6E8-003` run:

- step `2`:
  - `duration=17.28s`
  - `batch_prep=41.26s`
  - current code clamps `17.28 - 41.26` to `0.00`
- step `4`:
  - `duration=17.29s`
  - `batch_prep=45.56s`
  - again clamps to `0.00`
- step `5`:
  - `duration=22.93s`
  - `batch_prep=24.88s`
  - again clamps to `0.00`

Those zeroes are therefore expected from the buggy formula, not proof of an
empty optimizer step.

### Strong evidence this is metric corruption, not immediate trainer non-progress

The run continues to make forward progress after the suspicious zeroes:

- trainer served:
  - `weight_id 2`
  - `weight_id 3`
  - `weight_id 4`
  - `weight_id 5`
  - `weight_id 6`
- trainer loss kept changing across steps
- rollout kept advancing to later generate phases
- later step `6` logged:
  - `duration=22.92s`
  - `batch_prep=20.68s`
  - `fwd_bwd=2.24s`

That pattern is much more consistent with a bad decomposition than with the
trainer alternately doing and not doing optimization steps.

### Impact

This is a major RL observability bug because it can cause a healthy-but-slow or
rollout-starved run to look like it is silently skipping training.

Affected signals:

- trainer log line:
  - `Training step ... (batch_prep=..., fwd_bwd=...)`
- W&B metric:
  - `throughput/forward_backward_duration_seconds`

### Correct semantic fix

The timing metrics should be split into non-overlapping phases:

- trainer step compute duration:
  - just `info.step_duration`
- rollout wait duration:
  - time blocked waiting for rollouts / fetching batch inputs
- batch create duration:
  - CPU batch construction
- batch shard duration:
  - host-to-device / sharding prep
- end-to-end iteration duration:
  - `rollout_wait + batch_create + batch_shard + train_step`

So the old `fwd_bwd = step_duration - batch_prep` logic should be removed.

### Current status

This bug now has priority `MAJOR`.

Reason:

- it directly distorts interpretation of live RL progress on Iris
- it created a false alarm that the trainer might be incrementing steps without
  doing work
- it obscures the real distinction between:
  - rollout starvation / input wait
  - trainer compute time
  - trainer weight-transfer overhead

Planned next action:

- patch RL timing metrics so they log separate phase durations with correct
  semantics
- stop reporting the invalid subtraction-based `fwd_bwd` metric
