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
