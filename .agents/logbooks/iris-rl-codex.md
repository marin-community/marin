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
