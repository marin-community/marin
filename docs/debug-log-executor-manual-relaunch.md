# Debugging log for executor manual relaunch resume

Investigate why manually killing and relaunching the executor-backed RL job with
the same stable run name resumes W&B identity and reuses the same GCS output
path, but does not resume actual RL training progress.

This fix must preserve the already-working retry/preemption behavior from the
validated direct `500/500` run. The target is one correct resume story for both:

- manual kill-and-relaunch with the same stable run name
- automatic retry after preemption within a running Iris job

## Initial status

Observed on March 30, 2026 for the executor-backed Math500 run:

- Original logical run name:
  `llama-3.1-8bi-math500-exec-20260331-011313`
- Killed root job:
  `/ahmed/iris-rl-math500-exec-20260330-181209`
- Relaunched root job:
  `/ahmed/iris-rl-math500-exec-resume-20260330-210056`

The relaunch was intentionally started with the same stable `--run-name`.

What clearly worked:

- trainer W&B resumed the original run id
  `llama-3.1-8bi-math500-exec-20260331-011313-train`
- rollout W&B resumed the original run ids
  `...-rollout-0` and `...-rollout-1`
- the executor reused the same step key
  `rl_testing/llama-3.1-8bi-math500-exec-20260331-011313_4160fdff`
- the executor reused the same output path
  `gs://marin-us-central1/rl_testing/llama-3.1-8bi-math500-exec-20260331-011313-8e0fad`
- trainer checkpoints exist at
  `gs://.../checkpoints/llama-3.1-8bi-math500-exec-20260331-011313-train/step-{6,12,...,68}`

What did not work:

- the trainer reopened the old rollout-storage path but behaved like a fresh RL
  run
- trainer logs showed stale rollout filtering with `current_step=0`
- trainer later served weights with `weight_id -1`
- trainer then waited for initial rollouts from step `-1`
- rollout workers resumed W&B, reopened the old rollout-storage path, and
  polled Arrow Flight for new weights, but no new rollouts reached the trainer

So the current bug is not "stable run identity." That part works. The bug is
"manual kill-and-relaunch does not restore the RL state machine."

## Correctness requirement

Any fix here must satisfy both resume modes:

- **Manual relaunch:** a user can kill a run, relaunch with the same stable
  run name, and continue from the latest committed trainer checkpoint with the
  same W&B lineage.
- **Preemption retry:** the existing in-run resume behavior must continue to
  work. We already have evidence from the clean direct `500/500` run that
  automatic retry after preemption can recover from the latest checkpoint.

So the design constraint is not "make manual relaunch special." It is:

- define a single resume path that is correct whenever a committed trainer
  checkpoint already exists for the stable run id
- avoid regressing the already-working preemption/retry path while doing that

## Hypothesis 1 - trainer resume is immediately overridden by unconditional bootstrap logic

The strongest code-level suspicion is in `TrainWorker.train()`:

- `trainer.initial_state(...)` may load a valid trainer checkpoint
- but RL then unconditionally does all of the fresh-run bootstrap work anyway:
  - `self.transfer_server.serve_weights(-1, state.model)`
  - `self.replay_buffer.set_current_step(-1)`
  - `self._wait_for_initial_rollouts()`

That means a resumed trainer can still be forced back into "bootstrap rollout"
semantics even if `state.step > 0`.

### Changes to make

No code changes yet. First confirm whether:

- `trainer.initial_state(...)` is actually loading the trainer checkpoint on the
  resumed run
- if it is, whether `state.step` is ignored by the subsequent RL bootstrap path
- whether the same bootstrap path is currently harmless for preemption retries
  only because of surrounding timing/state assumptions

Files to inspect first:

- `lib/marin/src/marin/rl/train_worker.py`
- `lib/levanter/src/levanter/trainer.py`
- `lib/levanter/src/levanter/checkpoint.py`

### Results

Initial trace supports this hypothesis:

- `Trainer.initial_state()` in Levanter will load from
  `checkpointer.expanded_path(trainer.run_id)` when a checkpoint exists
- trainer checkpoints really are stored under
  `checkpoints/<run_id>-train/step-*`
- but RL bootstrap in `TrainWorker.train()` is unconditional and does not branch
  on resumed `state.step`

This likely explains why relaunch gets the same model/output identity while
still re-entering "serve step -1, wait for initial rollouts" behavior.

Non-regression requirement for this hypothesis:

- if bootstrap behavior is made conditional on resumed `state.step`, the same
  branch must still handle automatic post-preemption retry correctly
- the fix should key off resumed trainer state / discovered checkpoint state,
  not off "manual relaunch" as a separate mode

## Hypothesis 2 - replay-loader startup reuses old rollout storage before RL resume state is re-established

The replay loader is started in the `with self.replay_loader` context before the
trainer finishes resume/bootstrapping decisions.

Because `ReplayBuffer._current_step` defaults to `0`, old rollout batches from
the previous attempt are immediately read from GCS and compared against
`current_step=0`, which produces the observed stale-rollout spam.

### Changes to make

No code changes yet. Confirm startup ordering:

- when the replay-loader background thread starts
- when `_current_step` is first updated
- whether resumed state should seed replay-buffer current step before the
  background reader is allowed to ingest old rollouts
- whether the same seeding behavior is also needed for automatic retry after
  preemption

Files to inspect:

- `lib/marin/src/marin/rl/replay_buffer.py`
- `lib/marin/src/marin/rl/train_worker.py`

### Results

Confirmed:

- `ReplayDataLoader.__enter__()` starts a background thread immediately
- `ReplayBuffer._current_step` defaults to `0`
- stale rollout filtering happens inside `ReplayBuffer.add_batches(...)`
- RL does not update replay-buffer current step until after the trainer state is
  already created, and later forcibly resets it to `-1`

So even if trainer checkpoint restore is working, the replay buffer is not
being seeded with resumed trainer step before old rollout storage is consumed.

Non-regression requirement for this hypothesis:

- replay-buffer seeding and rollout-storage reuse rules must remain compatible
  with the preemption-retry path that already worked on the direct baseline

## Hypothesis 3 - rollout startup may have an independent vLLM compile/runtime issue

Rollout logs also show repeated TPU vLLM scoped-Vmem `INVALID_ARGUMENT` errors
around `ragged_paged_attention` during startup.

This may be:

- a separate rollout startup issue exposed by the relaunch
- or just noisy compiler/runtime diagnostics that do not actually prevent the
  rollout process from continuing

It is probably secondary, because the primary trainer-side resume logic already
looks wrong, but it should be kept in mind while validating any fix.

### Changes to make

- keep watching rollout logs after the trainer-side resume logic is understood
- do not blame rollout startup first unless trainer resume is proven correct

## Hypothesis 4 - resume should republish the recovered trainer step before replay ingestion starts

The cleanest unified resume rule is:

- fresh run with no trainer checkpoint:
  - serve bootstrap weights at `-1`
  - wait for bootstrap rollouts
- resumed run with trainer checkpoint at step `N > 0`:
  - seed replay-buffer current step to `N`
  - republish shared run-state train step `N`
  - serve startup weights at `N`
  - wait for rollout batches stamped with `weight_step=N`

That keeps manual relaunch and preemption retry on the same discovered-state
path instead of introducing separate resume modes.

### Changes made

Code changes in `lib/marin/src/marin/rl/train_worker.py`:

- added `InitialRolloutState`
- added `_initial_rollout_state(train_step)`
- added `_seed_initial_rollout_state(...)`
- `TrainWorker.train()` now:
  - calls `trainer.initial_state(...)`
  - logs recovered `state.step`
  - derives startup rollout semantics from that recovered step
  - seeds replay-buffer/shared run-state before replay-loader ingestion starts
  - only uses bootstrap `-1` semantics for true fresh runs (`state.step == 0`)

Added unit coverage in `tests/rl/test_train_worker.py` for:

- fresh-run bootstrap semantics
- resumed-run startup semantics
- replay-buffer seeding
- shared run-state publication for resumed runs

Validation run:

- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/train_worker.py tests/rl/test_train_worker.py docs/debug-log-executor-manual-relaunch.md`
- `uv run pytest -q tests/rl/test_train_worker.py tests/rl/test_rl_experiment_utils.py`

### Results

The trainer-side bug is now concretely fixed in code:

- resumed trainer state is no longer immediately overridden by
  unconditional bootstrap `-1` logic
- replay-buffer current step is seeded before replay ingestion starts
- the design is shared between manual relaunch and preemption retry

Operational validation status:

- killed the broken relaunch root job:
  `/ahmed/iris-rl-math500-exec-resume-20260330-210056`
- submitted a new validation relaunch with the same stable run name:
  - root:
    `/ahmed/iris-rl-math500-exec-resume2-20260330-212944`
  - stable run name:
    `llama-3.1-8bi-math500-exec-20260331-011313`

The new validation relaunch is structurally healthy:

- root job, executor step, nested RL coordinator, trainer, and both rollout
  children are `RUNNING`
- the trainer and both rollout children were submitted under the same stable
  executor step key / GCS output path as the original logical run

Live validation now confirms the resume path is working:

- trainer discovered and loaded the existing checkpoint:
  - `Discovered latest checkpoint ... step-62`
  - `Loading checkpoint ... step-62`
- trainer resumed with the recovered trainer step instead of resetting to a
  fresh-run bootstrap state:
  - `Trainer recovered state.step=63; startup rollout weight_step=63`
- replay ingestion was seeded with the resumed step before old rollout storage
  was consumed:
  - old rollout batches were filtered against `current_step=63`, not
    `current_step=0`

## Hypothesis 5 - rollout W&B tracker step was only resume-safe within one coordinator lifetime

After the trainer-side relaunch fix, rollout W&B still emitted warnings like:

- `Tried to log to step 7 that is less than the current step 78`

This happened on the resumed rollout workers, which proves the previous
"resume-safe" tracker-step design was only safe across worker retries that
shared the same coordinator actor. It was **not** safe across manual relaunch,
because the new coordinator created a fresh `RLRunState` actor and the rollout
tracker counter restarted from zero.

### Changes made

Code changes:

- removed rollout tracker-step storage from
  `lib/marin/src/marin/rl/run_state.py`
- changed `lib/marin/src/marin/rl/rollout_worker.py` so each rollout worker now
  keeps a local monotonic tracker step
- the local tracker step is lazily anchored on recovered
  `max(current_train_step, current_weight_step, 0) * 1000`
- subsequent rollout/eval logs increment that local counter by `1`

Why this design is better:

- each rollout worker already has its own W&B run id, so cross-worker
  coordination is unnecessary
- the step no longer depends on ephemeral coordinator memory
- the first log after a resumed run starts well above old rollout tracker
  values, so manual relaunch and preemption retry both stay monotonic

Tests updated:

- removed the old `RLRunState` rollout-tracker-step unit test
- updated rollout-worker tests to validate resume-safe anchoring off recovered
  train/weight state

Validation:

- `uv run pytest -q tests/rl/test_run_state.py tests/rl/test_rollout_worker.py tests/rl/test_train_worker.py tests/rl/test_rl_experiment_utils.py`
- `52 passed`
- trainer accepted initial rollouts for the resumed startup step:
  - `Received initial rollouts for step 63! Buffer size: 4096 (waited 5.0s)`
- trainer then continued normal training progress:
  - `Progress on:train 64.0it/500it`
  - `Transferring weights at step 63`
  - `Training step 63 completed`
  - `Progress on:train 65.0it/500it`
  - `Transferring weights at step 64`
  - `Training step 64 completed`
- rollout workers also resumed the same weight lineage:
  - `Received new weights from step 63`
  - `First weight transfer complete, inference can proceed`

So manual kill-and-relaunch with the same stable run name now resumes the same:

- W&B run ids
- executor output path / checkpoint lineage
- actual RL training state machine

## Follow-up Notes

- The core manual-relaunch resume bug is fixed.
- The same fix path is intentionally shared with automatic retry/preemption
  resume, since both now derive startup rollout semantics from recovered
  trainer state.
- There is still some rollout-side startup lag while vLLM applies the first
  resumed weight update, but this is not the original resume correctness bug:
  rollout eventually receives the resumed step and proceeds.
- A future integration test for real kill-and-relaunch behavior would still be
  useful, but it is no longer needed to prove the basic correctness fix.

## Hypothesis 6 - the rollout W&B monotonicity fix preserved ordering by exploding the W&B step axis

After manual-relaunch resume was fixed, a new regression showed up in the live
executor run:

- rollout workers no longer emitted `step=-1` warnings
- but rollout W&B logs started using giant synthetic step values such as
  `tracker_step=78000` for `weight_step=78`
- this made the rollout W&B charts effectively unusable and also interacted
  badly with the already-confusing resume state in the UI

The source was our first rollout-worker fix:

- we stopped using `weight_step` directly as the W&B `step`
- but then we replaced it with a synthetic anchor
  `max(train_step, weight_step, 0) * 1000`
- that preserved monotonicity, but it did so by forcing the W&B x-axis to jump
  from normal rollout-scale numbers into huge synthetic values

That is not acceptable. The invariant we actually need is:

- W&B step must be monotonic across resume
- W&B step must remain a normal run-local counter
- `inference.weight_step` / `inference.train_step` should remain logged as
  ordinary metrics for semantic alignment

### Changes made

Code changes in `lib/marin/src/marin/rl/rollout_worker.py`:

- removed `_ROLLOUT_TRACKER_RESUME_STRIDE`
- removed `_initial_rollout_tracker_step(...)`
- removed the worker-local synthetic tracker-step counter
- rollout/eval metrics now call `tracker.log(...)` without an explicit `step=...`
- rollout logs still record:
  - `inference.weight_step`
  - `inference.train_step`

This returns rollout W&B stepping to W&B's own resumed run counter, which is
the correct place to solve monotonicity for a resumed run id.

Tests updated in `tests/rl/test_rollout_worker.py`:

- rollout metric logging now asserts `step=None`
- lesson eval logging now asserts `step=None`
- removed the synthetic tracker-step unit test

### Results

Local validation:

- `uv run pytest -q tests/rl/test_rollout_worker.py`
- `24 passed`
- `uv run python -m py_compile lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_rollout_worker.py`

Important note:

- the currently running executor resume experiment already has polluted rollout
  W&B history from the bad synthetic-step version
- so this code fix removes the regression going forward, but it does not erase
  the already-written bad W&B points from the live run

## 2026-03-31 - short kill-and-relaunch smoke validation is currently blocked by Iris launch instability

I tried to follow up the code fixes with the most useful operational check:

- launch a **small** RL run
- wait for the first checkpoint
- kill it
- relaunch with the same stable run name
- verify checkpoint + W&B resume

That test is still the right test. The problem today is that Iris launch
stability is not good enough to get a clean small smoke run through startup.

### Attempt A: executor smoke on production cluster

Root job:

- `/ahmed/iris-rl-math500-exec-manualresume-smoke1-20260331-0006`

What happened:

- outer executor root launched
- first nested child-job launch timed out at the controller
- retry then hit `UNIQUE constraint failed: jobs.job_id`

So this did not fail inside trainer/rollout resume logic. It failed while
trying to create the smoke-run subtree itself.

### Attempt B: second executor smoke on production cluster

Root job:

- `/ahmed/iris-rl-math500-exec-manualresume-smoke2-20260331-0006`

What happened:

- this one got farther than attempt A
- the nested RL coordinator really did start:
  `RL coordinator starting for run llama-3.1-8bi-math500-exec-manualresume-smoke-20260331-0006`
- but even after that, trainer/rollout children never produced W&B runs
  within the observation window

So this attempt also got stuck in launch/control-plane territory before the
actual kill-and-relaunch validation could happen.

### Attempt C: executor smoke on `marin-dev`

Root job:

- `/ahmed/iris-rl-math500-exec-manualresume-dev1-20260331-0026`

What happened:

- the root submitted cleanly against `marin-dev`
- the root then failed quickly with:
  `RuntimeError: 1 step(s) failed`
- descendant executor-step job:
  `/ahmed/iris-rl-math500-exec-manualresume-dev1-20260331-0026/rl_testing-llama-3.1-8bi-math500-exec-manualresume-smoke-dev-20260331-0026_bcb24671-2678ed9f`

This gave a faster failure signal than production, but it still did not reach a
usable trainer/rollout checkpointed run.

### Attempt D: direct smoke on production cluster

Root job:

- `/ahmed/iris-rl-direct-manualresume-smoke1-20260331-0035`

Rationale:

- the trainer manual-relaunch fix and rollout W&B fix are shared with direct RL
- direct RL is operationally healthier than executor-backed smoke launches

At the time of writing this note:

- the direct smoke root had been submitted
- but controller responsiveness was still poor enough that even routine status
  checks were slow
- the run had not yet reached visible trainer/rollout W&B startup

### Current conclusion

The code fixes are no longer the clearest blocker. The current blocker is Iris
launch/control-plane instability during nested or even small RL smoke startup.

So the right interpretation right now is:

- manual-relaunch logic is still worth validating with a small smoke run
- but a clean kill-and-relaunch proof has **not** yet been obtained in this
  session
- the thing blocking that proof is control-plane reliability, not a new clear
  trainer/rollout resume failure
