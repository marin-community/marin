# Debugging log for rollout workers exiting on trainer retry

Rollout workers in `/ahmed/iris-rl-v6e-e5b-0328-v3` were showing `JOB_STATE_SUCCEEDED`
even though the run was supposed to continue for 500 RL steps and the trainer child
was still retrying. The goal here is to explain that state transition and patch the
control-plane bug that caused it.

## Initial status

Observed on March 28, 2026:

- root job `/ahmed/iris-rl-v6e-e5b-0328-v3` still `RUNNING`
- trainer child still `RUNNING`, but with `failure_count=8`
- both rollout children already `JOB_STATE_SUCCEEDED`

That combination is wrong for a healthy RL run. If the trainer is still retrying under
the same coordinator, rollout workers should not decide the run is terminal and exit
cleanly.

## Hypothesis 1

The shared RL lifecycle actor is being marked failed on a trainer *attempt* crash,
not on a trainer *job* terminal failure. Because the coordinator and lifecycle actor
outlive trainer task attempts, rollout workers may see a stale terminal state and
exit permanently while Iris is still retrying the trainer child.

## Changes to make

Inspect:

- `lib/marin/src/marin/rl/orchestration.py`
- `lib/marin/src/marin/rl/run_state.py`
- `lib/marin/src/marin/rl/rollout_worker.py`
- live Iris logs for `/ahmed/iris-rl-v6e-e5b-0328-v3`

Patch:

- stop calling `runtime.run_state.mark_failed("trainer crashed")` from
  `_train_worker_entry()` on attempt-local exceptions
- add a regression test proving trainer-attempt crashes do not flip shared run state

## Results

The bug is confirmed.

Evidence from code:

- trainer attempt entrypoint caught all exceptions and called:
  - `runtime.run_state.mark_failed.remote("trainer crashed").result()`
  - in `lib/marin/src/marin/rl/orchestration.py`
- rollout workers poll the shared lifecycle actor and exit normally when they see:
  - `snapshot.status in ("completed", "failed")`
  - in `lib/marin/src/marin/rl/rollout_worker.py`

Evidence from live logs on `/ahmed/iris-rl-v6e-e5b-0328-v3`:

- `RL run marked as failed: trainer crashed`
- then:
  - `Run state is 'failed', stopping rollout worker`
  - once for each rollout worker

That explains the dashboard state exactly:

- rollout children returned normally, so Iris marked them `SUCCEEDED`
- trainer child kept retrying under the same coordinator
- the shared run state stayed terminal, so rollout workers never came back

Patch applied:

- removed the attempt-local `mark_failed()` call from `_train_worker_entry()`
- added a regression test in `tests/rl/test_orchestration.py`

Validation:

- `uv run pytest -q tests/rl/test_orchestration.py`
- result: `6 passed`

## Future Work

- [ ] Decide whether the coordinator should mark run state failed only after a child job exhausts retries and becomes terminal
- [ ] Relaunch the affected v6e runs after this patch, since already-submitted jobs still carry the old behavior
