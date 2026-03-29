# Debugging log for iris-rl-e4ms2-500 final failure

Investigate why `/ahmed/iris-rl-e4ms2-500-0327-v2` reached the high-460s in trainer steps and then terminated instead of auto-resuming again.

## Initial status

- Root job: `/ahmed/iris-rl-e4ms2-500-0327-v2`
- Trainer W&B: `marin-community/marin_iris_rl_debug/iris-rl-e4ms2-500-train`
- User report: run reached `~470` steps and then died; earlier preemptions and errors had resumed successfully.

Evidence collected:
- `iris job list --prefix /ahmed/iris-rl-e4ms2-500-0327-v2 --json`
- `iris job bug-report` for root, trainer, and rollout child jobs
- `iris job logs` on trainer near the terminal window
- `iris query` against `tasks` and `task_attempts`
- W&B API reads for trainer + rollout runs

## Hypothesis 1

The rollout workers failed first, starving the trainer and causing the run to die.

## Results

Rejected.

The root coordinator failed because the trainer child failed:

- root error: `fray.v2.client.JobFailed: Job ...-train finished with status failed`
- trainer child state: `failed`
- rollout children state: `killed`

Rollout evidence shows they were collateral shutdown, not the first failure source:

- rollout-0 task: `failure_count=0`, `preemption_count=1`, final state `killed`
- rollout-1 task: `failure_count=0`, `preemption_count=0`, final state `killed`
- both rollout tasks finished at `2026-03-28T22:06:25Z`, about `24s` after the trainer fatal

That is consistent with the coordinator failing on trainer failure and Iris cascading cleanup killing descendants.

## Hypothesis 2

The trainer hit one more recoverable TPU/runtime failure, but this time the retry budget was exhausted.

## Results

Confirmed.

Trainer task row from Iris:

- `max_retries_failure = 3`
- `max_retries_preemption = 100`
- `failure_count = 4`
- `preemption_count = 1`

Trainer attempt history:

1. `attempt_id=0` -> `failed`
   - error: `RPC: /tensorflow.CoordinationService/PollForError`
2. `attempt_id=1` -> `worker_failed`
   - error: worker timeout
3. `attempt_id=2` -> `failed`
   - error: `RPC: /tensorflow.CoordinationService/PollForError`
4. `attempt_id=3` -> `failed`
   - error: `RPC: /tensorflow.CoordinationService/PollForError`
5. `attempt_id=4` -> `failed`
   - error: `RPC: /tensorflow.CoordinationService/PollForError`

Interpretation:

- the preemption/worker-failure budget was not the limiter
- the run already consumed its ordinary failure retries
- the last trainer failure was the fourth non-preemption failure on a task configured with `max_retries_failure=3`
- once that last failure happened, Iris stopped retrying the trainer, the coordinator failed, and the rollout children were killed

This explains why earlier incidents resumed but the last one did not.

## Hypothesis 3

The terminal trainer error was a normal Python exception inside the RL code path.

## Results

Not supported by the evidence collected.

The terminal trainer log window ends with a TPU/JAX coordination failure, not a surfaced Python traceback:

- `Use error polling to propagate the following error to all tasks: UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats): /job:jax_worker/replica:0/task:0`
- `Terminating process because the JAX distributed service detected fatal errors`
- final error surfaced by Iris: `RPC: /tensorflow.CoordinationService/PollForError`

The last normal trainer telemetry before that looked healthy:

- W&B trainer summary reached `global_step = 469`
- recent steps still had normal durations:
  - step `466`: duration `60.98s`, batch prep `4.76s`
  - step `468`: duration `60.97s`, batch prep `5.42s`
  - step `469`: duration `60.96s`, batch prep `4.17s`

The rollout runs were also still active late into the same window:

- rollout-0 W&B last logged `_step=467`, cumulative batches `252`
- rollout-1 W&B last logged `_step=467`, cumulative batches `447`

So the terminal event does not look like gradual throughput collapse. It looks like another trainer-side TPU/JAX coordination death of the same broad family as earlier failures, but this time after the failure budget had already been spent.

## Interpretation of the Iris dashboard

What the dashboard is showing:

- trainer row in red `FAILED`
- trainer diagnostic: `Exit code: 1. stderr: RPC: /tensorflow.CoordinationService/...`
- rollout rows in gray `KILLED`
- rollout diagnostic: `Job exceeded max_task_failures`

Important nuance:

- the rollout rows are misleading if read literally as "the rollout jobs independently failed too many times"
- their task records show no ordinary failures at all
- the meaningful terminal event is the trainer child failure
- the rollout `killed` state is best interpreted as shutdown propagation after the RL coordinator/tree failed

## Hypothesis 4

The repeated trainer failures are random JAX coordination crashes with no narrower common signature.

## Results

Rejected.

Attempts `0`, `2`, `3`, and `4` all share the same pre-fatal pattern:

- normal training continues up to the checkpoint boundary
- Levanter logs `Saving temporary checkpoint at step ...`
- JAX checkpointing logs:
  - `Waiting for previous serialization to finish`
  - `Thread joined successfully`
  - `Error check finished successfully`
- the trainer main thread then goes quiet
- a replay-buffer / rollout-ingest thread continues logging for roughly `45s` to `3.5m`
- the process finally dies with:
  - `UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats): /job:jax_worker/replica:0/task:0`
  - `RPC: /tensorflow.CoordinationService/PollForError`

Attempt timelines:

- attempt `0`
  - checkpoint start: step `67` at `05:35:07`
  - fatal coordination error: `05:37:59`
- attempt `2`
  - checkpoint start: step `281` at `14:32:19`
  - fatal coordination error: `14:34:09`
- attempt `3`
  - checkpoint start: step `405` at `19:21:50`
  - fatal coordination error: `19:22:46`
- attempt `4`
  - checkpoint start: step `470` at `22:01:59`
  - fatal coordination error: `22:05:32`

More specifically, the failing checkpoints never emit the JAX async-commit log line:

- `Starting commit to storage layer by process: 0`

but successful earlier checkpoints in the same attempts do emit:

- `Starting commit to storage layer by process: 0`
- `Finished committing to storage layer by process: 0`
- `on_commit_callback successfully ran!`

That narrows the failure window considerably.

From the local code path:

- Levanter checkpoint save enters `tree_serialize_leaves_tensorstore(...)` in `lib/levanter/src/levanter/checkpoint.py`
- that calls `GlobalAsyncCheckpointManager.serialize(...)` in JAX
- JAX first waits for the previous async save to finish, then runs `asyncio.run(_run_serializer())`
- only after that returns does JAX start the async commit thread that logs `Starting commit to storage layer`

Because the failing attempts stop after `Error check finished successfully` and never log `Starting commit to storage layer`, the trainer is most likely dying or wedging inside the pre-commit serializer phase, not during the later storage-commit phase.

Concretely, that means the repeated failure signature is:

- checkpoint-triggered trainer heartbeat loss during `asyncio.run(_run_serializer())`
- likely while opening TensorStore writers, transferring shards from TPU to host, or waiting on TensorStore `write_future.copy`
- followed by JAX distributed noticing that `/job:jax_worker/replica:0/task:0` stopped heartbeating and force-aborting the process

This is more specific than "generic coordination error." The coordination error is the outer death notice; the shared inner signature is a checkpoint-path stall or crash before async commit starts.

## Bottom line

The run died because the trainer had another fatal TPU/JAX coordination failure at about `2026-03-28T22:05:32Z`, and by then the trainer task had already exhausted its configured ordinary failure retry budget (`max_retries_failure=3`).

Earlier incidents kept going because they were either:

- preemptions / worker failures, which use the larger preemption budget, or
- ordinary failures that still had remaining retries

This final one did not recover because there were no failure retries left. The rollout jobs were then killed as descendants after the trainer/coordinator failure, which is why the dashboard shows them as `KILLED` with the confusing `Job exceeded max_task_failures` diagnostic.

## Future Work

- [ ] Pull lower-level worker / TPU runtime logs for attempts `0`, `2`, `3`, and `4` to determine whether the checkpoint-path wedge is a TensorStore stall, host-transfer issue, or TPU runtime abort.
- [ ] Decide whether `max_retries_failure=3` is too small for long `500`-step production-like RL runs on preemptible TPU.
- [ ] Improve trainer failure surfacing so the earlier causal error is logged before the coordination-service fatal.
- [ ] Revisit the zombie/liveness issue noted in the Claude logbook, since delayed failure detection still makes postmortems harder than they should be.
