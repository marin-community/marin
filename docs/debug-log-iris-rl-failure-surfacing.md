# Debugging log for iris rl failure surfacing

Investigate why the Iris cluster kept showing `/ahmed/iris-rl-oom-b120-uc1-r6` as running even after RL child jobs had already crashed with `AttributeError: 'OutputName' object has no attribute 'rstrip'`.

## Initial status

Observed on March 24, 2026:
- `iris job bug-report /ahmed/iris-rl-oom-b120-uc1-r6` showed the root job as `running` with `Failures 0`.
- `iris job logs /ahmed/iris-rl-oom-b120-uc1-r6 --include-children` showed repeated train and rollout worker crashes.
- After retries were exhausted, both child TPU jobs transitioned to `JOB_STATE_FAILED`, but the RL coordinator job and all ancestors still appeared as `JOB_STATE_RUNNING`.

## Hypothesis 1

The root bug report is shallow and only reports the exact job ID, so descendant failures are not surfaced there.

## Changes to make

No code change for this hypothesis. Verify by comparing:
- `iris job bug-report <root-job>`
- `iris job list --prefix <root-job> --json`
- `iris job logs <root-job> --include-children`

## Results

Confirmed.

`bug-report` only calls `get_job_status()` and `list_tasks()` on the exact job ID passed in. It does not aggregate descendant job states or descendant task failures.

Relevant code:
- `lib/iris/src/iris/cli/bug_report.py`
- `lib/iris/src/iris/cli/job.py`

That explains why the root bug report missed the failing train/rollout jobs while they were nested below the RL coordinator.

## Hypothesis 2

The parent remained `running` because child jobs still had retry budget left.

## Changes to make

No code change. Inspect descendant states from the controller and compare with Marin RL retry configuration.

## Results

Confirmed, but only for the first part of the incident.

The train and rollout jobs were submitted with `max_retries_failure=3`, so Iris kept them in non-terminal states until the fourth failure:
- `lib/marin/src/marin/rl/rl_job.py`
- `lib/marin/src/marin/rl/orchestration.py`
- `lib/iris/src/iris/cluster/controller/transitions.py`

This explains why the root looked healthy early in the run.

## Hypothesis 3

Even after the train child hit terminal `JOB_STATE_FAILED`, the coordinator stayed `running` because the coordinator process never actually exited.

## Changes to make

Trace the coordinator error path and inspect in-process actor lifecycle.

## Results

Confirmed.

The coordinator log shows:
- `wait_all(jobs, raise_on_failure=True)` raised `fray.v2.client.JobFailed`
- the traceback came from the coordinator task itself

However, the controller still reports the coordinator task row as `state=3` (`RUNNING`) with no recorded error. This means Iris never observed the task container stop.

The reason is that the coordinator hosts actors in-process:
- `lib/marin/src/marin/rl/orchestration.py:_create_runtime_handles`
- `lib/fray/src/fray/v2/iris_backend.py:530`

`host_actor()` returns `HostedActor(handle, stop=server.stop)`, but the coordinator never calls `shutdown()` on those hosted actors when `wait_all(...)` raises.

Those actor servers use `ActorServer`, which owns non-daemon managed threads and executor threads:
- `lib/iris/src/iris/actor/server.py`
- `lib/iris/src/iris/managed_thread.py`

So the main coordinator thread can crash, log a traceback, and still leave the Python process alive because the hosted actor server threads are still running. Iris task state stays `RUNNING` until the process actually exits.

## Future Work

- [ ] Fix the unresolved `OutputName` paths in runtime-built RL worker configs.
- [ ] Add coordinator cleanup so hosted actors are always shut down in a `finally` block.
- [ ] Decide whether `iris job bug-report` should optionally aggregate descendant job failures.
