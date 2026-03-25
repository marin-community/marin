# Debugging log for E4 root preemption recovery

Investigate why the 100-step direct + GCS + prod RL run did not appear to recover cleanly after the outer root job was marked as preempted / worker-failed, and determine whether this is an Iris policy issue, an RL topology issue, or an operator error.

## Initial status

Observed on `/ahmed/irl-e4p-100-0324-1701`:
- root job entered real RL work and submitted train + rollout children
- train and rollout both started on TPU workers
- root bug report later showed:
  - task attempt `0` ended `worker_failed`
  - worker error: `Worker marin-cpu-vm-e2-highmem-2-ondemand-us-ce-20260324-2354-a0ab70dd-worker-0 failed: Request timed out`
  - job-level `preemption_count=1`
- child train and rollout jobs were both transitioned to `KILLED` with error `Parent task preempted`
- root job itself was not terminal; it returned to a retriable/pending state before being manually killed.

## Hypothesis 1

This is not a true RL-code failure. It is the expected Iris reaction to a parent worker failure on a single-task parent job whose effective preemption policy is `TERMINATE_CHILDREN`.

## Changes to make

No code changes yet. Inspect:
- Iris controller transition logic for worker-failed tasks and descendant cascade
- effective preemption policy resolution for single-task jobs
- RL direct topology to see whether shared runtime actors live in the parent process

## Results

Confirmed:
- In `lib/iris/src/iris/cluster/controller/transitions.py`, when a worker dies while a task is executing, Iris increments `preemption_count` and sets the task back to `PENDING` if it still has preemption retries left.
- In the same transition, if the effective preemption policy is `JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN`, Iris calls `_cascade_children(..., "Parent task preempted")` and kills all descendant jobs.
- `_resolve_preemption_policy(...)` defaults single-task jobs (`replicas <= 1`) to `JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN` when no explicit policy is set.
- The direct RL path runs `_run_rl_coordinator(...)` in the outer/root job process. That coordinator hosts the curriculum actor, run-state actor, and Arrow Flight weight-transfer coordinator in-process via `client.host_actor(...)`.
- Therefore, when the root task's worker dies, those hosted actors disappear with it. Preserving train/rollout children across parent death would not be safe in the current architecture because they depend on those actor endpoints.
- The root job itself *was* configured for automatic preemption retry at the Iris layer (default `max_retries_preemption=100` for jobs submitted via the client), which is why the root job remained non-terminal and had `preemption_count=1`.
- So the statement "why wouldn't it automatically restart" has a nuanced answer:
  - the root *was* automatically retrying
  - but Iris intentionally killed descendant train/rollout jobs on parent preemption
  - and I manually killed the root before the retry completed and re-submitted children.

## Hypothesis 2

The current direct RL topology is only restart-safe at the granularity of the whole subtree. It is not child-preserving restart-safe because the parent/root process is also the live host for critical shared actors.

## Future Work

- [ ] Decide whether operator policy should be: do not manually restart after the first parent preemption; allow the root retry path to run.
- [ ] If we want stronger robustness, add explicit preemption policy support to Fray/Iris job requests and decide where `PRESERVE_CHILDREN` is actually safe.
- [ ] If we want children to survive parent restart, move shared runtime actors out of the parent process into durable dedicated jobs/services first.
- [ ] Consider reducing checkpoint interval for longer direct/prod probes so whole-subtree restart loses less work.
