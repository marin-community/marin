# Debugging log for qsplit240 520M/1.2B SIGSEGV compress failures

Investigate stopped qsplit240 520M and 1.2B pilot jobs that reportedly hit
`Failed to compress input file` and `SIGSEGV` errors.

## Initial status

User reported that the 520M and 1.2B runs were stopped after repeated
`Failed to compress input file` / `SIGSEGV` failures. Need to identify the
exact jobs, inspect the failing logs, and determine whether the compression
message is causal or incidental.

## Hypothesis 1

The visible `Failed to compress input file` message is memray profiler noise,
while the actual task failures come from a different native crash path
triggered during training or eval on the larger-scale pilot runs.

## Findings

- The current 520M/1.2B pilot failures are not explained by `Failed to compress input file`.
- The failing 1.2B child is `/calvinxu/dm-qsplit240-1-2b-chinchilla-pilot-20260409-194115/train_lm_baseline_unimax-857505`.
- Task `/6` fails with `RPC: /tensorflow.CoordinationService/RegisterTask`.
- Child logs show split-brain JAX rendezvous on retry:
  - task `0` starts the coordinator on `10.128.0.24:8476`
  - task `2` later starts a second coordinator on `10.128.0.123:8476`
  - peers connect to different coordinator addresses and time out with `DEADLINE_EXCEEDED`
- The 1.2B proportional child and the 520M children show a single coordinator and no self-initiated failure before manual termination.
- Iris endpoint lookup was not ordered, and retry assignment did not proactively delete endpoints from prior attempts before launching the next one.
- That leaves a real overlap window where old and new `jax_coordinator` endpoints can coexist during preemption/retry churn.

## Root Cause

The likely root cause is an Iris retry bug in JAX coordinator discovery. During
coscheduled retries, different tasks can resolve different stale/current
`jax_coordinator` endpoints and bootstrap against different coordinators. That
matches the observed `CoordinationService/RegisterTask` timeouts.

## Fix

- Delete endpoints for a task immediately when assigning a new attempt.
- Return endpoints newest-first so `resolved.first()` prefers the latest registration.

## Hypothesis 2

The stale-endpoint retry bug is real, but it is not the primary reason the
1.2B `v5p-64` jobs are failing. Instead, Levanter is manually calling
`jax.distributed.initialize(...)` in Iris TPU jobs that should defer
distributed init to the TPU runtime.

## Changes to make

- Keep the Iris retry cleanup patch, since it removes a real source of split
  coordinators on retries.
- Update `levanter.distributed.DistributedConfig.initialize()` so Iris TPU jobs
  go through `iris.runtime.jax_init.initialize_jax()` even when distributed
  cluster environment variables are already present.
- Add a regression test covering Iris TPU jobs with `_is_distributed() == True`.

## Results

- The relaunched stratified 1.2B child
  `/calvinxu/dm-stratified-1-2b-24b-20260410-033515/train_lm_baseline_stratified-6dbf91`
  still fails with `RPC: /tensorflow.CoordinationService/RegisterTask`.
- Querying `task_attempts` shows the same signature on attempt `0` for multiple
  tasks, so the failure happens before any retry-specific endpoint cleanup
  could matter.
- The relaunched qsplit 1.2B pilot child
  `/calvinxu/dm-qsplit240-1-2b-chinchilla-pilot-20260410-033515/train_lm_baseline_proportional-4f66ff`
  also shows repeated first-attempt `RegisterTask` failures across tasks `0-5`.
- That pattern is consistent with a bad initial TPU distributed bootstrap, not
  just stale retry metadata.
- Local verification for the Levanter fix passes:
  - `uv run pytest lib/levanter/tests/test_distributed.py`
  - `uv run pytest lib/iris/tests/test_jax_init.py`
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/distributed.py lib/levanter/tests/test_distributed.py`

## Future Work

- [ ] Resubmit the failing 1.2B pilot and stratified jobs from the Levanter fix
- [ ] Confirm the new jobs stop calling manual `jax.distributed.initialize` on TPU
- [ ] If `RegisterTask` persists, capture early task logs to isolate whether TPU runtime or worker allocation is misconfigured
