# Iris JAX gangs: retries resolved a stale coordinator

Nested-MoE experiment gangs on `cw-us-east-08a` repeatedly hung during
`jax.distributed.initialize` after a whole-gang retry.

## Initial status

The r16 E128 Gate 1 and nested25 Gate 2 jobs each showed all 16 tasks running,
but no training process reached compilation or W&B. Iris reported two task
failures in each job. Worker logs showed different members of the same gang
connecting to two coordinator addresses.

## Stale endpoint hypothesis

The retrying workers had `IRIS_TASK_ID` attempt 1 while the endpoint registry
still exposed task 0's attempt-0 `jax_coordinator` during the restart window.
Some workers resolved the old address before task 0 registered its new
address. The registry later showed only the current address, but the early
workers were already blocked on the old coordinator.

This split is sufficient to deadlock JAX initialization and is independent of
the model or training architecture.

## Change

`iris.runtime.jax_init.initialize_jax` now scopes coordinator endpoint names by
nonzero task attempt. Attempt 0 keeps `jax_coordinator`; a retried gang uses
`jax_coordinator/attempt-N`. A regression places an attempt-0 address in the
registry and verifies that both ranks in attempt 1 join the newly registered
attempt-1 address.

## Results

- 31 focused JAX initialization tests passed.
- The required pre-commit entry point passed for both changed files.
- Commit `b9029615fa` contains the fix.
- The poisoned r16 experiment jobs were stopped and relaunched as r17 at batch
  priority.

## Future work

- [ ] Confirm a live r17 whole-gang retry reconnects every rank to the same
      attempt-scoped endpoint.
- [ ] Review endpoint cascade-delete timing separately; attempt scoping makes
      JAX robust without depending on immediate deletion.
