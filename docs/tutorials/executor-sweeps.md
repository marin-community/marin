# Executor Sweeps: Racing Workers Over a Target List

This tutorial covers `marin.execution.sweep`, a small primitive for running
parameter sweeps as a pool of independent coordinator replicas that race to
claim each target.

## When to use it

Reach for `claim_and_run` when you have:

- A flat list of independent units of work — typically a hyper-parameter grid
  — that each need to run **exactly once** across the pool.
- N independent workers (e.g. N TPU Iris jobs) that each train one target
  inline. The sweep gives you N-way parallelism over the target list without
  a central coordinator.
- Crash tolerance: a worker can die mid-target and a peer will retry it.

If you only have a single worker, you do not need this; just call your
training function in a loop.

## The API

```python
from marin.execution.sweep import SweepTarget, claim_and_run
```

`SweepTarget(target_id, config)` is one unit of work. `target_id` is a
filesystem-safe string used as both the per-target output directory and the
lock label. `config` is opaque — the sweep library does not introspect it.

`claim_and_run(sweep_root, targets, run_fn)` walks `targets` in order and, for
each one, races peers on the lock at `f"{sweep_root}/{target.target_id}"`:

- If the worker wins the race, it holds the lock (with heartbeat refresh)
  while it calls `run_fn(target)`. On success it writes `STATUS_SUCCESS`; on
  exception it writes `STATUS_FAILED` and re-raises.
- If a peer already finished the target, `step_lock` raises `StepAlreadyDone`
  and the worker silently moves on.
- If a peer holds the lock right now, this worker also moves on; another
  target in the list is still up for grabs.

Coordination is pure-races on `step_lock`. Workers do not exchange messages.
With N workers and M targets each worker grabs roughly M/N targets in
parallel; workers exit naturally once every target reads `STATUS_SUCCESS`.

## The race-on-`step_lock` pattern

`step_lock` is the same primitive the executor uses to dedupe step execution.
For sweeps it provides three properties for free:

- **At-most-once-per-success.** A target that has already written
  `STATUS_SUCCESS` is a no-op for any later worker.
- **Heartbeat-based liveness.** A worker that crashes while holding the lock
  has its claim expire; another worker can pick the target up.
- **Failed-target retry.** `step_lock` defaults to `force_run_failed=True`,
  so a target that wrote `STATUS_FAILED` is retried by a later peer.

Workers iterate the same `targets` list in the same order, so contention on
the first few targets is high and quickly fans out as winners walk forward.

## Worked example

[`experiments/tutorials/train_tiny_sweep_tpu.py`](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_sweep_tpu.py)
runs a 3x3 LR/WD grid on a tiny Llama:

```python
from fray.cluster import ResourceConfig
from marin.execution.executor import materialize
from marin.execution.sweep import SweepTarget, claim_and_run

NUM_WORKERS = 3
SWEEP_NAME = "train-tiny-sweep"

# Pre-build all trials at submission time so workers do no config work.
targets = [SweepTarget(target_id=t.name, config=t) for t in trials]


def _run_one(target: SweepTarget) -> None:
    """Materialize the trial's config and train inline on this TPU worker."""
    trial = target.config
    config = materialize(trial.inner_config)
    levanter_train_lm.main(config)


def _sweep_worker_entrypoint(sweep_root: str) -> None:
    claim_and_run(sweep_root, targets, _run_one)
```

The submitter resolves `sweep_root` once (using `marin_prefix()` from the
user's region) and bakes it into the entrypoint args, so all replicas contend
on the same lock namespace regardless of where Iris schedules them. Then it
submits N independent TPU Iris jobs that all run the same entrypoint:

```python
sweep_root = os.path.join(marin_prefix(), "sweeps", SWEEP_NAME)
client = fray_client.current_client()

env = resolve_training_env(base_env=None, resources=RESOURCES)
handles = []
for i in range(NUM_WORKERS):
    handle = client.submit(
        JobRequest(
            name=f"{SWEEP_NAME}-{i}",
            entrypoint=Entrypoint.from_callable(_sweep_worker_entrypoint, args=[sweep_root]),
            resources=RESOURCES,
            environment=create_environment(env_vars=env, extras=extras_for_resources(RESOURCES)),
        )
    )
    handles.append(handle)
for h in handles:
    h.wait(raise_on_failure=True)
```

`SWEEP_NAME` is the stable lock-path key. Bump it to start a fresh sweep over
the same grid; leave it alone to resume an in-progress one (succeeded targets
are skipped, failed and unclaimed ones get picked up).

## Notes

- `run_fn` should be idempotent enough that a retry after a crashed peer does
  not corrupt prior state. The common case — submitting a child training job
  whose own output path is hash-derived — already satisfies this.
- The sweep root is an fsspec URL, so the lock files can live anywhere
  (local, GCS, etc.). For multi-region sweeps put it under the regional
  prefix so workers in that region do not pay cross-region read costs to
  check status.
- `claim_and_run` re-raises on `run_fn` failure, so the failing worker exits.
  Peer workers continue and eventually retry the failed target.
