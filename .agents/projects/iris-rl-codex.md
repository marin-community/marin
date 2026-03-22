# Iris RL (Codex): A Migration Plan That Actually Fits Iris

**Status**: Planning
**Replaces**: `iris-rl.md`
**Related**: `lib/fray/docs/fray-lite-design.md`, `.agents/projects/on-demand-rl.md`

## Problem

The RL stack in `lib/marin/src/marin/rl/` is still built around Fray v1 and Ray-era assumptions, but the hard part is not the import rename.

- `lib/marin/src/marin/rl/rl_job.py:192-257` is a client-side launcher. It calls `current_cluster()`, submits sibling jobs, and waits. It is not a long-lived in-cluster coordinator.
- `lib/marin/src/marin/rl/curriculum.py:610-624` creates the curriculum actor lazily inside workers via `get_default_job_ctx().create_actor(..., get_if_exists=True)`.
- `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py:398-440` and `lib/marin/src/marin/rl/weight_transfer/jax.py:345-352` / `437-523` do the same for weight-transfer coordination.
- `tests/conftest.py:7-26` and `tests/rl/test_weight_transfer.py:132-140` still set up Fray v1 cluster/job fixtures, so the current test harness is part of the migration surface.
- Iris job hierarchy only appears when a job submits child jobs from inside an Iris job context; that is where parent job IDs, env/extras inheritance, region inheritance, and namespace sharing come from (`lib/iris/src/iris/client/client.py:595-729`, `lib/iris/src/iris/cluster/types.py:548-579`).

That means a mechanical `current_cluster()` -> `current_client()` migration in `rl_job.py` is not enough. It would make the syntax v2-compatible while keeping the topology wrong for Iris.

### Why `iris-rl.md` is not sufficient

The existing plan is directionally useful, but it is not accurate enough to execute safely.

1. It treats `rl_job.py` as the “controller”, but today it is only a submitter (`rl_job.py:192-257`). If that shape is preserved, trainer, rollout workers, and shared actors remain sibling root jobs on Iris instead of child jobs in one hierarchy.
2. It proposes putting actor creation in `rl_job.py` without first moving RL orchestration into an actual job. That loses the main Iris benefit the migration is supposed to gain: child-job lifecycle and cascading cleanup.
3. It lists the weight-transfer coordinator API incorrectly. `ArrowFlightCoordinator` only exposes `update_server()` and `fetch_server()` (`arrow_flight.py:202-228`), while the JAX coordinator exposes `get_transfer_info()`, `register_transfer_server()`, `schedule_weight_transfer()`, `poll_transfers()`, `report_transfer_finished()`, and `await_transfers()` (`jax.py:115-207`).
4. It suggests threading runtime handles through `WeightTransferConfig`, but `RLJob.to_worker_configs()` hashes `dataclasses.asdict(self.config.weight_transfer)` to derive the coordinator name (`rl_job.py:277-287`). Putting handles in config would break purity, hashing, and serialization.
5. It ignores test and local-mode coupling. A large chunk of RL integration coverage constructs configs directly from `RLJob.to_worker_configs()` and instantiates `TrainWorker` / `RolloutWorker` in threads (`tests/rl/integration/config.py:551-616`).
6. It overstates the repo’s current state. `infra/README.md:30-36` still describes Ray as the underlying cluster infrastructure for Marin. Iris is the new orchestrator path we want RL to support well, not the finished replacement for every existing Ray path.
7. It does not call out existing functional gaps already visible in `RunConfig`: `num_train_slices`, retries, and `env_vars` exist in config (`rl_job.py:42-65`) but are not actually applied in `run()` (`rl_job.py:201-257`).

## Goals

- Move RL orchestration from Fray v1 APIs to Fray v2 APIs.
- Change the RL topology so Iris sees one top-level RL coordinator job with child jobs beneath it.
- Keep config objects serializable and deterministic; pass runtime handles separately.
- Preserve local and threaded integration testing rather than forcing every test through a remote cluster.
- Prove correctness on Iris first, then optimize weight transfer.
- Remove Fray v1 from RL production code and RL tests together.
- Fix existing `RunConfig` drift while touching orchestration: TPU slice count, retries, env vars, and region selection.

**Backwards compatibility**: none inside RL. We should update call sites and tests together rather than carry a permanent v1/v2 dual path.

### Non-goals

- Replacing every Ray path in Marin.
- Achieving day-one performance parity with the best Arrow Flight or JAX transfer mode before correctness is proven.
- General RL algorithm changes unrelated to orchestration.
- Making actor state magically restart-safe by auto-restarting in-memory actors.

## Proposed Solution

### Core idea

Introduce a real RL coordinator job that runs inside Fray v2 and submits child jobs from there. The outer client submits one job; the inner coordinator owns shared actors, worker runtime handles, failure propagation, and shutdown.

This should look much closer to Zephyr’s job-based execution model (`lib/zephyr/src/zephyr/execution.py:4-12`) than to the current `RLJob.run()`.

```python
def submit_rl_job(config: RLJobConfig) -> JobHandle:
    client = current_client()
    request = JobRequest(
        name=f"rl-{config.run_id}",
        entrypoint=Entrypoint.from_callable(run_rl_coordinator, args=(config,)),
        resources=ResourceConfig.with_cpu(cpu=2, preemptible=False),
        environment=create_environment(env_vars=config.run_config.env_vars, extras=config.pip_dependency_groups),
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    return client.submit(request)


def run_rl_coordinator(config: RLJobConfig) -> None:
    client = current_client()
    train_config, rollout_config = RLJob(config).to_worker_configs()
    runtime = create_runtime_handles(client, config)
    jobs = submit_rl_children(client, config.run_config, train_config, rollout_config, runtime)
    wait_all(jobs, raise_on_failure=True)
```

### Make runtime explicit

Shared runtime objects should stop being discovered implicitly from inside workers. They should be created once and passed explicitly.

```python
@dataclass(frozen=True)
class WeightTransferRuntime:
    arrow_flight_coordinator: ActorHandle | None = None
    jax_transfer_coordinator: ActorHandle | None = None


@dataclass(frozen=True)
class RLRuntimeHandles:
    curriculum: ActorHandle
    run_state: ActorHandle
    weight_transfer: WeightTransferRuntime
```

`WeightTransferConfig` stays declarative. `RLRuntimeHandles` is runtime-only and never hashed into config.

### Add a dedicated run-state actor

The current and previous plans overload weight-transfer coordination with job-lifecycle concerns. That is the wrong abstraction boundary.

RL should have a dedicated control-plane actor, for example `RLRunState`, with responsibilities like:

- current terminal state: `RUNNING`, `COMPLETED`, `FAILED`
- trainer completion signal
- optional failure message / failing child job ID
- optional “stop sampling now” signal for rollout workers

This replaces the ad hoc “tri-state coordinator” idea being bolted onto weight transfer. Weight transfer should coordinate weight transfer; RL lifecycle should be its own actor.

### Use staged weight-transfer bring-up

The first Iris proof should use `GCS_CHECKPOINT` mode from `lib/marin/src/marin/rl/weight_transfer/__init__.py:42-97`.

Reason:

- it removes Arrow Flight service discovery as a blocker,
- it removes JAX transfer server address registration as a blocker,
- it gives us a correctness baseline for job hierarchy, actor wiring, and worker shutdown.

Only after that path is solid should we wire Arrow Flight and JAX transfer back in for performance.

### Preserve local testability

`RLJob.to_worker_configs()` is useful and should stay. The new layer is an explicit runtime-builder rather than implicit actor lookup:

- `RLJob.to_worker_configs()` returns pure serializable configs.
- `RLJob.build_runtime_handles(client)` creates control-plane actors for the chosen backend.
- tests can use `LocalClient()` and direct threaded workers without Fray v1 fixtures.

That keeps tests realistic while removing hidden global context from worker constructors.

## Implementation Outline

1. Introduce a top-level RL coordinator entrypoint and explicit runtime-handle dataclasses.
2. Move curriculum and weight-transfer actor creation out of workers and into coordinator-owned runtime assembly.
3. Convert RL job submission to Fray v2, including slice count, retries, env vars, and region handling.
4. Migrate RL tests and fixtures from Fray v1 globals to Fray v2 local/ray/iris clients.
5. Enable the first Iris correctness path with `GCS_CHECKPOINT`, then add failure-handling validation.
6. Re-introduce Arrow Flight and JAX transfer on top of the stable Iris topology and remove remaining stale RL/v1 code.

## Notes

### 1. Topology is the most important change

This plan is intentionally stricter than `iris-rl.md`: the main goal is not “Fray v2 syntax”, it is “correct orchestration semantics on Iris”.

If the laptop submits trainer, rollout workers, curriculum, and weight-transfer actors directly:

- they become siblings, not children;
- Iris cleanup no longer naturally tracks RL as one run;
- region and environment inheritance do not flow through one parent job;
- actor namespace and job hierarchy are no longer aligned around a single root run.

That is why the coordinator job is mandatory.

### 2. Control-plane jobs must be non-preemptible

Current v1 actor creation uses `preemptible=False` and essentially zero CPU for the curriculum and transfer coordinators (`curriculum.py:612-614`, `arrow_flight.py:399-405`, `jax.py:346-348`, `jax.py:438-440`). The new plan should preserve that intent.

I would make the coordinator job and shared control-plane actors explicitly:

- `preemptible=False`
- CPU-only
- small but non-zero CPU reservation
- `max_restarts=0` for actors whose state is in memory and not reconstructible from the constructor alone

For the JAX weight-transfer coordinator specifically, I would also set a higher actor concurrency because it serves overlapping poll/report calls.

### 3. Fix `RunConfig` while we are here

The current orchestration path already drops important config values.

- `num_train_slices` exists (`rl_job.py:55-56`) but `ResourceConfig.with_tpu()` is called without `slice_count` (`rl_job.py:203`).
- `max_retries_failure` and `max_retries_preemption` exist (`rl_job.py:58-62`) but are not passed into the submitted jobs.
- `env_vars` exists (`rl_job.py:64-65`) but is not merged into the actual environment in `run()`.

The v2 migration should fix this instead of preserving it. Fray v2 already has the matching fields in `ResourceConfig.with_tpu(..., slice_count=...)` and `JobRequest` (`lib/fray/src/fray/v2/types.py:349-379`, `519-546`).

### 4. Add explicit region control

Iris child jobs inherit the parent’s region if the parent has one (`lib/iris/src/iris/client/client.py:686-693`), but today RL does not expose region placement explicitly through `RunConfig`.

I would add a region field to RL deployment config, for example:

```python
@dataclass
class RunConfig:
    train_tpu_type: str
    num_rollout_workers: int = 4
    inference_tpu_type: str | None = None
    num_train_slices: int = 1
    regions: list[str] | None = None
    max_retries_failure: int = 3
    max_retries_preemption: int = 100
    env_vars: dict[str, str] = field(default_factory=dict)
```

Then apply `regions=run_config.regions` when building TPU `ResourceConfig`s. For Iris, that gives us explicit co-location for weight transfer and makes inherited placement meaningful.

### 5. Worker constructors should stop owning actor discovery

Today workers effectively do this:

- `TrainWorker` creates transfer server state and discovers the curriculum actor.
- `RolloutWorker` discovers the curriculum actor and builds weight-transfer clients that also discover their own coordinators.

That makes worker construction depend on hidden global context and backend-specific actor lookup.

I would change the public construction pattern to:

```python
train_worker = TrainWorker(config=train_config, runtime=runtime)
rollout_worker = RolloutWorker(config=rollout_config, runtime=runtime)
```

Then:

- `curriculum.py` keeps the `Curriculum` actor class but loses `get_or_create_curriculum_actor()`.
- weight-transfer factories accept runtime handles explicitly.
- no worker calls `get_default_job_ctx()`.

### 6. File-by-file plan

| File | Change | Why |
|------|--------|-----|
| `lib/marin/src/marin/rl/rl_job.py` | Stop submitting child workers directly; keep config assembly, add v2-facing submit wrapper or delegate to new orchestration module | separates pure config generation from orchestration |
| `lib/marin/src/marin/rl/orchestration.py` | New module for `submit_rl_job()`, `run_rl_coordinator()`, runtime-handle creation, child-job submission | creates the missing in-cluster coordinator layer |
| `lib/marin/src/marin/rl/run_state.py` | New actor + enum for RL lifecycle state | keeps shutdown/failure signaling out of weight transfer |
| `lib/marin/src/marin/rl/curriculum.py` | Delete `get_or_create_curriculum_actor()`; keep actor methods and checkpoint logic only | removes hidden global actor discovery |
| `lib/marin/src/marin/rl/train_worker.py` | Accept runtime handles; report completion/failure to run-state actor; stop using Fray v1 context | makes trainer explicit and backend-neutral |
| `lib/marin/src/marin/rl/rollout_worker.py` | Accept runtime handles; poll run-state actor for shutdown; stop using Fray v1 context | allows graceful stop without coordinator hacks |
| `lib/marin/src/marin/rl/weight_transfer/__init__.py` | Thread runtime handles into factories; delete stale `get_or_create_actor` export (`__all__` currently references a missing symbol at line 118) | cleans up broken API surface |
| `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` | Accept explicit coordinator handle; add explicit timeout policy; preserve non-preemptible control-plane semantics; backport OOM fix | makes Arrow Flight compatible with explicit runtime wiring |
| `lib/marin/src/marin/rl/weight_transfer/jax.py` | Accept explicit coordinator handle; set proper actor concurrency; tighten timeout and error reporting | current JAX coordinator is concurrency-sensitive |
| `lib/marin/src/marin/rl/scripts/evaluate_environment.py` | Move any remaining Fray v1 submit path to v2 | keeps RL-adjacent tooling aligned |
| `tests/conftest.py` | Replace Fray v1 cluster/job fixtures with Fray v2 `set_current_client()` or explicit client fixtures | test harness must migrate too |
| `tests/rl/test_weight_transfer.py` | Build explicit runtime handles in tests rather than relying on v1 job context | keeps transfer tests meaningful |
| `tests/rl/integration/config.py` | Add helpers that create configs plus local runtime handles from `RLJob` | preserves thread-based integration tests |
| `tests/rl/integration/test_iris_integration.py` | Add real v2/Iris RL topology coverage; current file only tests Iris actor communication under Ray | existing coverage is too narrow |

### 7. Migration order I would actually use

I would not do this as one giant backend switch.

#### Step A: Land non-controversial RL fixes first

Selectively backport operational fixes from `.agents/projects/on-demand-rl.md`, but not by blind cherry-pick. The principle is “take the production-hardening fixes that are backend-independent”.

That includes:

- faulthandler watchdogs and phase logging,
- trainer-side bounded wait behavior,
- stale-weight guards,
- math grading timeout,
- `copy_and_flatten` OOM fix,
- top-level exception visibility.

It does **not** automatically include every manual-mode or bootstrap script change from that branch.

#### Step B: Introduce v2 orchestration on LocalClient and Ray first

Before Iris enters the picture, get RL onto Fray v2 and explicit runtime handles under:

- `LocalClient()` for most tests,
- Ray-backed Fray v2 where helpful for parity with current deployment assumptions.

That isolates “API migration” from “Iris backend validation”.

#### Step C: Add the coordinator job topology

Once workers and runtime handles are explicit, land the coordinator job and child-job submission model. This is where RL starts matching Iris semantics.

#### Step D: Enable the first Iris correctness run with `GCS_CHECKPOINT`

The first Iris end-to-end run should validate:

- one coordinator job,
- one train child job,
- one rollout child job,
- one curriculum actor,
- one run-state actor,
- correct shutdown and failure propagation.

Only after that should we bring faster weight-transfer modes back.

#### Step E: Re-enable Arrow Flight on Iris

At this point, decide whether to:

- keep self-advertised addresses with a real internal-IP resolver, or
- lean into Iris endpoint registration more directly.

I would keep the first version simple and proven, then optimize.

#### Step F: Re-enable JAX transfer mode

This comes last because it is more concurrency-sensitive and has stricter assumptions around explicit server registration and completion tracking (`jax.py:122-207`, `213-305`).

### 8. Testing matrix

The existing plan says “write an Iris integration test”; that is not enough. I would treat validation as a matrix.

| Layer | What to prove |
|------|----------------|
| Unit | `RLRunState` transitions, curriculum checkpoint restore, explicit runtime factory behavior, `WeightTransferConfig` remains pure |
| Local integration | `RLJob.to_worker_configs()` plus `build_runtime_handles(LocalClient())`, threaded `TrainWorker` and `RolloutWorker`, graceful stop, timeout behavior |
| Ray-backed Fray v2 | actor-handle serialization across jobs, child-job submission shape, parity with existing worker images/extras |
| Iris CPU/small run | real hierarchical job IDs, namespace sharing, env/extras inheritance, region handling, cascading termination |
| Iris TPU validation | one-slice trainer plus one rollout worker in one region, then multiple rollout workers |
| Failure tests | kill a rollout worker, force trainer failure, verify run-state transitions and child cleanup |

The current `tests/rl/integration/test_iris_integration.py:4-113` is only a hybrid actor communication test under Ray. It is useful, but it does not validate RL orchestration.

### 9. Differences from `iris-rl.md`

These are the deliberate differences, not minor edits.

1. **Coordinator job is mandatory**. The old doc implicitly keeps `rl_job.py` as a client-side controller; I would replace that with a real in-cluster RL coordinator job.
2. **Runtime is explicit**. I would never put actor handles inside `WeightTransferConfig`; the old doc leaves that door open.
3. **Run lifecycle is its own actor**. I would not overload the weight-transfer coordinator with trainer-completed / failed state.
4. **GCS checkpoint is the first Iris proof**. The old doc jumps too quickly toward Arrow Flight/JAX-style service coordination.
5. **Tests are first-class migration scope**. The old doc understates how much of the RL test harness is Fray v1-specific.
6. **`RunConfig` drift gets fixed during migration**. The old doc does not call out the currently ignored slice-count, retries, and env-var fields.
7. **Placement is explicit**. I would add region support and preserve non-preemptible control-plane semantics.
8. **The repo’s current Ray reality is acknowledged**. I would describe Iris as the target orchestrator path for RL, not pretend the whole repository has already moved.

### 10. Things I would clean up opportunistically

These are not the primary goal, but the migration is the right time to remove obviously stale edges.

- Fix the stale `RLJobConfig.run_config` docstring that says `None` means “simple Ray actors”; the current implementation does not actually provide that path.
- Remove the missing `get_or_create_actor` export from `weight_transfer/__init__.py`.
- Replace ad hoc global Fray context assumptions with explicit client/runtime plumbing throughout RL.
- Add explicit timeouts around actor calls where the v2 API now exposes `.result(timeout=...)` semantics on Iris-backed operation futures.

## Future Work

- Optimize Iris weight transfer with endpoint-aware Arrow Flight coordination if the simpler address approach becomes painful.
- Consider whether hosted actors (`host_actor`) are appropriate for any single-process control-plane pieces after the basic topology is stable.
- Add richer RL coordinator observability: per-child status summaries, weight-transfer lag, rollout freshness, and explicit progress reporting.
- Revisit actor persistence or checkpointed control-plane state only after the first stable Iris runs exist.

## Success Criteria

1. RL submits as one Fray v2 job whose coordinator creates all shared actors and worker child jobs.
2. RL workers and RL tests no longer import `fray.v1`.
3. `WeightTransferConfig` remains pure config; runtime handles are separate.
4. The first Iris end-to-end run works with `GCS_CHECKPOINT` and cleans up children on success or failure.
5. Arrow Flight and JAX transfer can be reintroduced without changing the outer RL topology again.
