# Iris RL: Migrate RL Pipeline from Fray v1 (Ray) to Fray v2 (Iris)

**Status**: In progress
**Branch**: `iris_rl`
**Logbook**: [.agents/logbooks/iris-rl.md](../logbooks/iris-rl.md)
**Related**: [fray-lite-design.md](../../lib/fray/docs/fray-lite-design.md) Phase 5+6, [on-demand-rl.md](on-demand-rl.md), [iris-rl-codex.md](iris-rl-codex.md)

## Decisions (locked)

- **In-cluster coordinator**: Mandatory. Client submits one coordinator job; coordinator creates child jobs. No client-side orchestration.
- **Arrow Flight first**: Primary weight transfer mode. JAX transfer deferred to follow-up.
- **Graceful shutdown**: Via dedicated `RLRunState` actor, not bolted onto weight transfer coordinator.
- **No backwards compatibility**: Update all call sites and tests together.

## Problem

The RL pipeline (`lib/marin/src/marin/rl/`) is built on Fray v1 with Ray-era assumptions. The problem is not just API syntax — it's **topology**.

Today `rl_job.py:192-257` is a client-side launcher: it calls `current_cluster()`, submits sibling jobs, and waits. It is not a long-lived in-cluster coordinator. On Iris, that means:

- Trainer, rollout workers, and shared actors are sibling root jobs — not children of one hierarchy
- Iris cleanup doesn't naturally track RL as one run
- Region and environment inheritance don't flow through a parent job
- Actor namespace and job hierarchy aren't aligned around a single root

Additionally:
- Workers discover actors implicitly via `get_default_job_ctx().create_actor(..., get_if_exists=True)` — a Ray-era pattern that hides global context in constructors
- `RunConfig` has fields (`num_train_slices`, `max_retries_*`, `env_vars`) that are defined but never actually applied in `run()`
- The RL test harness (`tests/conftest.py:7-26`, `tests/rl/test_weight_transfer.py`) uses Fray v1 fixtures
- Weight transfer coordination is overloaded with lifecycle concerns (tri-state shutdown signal bolted onto the coordinator)

## Goals

- Change the RL topology so Iris sees one top-level coordinator job with child jobs beneath it
- Move RL orchestration from Fray v1 APIs to Fray v2 APIs
- Keep config objects serializable and deterministic; pass runtime handles separately
- Preserve local and threaded integration testing
- Prove correctness on Iris with Arrow Flight (the production weight transfer path)
- Remove Fray v1 from RL production code and tests together
- Fix existing `RunConfig` drift (slice count, retries, env vars, region)

**No backwards compatibility** inside RL — update call sites and tests together.

### Non-goals

- Replacing every Ray path in Marin (Iris is the target for RL, not a repo-wide switch)
- Day-one Arrow Flight/JAX transfer performance parity before correctness is proven
- General RL algorithm changes unrelated to orchestration
- Auto-restart-safe actor state

## Current Fray v1 Coupling (7 files + tests)

| File | What it uses | How |
|------|-------------|-----|
| `rl_job.py` | `fray.v1.cluster` | `current_cluster()`, `cluster.launch(JobRequest)`, `cluster.wait()` |
| `rollout_worker.py` | `fray.v1.job` | `get_default_job_ctx()` for curriculum actor `.remote()` + `.get()` |
| `train_worker.py` | `fray.v1.job` | `get_default_job_ctx()` for curriculum actor `.remote()` + `.get()` |
| `curriculum.py` | `fray.v1.job` | `get_default_job_ctx().create_actor(Curriculum, ..., get_if_exists=True)` |
| `weight_transfer/arrow_flight.py` | `fray.v1.job.context` | `get_default_job_ctx().create_actor(ArrowFlightCoordinator, ...)` — exposes `update_server()`, `fetch_server()` |
| `weight_transfer/jax.py` | `fray.v1.job.context` | `get_default_job_ctx().create_actor(WeightTransferCoordinator, ...)` — exposes `get_transfer_info()`, `register_transfer_server()`, `schedule_weight_transfer()`, `poll_transfers()`, `report_transfer_finished()`, `await_transfers()` |
| `scripts/evaluate_environment.py` | `fray.v1.cluster` | Job submission for evaluation |
| `tests/conftest.py` | `fray.v1` | Cluster/job fixtures |
| `tests/rl/test_weight_transfer.py` | `fray.v1` | Job context setup |
| `tests/rl/integration/config.py` | `fray.v1` | Direct worker config + threaded execution |

## Lessons from on-demand-rl (backend-independent fixes)

These are real operational issues regardless of orchestrator. Cherry-pick selectively:

| Fix | What | Status on `on-demand-rl` |
|-----|------|--------------------------|
| Faulthandler watchdogs + phase logging | Silent sampler crashes → 4+ hour trainer wait | Fixed |
| Trainer cumulative timeout | 1-hour fail-fast instead of infinite retry | Fixed |
| Dummy weight guard | `if weight_step < -1: continue` prevents stale generation | Fixed |
| Top-level exception handlers | Workers crash without traceback | Fixed |
| SymPy grading timeout | `sympy._eval_power` hangs on garbage output | Fixed (10s timeout) |
| `copy_and_flatten` OOM | `@jax.jit` pre-allocates 9.62G, only 7.14G free | Fixed (numpy on host) |

**Do NOT cherry-pick**: `FileSystemArrowFlightCoordinator`, `_resolve_advertise_host`, `bootstrap_rl.sh`, manual mode CLI — these are manual-mode-specific.

## Architecture: The Coordinator Job

The core architectural change: introduce a real in-cluster RL coordinator job.

```python
# Client side — submits ONE job
def submit_rl_job(config: RLJobConfig) -> JobHandle:
    client = current_client()
    return client.submit(JobRequest(
        name=f"rl-{config.run_id}",
        entrypoint=Entrypoint.from_callable(run_rl_coordinator, args=(config,)),
        resources=ResourceConfig.with_cpu(cpu=2, preemptible=False),
        environment=create_environment(config),
        max_retries_failure=0,
        max_retries_preemption=0,
    ))

# Runs inside the cluster as a real job
def run_rl_coordinator(config: RLJobConfig) -> None:
    client = current_client()  # Iris child context — inherits region, namespace
    train_config, rollout_config = RLJob(config).to_worker_configs()
    runtime = build_runtime_handles(client, config)  # creates actors
    jobs = submit_child_jobs(client, config.run_config, train_config, rollout_config, runtime)
    wait_all(jobs, raise_on_failure=True)
```

This gives us:
- **Job hierarchy**: coordinator → {trainer, rollout-0..N, curriculum-actor, run-state-actor}
- **Cascading cleanup**: killing the coordinator kills all children
- **Region inheritance**: child jobs inherit coordinator's region
- **Namespace alignment**: all actors share the coordinator's namespace

### Explicit runtime handles (separate from config)

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

`WeightTransferConfig` stays pure config (serializable, hashable for coordinator naming). `RLRuntimeHandles` is runtime-only, never stored in config.

Worker construction becomes:
```python
train_worker = TrainWorker(config=train_config, runtime=runtime)
rollout_worker = RolloutWorker(config=rollout_config, runtime=runtime)
```

No worker calls `get_default_job_ctx()`. No implicit actor discovery.

### Dedicated run-state actor

RL lifecycle is its own actor, not overloaded onto weight transfer:

```python
class RLRunState:
    status: Literal["running", "completed", "failed"]
    failure_message: str | None

    def mark_completed(self) -> None: ...
    def mark_failed(self, message: str) -> None: ...
    def get_status(self) -> str: ...
```

- Trainer calls `run_state.mark_completed()` when done
- Rollout workers poll `run_state.get_status()` for shutdown signal
- Replaces the ad-hoc tri-state coordinator from on-demand-rl

### Control-plane actors: non-preemptible, CPU-only

Current v1 uses `preemptible=False, num_cpus=0`. Preserve that:
- Coordinator job: `preemptible=False`, small CPU
- Curriculum actor: `preemptible=False`, CPU-only
- Run-state actor: `preemptible=False`, CPU-only
- Weight transfer coordinator: `preemptible=False`, CPU-only

## Migration Steps

### Step A: Land backend-independent operational fixes

Selectively backport from on-demand-rl (not blind cherry-pick):
- Faulthandler watchdogs + phase logging
- Trainer bounded wait (cumulative timeout)
- Stale-weight guards
- Math grading timeout
- `copy_and_flatten` OOM fix
- Top-level exception visibility

Standalone PR, no Fray changes.

### Step B: Introduce v2 orchestration on LocalClient first

Before Iris, get RL onto Fray v2 with explicit runtime handles under `LocalClient()`:

1. Create `orchestration.py` with `submit_rl_job()`, `run_rl_coordinator()`, runtime-handle creation
2. Create `run_state.py` with `RLRunState` actor
3. Update `RLJob`: keep `to_worker_configs()` (pure config), add `build_runtime_handles(client)` (creates actors)
4. Update workers: accept `runtime: RLRuntimeHandles`, stop calling `get_default_job_ctx()`
5. Delete `get_or_create_curriculum_actor()` from `curriculum.py`
6. Update weight transfer: accept coordinator handles explicitly, remove internal actor creation
7. Migrate test fixtures from v1 to v2 `LocalClient()`

This isolates "API migration" from "Iris backend validation".

### Step C: Add the coordinator job topology

Once workers and runtime handles are explicit:
1. Wire `submit_rl_job()` to submit the coordinator as a real in-cluster job
2. Coordinator creates child jobs for trainer + rollout workers
3. Test parent-child lifecycle: killing coordinator kills children

### Step D: Fix RunConfig drift

While touching orchestration, fix the fields that exist but aren't applied:

| Field | Current state | Fix |
|-------|--------------|-----|
| `num_train_slices` (`rl_job.py:55`) | Exists but `ResourceConfig.with_tpu()` called without `slice_count` | Pass to v2 `ResourceConfig.with_tpu(..., slice_count=...)` |
| `max_retries_failure` (`rl_job.py:58`) | Exists but not passed to jobs | Pass to `JobRequest` |
| `max_retries_preemption` (`rl_job.py:60`) | Exists but not passed to jobs | Pass to `JobRequest` |
| `env_vars` (`rl_job.py:64`) | Exists but not merged into environment | Merge into `EnvironmentConfig` |

Add region support:
```python
@dataclass
class RunConfig:
    ...
    regions: list[str] | None = None  # explicit co-location
```

### Step E: First Iris end-to-end run with Arrow Flight

Validate the full topology on Iris with Arrow Flight (the production weight transfer mode):

- One coordinator job
- One train child job
- One rollout child job
- One curriculum actor
- One run-state actor
- Arrow Flight weight transfer with GCP metadata internal-IP resolver (proven on on-demand-rl)
- Correct shutdown and failure propagation
- Cascading cleanup on coordinator kill
- Test on real TPUs with Llama 8B

Arrow Flight coordinator receives handle from runtime, not from implicit discovery.

### Step F: Re-enable JAX transfer mode

Last, because it's more concurrency-sensitive with stricter assumptions around server registration and completion tracking (`jax.py:122-207`).

## File-by-file plan

| File | Step | Change |
|------|------|--------|
| `rl_job.py` | B,C,D | Keep config assembly; stop submitting children directly; add v2 submit wrapper |
| **`orchestration.py`** (new) | B,C | `submit_rl_job()`, `run_rl_coordinator()`, runtime-handle creation, child-job submission |
| **`run_state.py`** (new) | B | `RLRunState` actor + enum for RL lifecycle |
| `curriculum.py` | B | Delete `get_or_create_curriculum_actor()`; keep actor methods and checkpoint logic |
| `train_worker.py` | A,B | Backport diagnostics; accept runtime handles; report to run-state actor |
| `rollout_worker.py` | A,B | Backport diagnostics; accept runtime handles; poll run-state for shutdown |
| `weight_transfer/__init__.py` | B | Thread runtime handles into factories; fix stale `get_or_create_actor` export (line 118) |
| `weight_transfer/arrow_flight.py` | A,B,E | Backport OOM fix; accept coordinator handle; add timeout policy |
| `weight_transfer/jax.py` | B,F | Accept coordinator handle; set proper actor concurrency |
| `weight_transfer/base.py` | B | Clean interface — lifecycle signaling moves to `run_state.py` |
| `scripts/evaluate_environment.py` | B | v1→v2 job submission |
| `tests/conftest.py` | B | Replace v1 fixtures with v2 `set_current_client()` |
| `tests/rl/test_weight_transfer.py` | B | Build explicit runtime handles |
| `tests/rl/integration/config.py` | B | Helpers for configs + local runtime handles from `RLJob` |
| `tests/rl/integration/test_iris_integration.py` | E | Real v2/Iris RL topology coverage (current file only tests actor communication) |

## Testing matrix

| Layer | What to prove |
|-------|--------------|
| Unit | `RLRunState` transitions, curriculum checkpoint restore, runtime factory behavior, `WeightTransferConfig` stays pure |
| Local integration | `RLJob.to_worker_configs()` + `build_runtime_handles(LocalClient())`, threaded workers, graceful stop, timeout behavior |
| Ray-backed v2 | Actor-handle serialization across jobs, child-job submission, parity with existing worker images |
| Iris CPU/small | Real hierarchical job IDs, namespace sharing, env/extras inheritance, region handling, cascading termination |
| Iris TPU | One-slice trainer + one rollout worker in one region, then multiple rollout workers |
| Failure tests | Kill a rollout worker, force trainer failure, verify run-state transitions and child cleanup |

## Opportunistic cleanups

- Fix stale `RLJobConfig.run_config` docstring ("None means simple Ray actors" — no such path exists)
- Remove missing `get_or_create_actor` export from `weight_transfer/__init__.py`
- Add explicit timeouts on actor calls via `.result(timeout=...)`
- Replace ad hoc global Fray context assumptions throughout RL

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Actor handle serialization fails across job boundaries | Medium | High | Test early with cloudpickle on LocalClient |
| Arrow Flight address resolution on Iris TPU workers | Medium | High | GCP metadata approach proven on on-demand-rl |
| Fray v2 `future.result()` timeout semantics differ from v1 | Low | Medium | Check `OperationFuture` impl, add explicit timeouts |
| JAX transfer coordinator is concurrency-sensitive | Medium | Medium | Defer to Step G, set higher actor concurrency |
| Coordinator job adds a hop — extra latency for actor calls | Low | Low | Coordinator is CPU-only, calls are metadata-only |
| Config hashing breaks if runtime handles leak into config | Medium | High | `RLRuntimeHandles` is frozen dataclass, never in `WeightTransferConfig` |

## Success criteria

1. RL submits as one Fray v2 coordinator job whose children include all workers and actors
2. Zero `fray.v1` imports in `lib/marin/src/marin/rl/` and RL tests
3. `WeightTransferConfig` remains pure config; runtime handles are separate
4. First Iris end-to-end works with Arrow Flight and cleans up children on success or failure
5. JAX transfer can be re-enabled without changing the outer RL topology
6. `RunConfig` fields (slice count, retries, env vars, region) are actually applied
7. All existing RL tests pass under `LocalClient()`
