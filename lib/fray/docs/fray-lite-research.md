# Fray-Lite Research: Codebase Analysis

This document captures the research phase for the Fray V2 ("fray-lite") design.
See [GitHub #2552](https://github.com/marin-community/marin/issues/2552) for context.

## Goal

Define a minimal Fray interface that supports our three primary workloads
(Zephyr data processing, RL training, Levanter training) and can be backed by
either Ray or Iris, then migrate existing code to this interface and delete the
v1 Fray.

## Current Fray API Surface

Fray today exposes three orthogonal subsystems:

### 1. `fray.cluster` — Job Orchestration

The `Cluster` abstract class provides:

```python
class Cluster:
    def launch(request: JobRequest) -> JobId
    def monitor(job_id: JobId) -> JobInfo
    def poll(job_id: JobId) -> JobInfo
    def terminate(job_id: JobId) -> None
    def list_jobs() -> list[JobInfo]
    def wait(job_id, raise_on_failure=False) -> JobInfo
```

Supporting types: `JobRequest`, `ResourceConfig`, `DeviceConfig` (CpuConfig /
GpuConfig / TpuConfig), `EnvironmentConfig`, `Entrypoint`, `JobId`, `JobInfo`,
`JobStatus`.

Backends: `LocalCluster` (threads/subprocesses), `RayCluster` (Ray job
submission + remote tasks + TPU gang scheduling).

### 2. `fray.job` — In-Job Execution Context

The `JobContext` protocol provides:

```python
class JobContext:
    def put(obj) -> ref
    def get(ref) -> obj
    def run(fn, *args) -> future
    def wait(futures, num_returns=1) -> (ready, pending)
    def create_actor(cls, *args, name=..., get_if_exists=..., preemptible=..., num_cpus=...) -> ActorHandle
```

Backends: `SyncContext`, `ThreadContext`, `RayContext`.

### 3. `fray.queue` — Distributed Queue

Lease-based queue protocol with `MemoryQueue`, `FileQueue`, `HttpQueue`.

## How Each Workload Uses Fray

### Zephyr (Data Processing)

**Imports**: `JobContext`, `get_default_job_ctx` from `fray.job`

**Usage**: `zephyr/backends.py` uses `JobContext` for distributed map/reduce:
- `context.put(data)` — store chunks in object store
- `context.run(run_stage, ctx, ops)` — execute stage functions remotely
- `context.wait(futures)` — wait for stage completion
- `context.get(ref)` — retrieve results

**Does NOT use**: `fray.cluster`, `fray.queue`, actors.

**Key insight**: Zephyr only needs the task-dispatch portion of `JobContext`
(put/get/run/wait). It doesn't use actors or cluster-level job management.

### RL Training (marin.rl)

**Imports**: Both `fray.cluster` and `fray.job`.

#### Job orchestration (`rl_job.py`)

Uses `fray.cluster` to launch TPU jobs:

```python
cluster = current_cluster()
cluster.launch(JobRequest(name="train", resources=train_resources, entrypoint=...))
cluster.launch(JobRequest(name="rollout-N", resources=rollout_resources, entrypoint=...))
cluster.wait(jobs, raise_on_failure=True)
```

This is the "outer loop" — launching 1 train worker + N rollout workers as
separate cluster jobs, then waiting for all to complete.

#### Actor coordination (within jobs)

Uses `fray.job` to create shared actors:

1. **Curriculum actor** — shared between train + rollout workers via
   `get_default_job_ctx().create_actor(Curriculum, name="curriculum",
   get_if_exists=True, preemptible=False, num_cpus=0)`. Workers call
   `.sample_lesson.remote()`, `.update_lesson_stats.remote()`, etc.

2. **Weight transfer coordinators** — `ArrowFlightCoordinator` and
   `WeightTransferCoordinator` created similarly with `preemptible=False,
   num_cpus=0`. Coordinate weight sync between train and rollout workers.

All actors share these properties:
- Created with `get_if_exists=True` (singleton per name)
- `preemptible=False` (must survive worker preemptions)
- `num_cpus=0` (lightweight coordination, no compute)
- Called via `.method.remote()` / `ctx.get(future)` pattern

#### Data transfer (rollout storage)

Does NOT use `fray.queue`. Uses custom `RolloutStorage` abstraction with
file-based (GCS pickle files) or in-memory implementations. Rollout workers
write `RolloutBatch` objects; train worker reads them via polling.

### Levanter Training (training.py)

**Imports**: `fray.cluster` only.

**Usage**: Wraps `train_lm.main` in a `JobRequest` and launches via cluster:

```python
cluster = current_cluster()
job_id = cluster.launch(JobRequest(
    name="train_lm",
    entrypoint=Entrypoint.from_callable(train_lm.main, args=[config]),
    resources=config.resources,
    environment=EnvironmentConfig.create(env_vars=env),
))
cluster.wait(job_id, raise_on_failure=True)
```

Levanter itself does NOT import fray at all (except `device_flops` for MFU
logging). The training code is pure JAX/Haliax.

### Experiments (100+ files)

All experiment files use `ResourceConfig` to specify compute. Most never
interact with the cluster API directly — the executor framework handles job
submission. A few use `current_cluster()` for manual job management.

### Queue Usage

`fray.queue` is used **only internally** within fray tests and the queue
module itself. No external consumer in marin, zephyr, or levanter uses it.
The RL system has its own rollout storage. Queue can likely be left as-is
or removed.

## Iris Equivalents

The Iris API (`lib/iris/src/iris/client/client.py`) provides:

| Fray Concept | Iris Equivalent |
|---|---|
| `Cluster.launch(JobRequest)` | `IrisClient.submit(entrypoint, name, resources)` → `Job` handle |
| `Cluster.wait(job_id)` | `Job.wait(timeout, raise_on_failure)` |
| `Cluster.poll(job_id)` | `Job.status()` / `Job.state` |
| `Cluster.terminate(job_id)` | `Job.terminate()` / `IrisClient.terminate(job_id)` |
| `Cluster.list_jobs()` | `IrisClient.list_jobs()` |
| `JobRequest.resources` | `ResourceSpec` (proto-based) |
| `JobRequest.entrypoint` | `Entrypoint` (callable serialization) |
| `JobContext.create_actor()` | No direct equivalent — actors are Iris jobs with endpoint registration |
| `JobContext.put/get/run/wait` | No equivalent — Iris is job-level, not task-level |

Key Iris features not in Fray:
- **Hierarchical job IDs** (parent/child namespaces)
- **Co-scheduling** (atomic multi-task scheduling)
- **Endpoint registry** (actor discovery via controller)
- **Task-level status/logs** (per-task within a job)
- **Port allocation** for actor servers

## Usage Frequency Summary

| API | Files Using It | Workloads |
|---|---|---|
| `ResourceConfig` | ~120 | All experiments, training, eval |
| `current_cluster()` / `Cluster.launch/wait` | ~15 | Training, RL, evaluation |
| `JobContext.put/get/run/wait` | ~10 | Zephyr, processing |
| `JobContext.create_actor` | ~6 | RL (curriculum, weight transfer) |
| `fray.queue` | ~3 | Internal only |
| `run_on_pod` (TPU gang sched) | ~5 | Training, RL, evaluation |

## What Fray-Lite Must Support

Based on the analysis, fray-lite needs exactly three capabilities:

### 1. Job Submission & Monitoring

Submit jobs with resource requirements and wait for completion. Used by
training, RL, and evaluation code. This is the highest-traffic API
(~120 files use `ResourceConfig`).

### 2. Task Dispatch (put/get/run/wait)

Distribute work across a pool of workers within a single job. Used by
Zephyr for data processing.

### 3. Named Actors

Create named, persistent actor instances that multiple workers can
discover and call. Used by RL for curriculum management and weight
transfer coordination.

## What Fray-Lite Can Drop

- `fray.queue` — unused externally, RL has its own storage
- `run_on_pod` / TPU gang scheduling internals — these become backend
  implementation details (Ray's gang scheduling or Iris's co-scheduling)
- `LocalCluster` subprocess isolation (`use_isolated_env`) — rarely used
- `TemporaryVenv` — only used by isolated env mode
- `FakeProcess` internals — implementation detail of LocalCluster

## Key Design Tensions

### 1. Actor model mismatch

Ray actors are in-process objects living inside the Ray cluster, accessed via
`actor.method.remote()`. Iris actors are HTTP services running in job
containers, discovered via endpoint registry. These are fundamentally different
models.

The RL code relies heavily on the Ray actor model:
- `get_if_exists=True` for singleton discovery
- `preemptible=False` for persistence
- `.remote()` / `ctx.get()` for invocation

An Iris backend would need to either:
(a) Run actors as Iris jobs with HTTP endpoints and translate `.remote()` calls
    to RPC, or
(b) Provide a separate actor runtime outside of Iris job management.

### 2. Task dispatch scope

Zephyr's `JobContext` (put/get/run/wait) dispatches tasks within a single
process/cluster. Under Ray, tasks run on Ray workers. Under Iris, there's no
equivalent — Iris manages jobs, not fine-grained tasks.

Options:
(a) Keep `JobContext` as-is (thread/sync/ray backends) — it's orthogonal to
    job management and doesn't need Iris support.
(b) Add an Iris-based task dispatch mechanism.

Since Zephyr data processing currently runs within a single Ray job (not as
separate Iris jobs), option (a) seems correct.

### 3. ResourceConfig compatibility

Fray's `ResourceConfig` is used by ~120 experiment files. Iris has its own
`ResourceSpec` (proto-based). These need to be interconvertible without
requiring changes to all experiment files.

### 4. Backend selection

Currently: `FRAY_CLUSTER_SPEC=ray?namespace=...` or `local`. Adding Iris:
`iris?controller=http://...`. The `current_cluster()` pattern should continue
to work transparently.
