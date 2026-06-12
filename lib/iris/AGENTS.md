# Iris Agent Notes

Distributed job orchestration for Marin. Start with the shared instructions in `/AGENTS.md`; only Iris-specific conventions are below.

## Key Docs

- `README.md` — overview + quick start
- `OPS.md` — operating / troubleshooting a live cluster (also used by skills: `debug`, `restart-iris`)
- `TESTING.md` — testing policy, markers, and commands
- `docs/task-states.md` — task state machine + retry semantics
- `docs/coreweave.md` — CoreWeave platform + `runtime=kubernetes` behavior
- `docs/image-push.md` — multi-region image push/pull architecture

Archived design docs (implemented, read code instead): `.agents/projects/2026*_iris_*.md`

## Source Layout

- `src/iris/cli/` — CLI entry point (`main.py` has all commands including `login`, `submit`, `status`)
- `src/iris/cluster/controller/` — controller server: `service.py` (RPC handlers), `controller.py` (main loop), `backend.py` (the `TaskBackend` contract), `scheduling/` (`scheduler.py` + `policy.py`), `autoscaler/` (capacity), `auth_setup.py` (auth config), `dashboard.py` (dashboard serving), `db.py` (SQLite), `migrations/` (schema)
- `src/iris/cluster/backends/` — `TaskBackend` implementations (`rpc/backend.py` = `RpcTaskBackend`, `k8s/tasks.py` = `K8sTaskProvider`) plus machine-lifecycle providers (`gcp`, `k8s`, `local`, `manual`)
- `src/iris/cluster/worker/` — worker agent
- `src/iris/rpc/` — protobuf definitions (`.proto`), generated code (`_pb2.py`), and RPC client helpers (`cluster_connect.py`, `auth.py`)
- `dashboard/` — Vue 3 frontend (Vite + Tailwind)

## Development

```bash
# Unit tests (run from lib/iris/)
cd lib/iris && uv run --group dev python -m pytest --tb=short -m 'not slow and not docker and not requires_cluster' tests/
```

See `TESTING.md` for the complete testing policy, E2E test commands, and markers.

### Dashboard

The Vue 3 dashboard lives in `dashboard/`. To type-check and build:

```bash
cd lib/iris/dashboard && npm run build:check   # vue-tsc + rsbuild
```

Or use the Iris CLI which handles `npm ci` automatically:

```bash
uv run iris build dashboard
```

Always run `build:check` after editing `.vue` or `.ts` files to catch type errors before committing.

## Data Layer

The controller store uses SQLAlchemy Core. Read the code, not historical
design notes:

- `controller/schema.py` — table definitions and indexes.
- `controller/migrations/` — on-disk schema changes. Add a migration whenever
  changing persisted schema.
- `controller/db.py` — engine setup, transaction wrappers, and `Tx.execute`.
- `controller/reads.py` / `controller/writes.py` — shared read/write helpers.
- `controller/projections/` — write-through caches; do not write projection
  tables from outside their owning projection.

Prefer existing `reads.py`/`writes.py` helpers before adding new query code.
Use SQLAlchemy result APIs directly (`.first()`, `.all()`, `.scalar()`); do
not add wrapper methods that duplicate SQLAlchemy. Define row protocols or
dataclasses at the usage boundary when a caller needs a typed shape.

## Code Conventions

- Use Connect/RPC for APIs and dashboards. Do not use `httpx` or raw HTTP.
- After changing `.proto` files, regenerate from the repo root with `uv run python lib/iris/scripts/generate_protos.py`.
- Prefer shallow, functional code that returns control quickly; avoid callback-heavy or inheritance-driven designs.
- Dashboards must be a thin UI over the RPC API, not a second implementation path.
- Use `rigging.timing` for all time-related operations (`Timestamp`, `Duration`, `Deadline`, `Timer`, `ExponentialBackoff`) instead of raw `datetime` or `time`.
- Use `concurrent.futures.ThreadPoolExecutor` (not asyncio) for concurrent platform operations, with hard timeouts.
- Avoid `TYPE_CHECKING`. Use real imports. If you hit a cycle, prefer refactoring or use a `Protocol` at the boundary.
- Prefer spiral plans: each stage should be independently testable (proto → server stub → client wiring → end-to-end test).

### Decisions vs measurements

The controller SQLite DB stores the *registry and decisions*: worker liveness verdict, task↔worker assignments, scheduling state. Time-series *measurements* (per-tick utilization, per-attempt resource snapshots, profile captures) live in the finelog stats namespaces (`iris.worker`, `iris.task`, `iris.profile`) and are queried via the controller-bundled StatsService. New columns that record measurements should be added as stats namespaces, not controller tables.

Profiles in particular: the worker drives a 10-minute periodic CPU capture loop and writes rows to `iris.profile`. On-demand captures (cpu/memory/thread) flow through the same RPC path the dashboard's "Profile now" buttons use: controller → `TaskBackend.profile_task` (`RpcTaskBackend` forwards to the worker daemon; `K8sTaskProvider` runs `kubectl exec`) → finelog. The controller writes its own row for `/system/controller` self-captures only. See `lib/iris/OPS.md` for retention and example queries.

## Environment Variables

Never use `os.environ` to pass env vars to Iris jobs. Tasks run in Docker containers — the submitter's process environment is not available inside the container.

Use Iris's built-in mechanisms instead:

- **CLI**: `iris job run -e KEY VALUE -- python script.py`
- **SDK**: `EnvironmentSpec(env_vars={"KEY": "value"})` passed to `client.submit(environment=...)`

Key behaviors:
- `HF_TOKEN`, `WANDB_API_KEY`, `HF_DATASETS_TRUST_REMOTE_CODE`, and `TOKENIZERS_PARALLELISM` are auto-injected from the submitter's env by `EnvironmentSpec.to_proto()`.
- Child jobs inherit parent env vars automatically (child values take precedence).
- The CLI also loads env vars from `.marin.yaml`'s `env:` section.

See https://github.com/marin-community/marin/issues/3859 for context.

## Architecture Notes

### The TaskBackend contract

A `TaskBackend` (`controller/backend.py`) is the control-plane driver for ONE
cluster. It implements one uniform set of phase methods — `schedule` (pure
placement decision), `reconcile` (backend I/O: task observations + per-worker
health events), `autoscale` (provision, or tear down dead workers' slices +
healthy siblings) — plus the on-demand one-offs (`get_process_status`,
`profile_task`, `exec_in_container`). Each phase returns its own frozen result
type: `ScheduleResult`, `ReconcileResult`, `AutoscaleResult`. The controller is a
thin dispatcher: it owns the database and the loop cadences, and each loop reads
a DB snapshot → calls one backend method → commits the returned result
(dispatching within a method on which result field is non-empty, never by
`isinstance`). **The contract is DB-less**: backends take plain data in and
return plain data out; they never touch the controller DB.

A backend declares `capabilities: frozenset[BackendCapability]` — metadata the
dashboard and on-demand RPC routing key on. The controller calls all three
phases uniformly regardless, with one per-tick exception: `CLUSTER_VIEW` makes
the controller drain the dispatch queue (a DB write it owns) into that backend's
reconcile snapshot. The flags: `WORKER_DAEMON` (`"workers"`), `IRIS_AUTOSCALER`
(`"autoscaler"`), `CLUSTER_VIEW` (`"cluster"`).

Worker health is OBSERVED only by worker-daemon backends — REACHED / UNREACHABLE
events on `ReconcileResult.health_events` — and OWNED by the controller, which
folds them (together with BUILD_FAILED events it synthesizes from the reconcile
kernel's effects) through the single `WorkerHealthTracker.apply` site; a worker
over the failure threshold is reaped via `autoscale(dead_workers=...)`.
There is no ping loop and no separate liveness channel — the reconcile RPC
outcome is the only liveness signal. Cluster-view backends (Kubernetes) have no
Iris workers, so they emit no health events; pod status flows back as neutral
task `updates`.

Two implementations satisfy it: `RpcTaskBackend` (`backends/rpc/backend.py`,
`{WORKER_DAEMON, IRIS_AUTOSCALER}`, owns the `Scheduler` + `Autoscaler`) for
GCP/TPU, CoreWeave bare-metal, manual, and local; and `K8sTaskProvider`
(`backends/k8s/tasks.py`, `{CLUSTER_VIEW}`) for Kubernetes (Kueue schedules, the
cluster autoscaler provisions, so its `schedule`/`autoscale` are no-ops). The
contract type lives in `controller/backend.py`; see `docs/architecture.md` "The
TaskBackend contract".

Resource model: CPU demand is fungible and can route to any group; GPU/TPU demand is non-fungible and must match device type (and optionally variant).

The controller is a plain GCE VM (or K8s Deployment on CoreWeave) with no zone affinity to workers. See `docs/coreweave.md` for CoreWeave-specific deployment topology and `docs/image-push.md` for the GHCR → AR remote repo image pipeline.
