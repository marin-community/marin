# Iris Agent Notes

Distributed job orchestration for Marin. Start with the shared instructions in `/AGENTS.md`; only Iris-specific conventions are below.

## Key Docs

- `README.md` — overview + quick start
- `OPS.md` — operating / troubleshooting a live cluster (also used by skills: `debug-infra`, `restart-iris-controller`)
- `TESTING.md` — testing policy, markers, and commands
- `docs/task-states.md` — task state machine + retry semantics
- `docs/coreweave.md` — CoreWeave platform + `runtime=kubernetes` behavior
- `docs/image-push.md` — multi-region image push/pull architecture

Archived design docs (implemented, read code instead): `.agents/projects/2026*_iris_*.md`

## Source Layout

- `src/iris/cli/` — CLI entry point (`main.py` has all commands including `login`, `submit`, `status`)
- `src/iris/cluster/controller/` — controller server: `service.py` (RPC handlers), `controller.py` (main loop), `auth_setup.py` (auth config), `dashboard.py` (dashboard serving), `db.py` (SQLite), `migrations/` (schema)
- `src/iris/cluster/worker/` — worker agent
- `src/iris/rpc/` — protobuf definitions (`.proto`), generated code (`_pb2.py`), and RPC client helpers (`cluster_connect.py`, `auth.py`)
- `dashboard/` — Vue 3 frontend (Vite + Tailwind)

## Development

```bash
# Unit tests (run from lib/iris/)
cd lib/iris && uv run --group dev python -m pytest --tb=short -m 'not slow and not docker and not e2e' tests/
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

The controller uses **SQLAlchemy Core** end to end. There is no legacy
hand-rolled SQLite layer — all reads, writes, and projections build SA Core
expressions against tables defined in `controller/schema.py`.

- **Schema:** SA Core `Table` objects + `Index` declarations live in
  `controller/schema.py`. TypeDecorators (`JobNameType`, `WorkerIdType`,
  `TimestampMsType`, `BoolIntType`, `CachedProto`) adapt Python types to SQL.
  To add a column, edit `schema.py` **and** add a Python migration under
  `controller/migrations/`. Migrations are the source of truth for on-disk
  DDL; `schema.py` is the source for query generation. Migration 0001
  bootstraps a fresh DB from `schema.metadata` via `CreateTable(...,
  if_not_exists=True)` so the SA model and on-disk DDL never diverge.
- **Reads:** module-level functions in `controller/reads.py` taking
  `tx: db.Tx` as the first argument and returning SA `Row` objects (or
  `Sequence[Row]`). Callers use `row.column_name` attribute access; there
  are no wrapper dataclasses. Hot-path readers use `select(table.c.col)`
  directly; ad-hoc composites (dashboard, recursive CTEs) use `text(...)`.
- **Writes:** module-level functions in `controller/writes.py`,
  decorated with `@writes_to(*tables, cascades_into=())`. The decorator is
  pure metadata; `assert_owned_tables_not_externally_written()` runs at
  `ControllerDB.__init__` and rejects any write into a Projection-owned
  table from outside the owning Projection. Cascade hooks (e.g.
  `worker_attrs.invalidate_for_worker`) are called inline by the write
  function and registered via `tx.register(...)` so the dict update fires
  under the write lock after commit. Use `sqlalchemy.dialects.sqlite.insert`
  for UPSERT (`on_conflict_do_update` / `on_conflict_do_nothing`).
- **Projections (`endpoints`, `worker_attributes`):** write-through caches in
  `controller/projections/<name>.py`. Never write to these tables outside
  the projection. Read methods take no `tx` and serve latest-committed
  state from the in-memory dict. Mutating methods register post-commit
  hooks for atomic dict updates.
- **Transactions:** `db.write_transaction(engine, lock)` for writes (holds
  the lock across COMMIT + post-commit hooks); `db.read_snapshot(engine)`
  for reads (pooled query-only conns, no lock). `ControllerDB.transaction()`
  / `.read_snapshot()` are thin wrappers around those. `Tx.execute()` only
  accepts SA Core constructs — raw SQL strings raise `TypeError`. Use
  `sqlalchemy.text(...)` if you genuinely need literal SQL.
- **Returning rows:** SA `Row` objects from `select(...)` are the canonical
  return type. There are no wrapper dataclasses to construct. Use
  `row._mapping` if you need dict access; otherwise just `row.column_name`.

Design context: `.agents/projects/20260511_iris_store_view_refactor_v2.md`
and the migration enumeration at
`.agents/projects/20260512_iris_rewiring_enumeration.md`.

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

Profiles in particular: the worker drives a 10-minute periodic CPU capture loop and writes rows to `iris.profile`. On-demand captures (cpu/memory/thread) flow through the same RPC path the dashboard's "Profile now" buttons use: controller → provider → worker (or `K8sTaskProvider`) → finelog. The controller writes its own row for `/system/controller` self-captures only. See `lib/iris/OPS.md` for retention and example queries.

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

Resource model: CPU demand is fungible and can route to any group; GPU/TPU demand is non-fungible and must match device type (and optionally variant).

The controller is a plain GCE VM (or K8s Deployment on CoreWeave) with no zone affinity to workers. See `docs/coreweave.md` for CoreWeave-specific deployment topology and `docs/image-push.md` for the GHCR → AR remote repo image pipeline.
