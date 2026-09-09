# Iris Agent Notes

Distributed job orchestration for Marin. Start with the shared instructions in `/AGENTS.md`; only Iris-specific conventions are below.

## Key Docs

- `README.md` — overview + quick start
- `OPS.md` — operating / troubleshooting a live cluster (also used by skills: `debug`, `use-iris`)
- Echo — durable incident and debugging records; use `write-ops-log` after an
  infrastructure investigation and link the canonical Echo URL
- `TESTING.md` — testing policy, markers, and commands
- `docs/architecture.md` — controller, backend, persistence, and source layout
- `docs/task-states.md` — task state machine + retry semantics
- `docs/coreweave.md` — CoreWeave platform + `runtime=kubernetes` behavior
- `docs/federation.md` — peer routing, root-job-only handoff, and cross-cluster storage
- `docs/image-push.md` — multi-region image push/pull architecture

Archived design docs (implemented, read code instead): `.agents/projects/2026*_iris_*.md`

## Development

```bash
# Full safe Iris unit suite
uv run --package marin-iris --group test pytest --tb=short lib/iris/tests/
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
  changing persisted schema. A migration only ever runs against a DB created
  before it: a fresh DB is materialized from `schema.py` and records every
  migration as already applied. So write each one to carry the *previous* schema
  forward, and to be re-runnable after a mid-migration crash — never to depend on
  the current `schema.py`, which will keep moving.
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
- **Cluster-wide literals**: `defaults.task_env` in the cluster config — injected into every task container.
- **Cluster-wide from operator shell**: `defaults.inject_env` — a list of env var *names* captured from the operator's shell at `iris cluster start` and injected into every task (and the controller). A missing name aborts the launch. On Kubernetes the values go to the `iris-task-env` Secret and are projected via `envFrom` (they never enter the ConfigMap); on GCP/VM clusters they are folded into `task_env` in the bootstrap config. See `iris.cluster.inject_env`.

Key behaviors:
- `HF_TOKEN`, `WANDB_API_KEY`, `HF_DATASETS_TRUST_REMOTE_CODE`, and `TOKENIZERS_PARALLELISM` are auto-injected from the submitter's env by `EnvironmentSpec.to_proto()`.
- `defaults.inject_env` values are *defaults*: a literal `defaults.task_env` entry of the same name and a per-job `-e`/`env_vars` both override them.
- Child jobs inherit parent env vars automatically (child values take precedence).
- The CLI also loads env vars from `.marin.yaml`'s `env:` section.
- The submitting user for top-level jobs resolves as: explicit `user`/`--user` → `IRIS_USER` env var → the enclosing job's user → OS user → `root` (`resolve_job_user`). Export `IRIS_USER` when your OS username is uninformative (e.g. a shared `marin` account). Submissions from inside a job become child jobs of the enclosing job and skip this resolution entirely.

See https://github.com/marin-community/marin/issues/3859 for context.

## Task Setup

Before the command runs, the worker executes a list of setup scripts to prepare
the environment. The default is `uv sync --all-packages --no-dev`;
`sync_packages`/`--sync-package` scopes it to named workspace members. The
worker is pure mechanism; the list is resolved client-side from
`EnvironmentSpec.setup_scripts` — `None` for the default, `[]` to skip setup
(bring-your-own image), or a verbatim list — and iris always appends its own
runtime-deps step. The script builders, the `IRIS_*` env scripts parameterize
against (notably `$IRIS_VENV`, the venv the run phase activates), child
inheritance, and the Docker gotcha (setup runs in a separate container, so
`export` does not reach the command — use `env_vars`) all live in
`iris.cluster.setup_scripts`. See https://github.com/marin-community/marin/issues/6595.
