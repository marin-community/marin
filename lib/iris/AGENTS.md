# Iris Agent Notes

Distributed job orchestration replacing Ray with simpler primitives. Start with the shared instructions in `/AGENTS.md`; only Iris-specific conventions are below.

## Key Docs

- `README.md` — overview + quick start
- `OPS.md` — operating / troubleshooting a live cluster
- `TESTING.md` — testing policy, markers, and commands
- `docs/autoscaler-fix.md` — autoscaler design + terminology
- `docs/task-states.md` — task state machine + retry semantics
- `docs/coreweave.md` — CoreWeave platform + `runtime=kubernetes` behavior
- `docs/image-push.md` — multi-region image push/pull architecture
- `docs/constraints.md` — constraint system design
- `docs/users.md` — user/auth system design

## Source Layout

- `src/iris/cli/` — CLI entry point (`main.py` has all commands including `login`, `submit`, `status`)
- `src/iris/cluster/controller/` — controller server: `service.py` (RPC handlers), `controller.py` (main loop), `auth_setup.py` (auth config), `dashboard.py` (dashboard serving), `db.py` (SQLite), `migrations/` (schema)
- `src/iris/cluster/worker/` — worker agent
- `src/iris/rpc/` — protobuf definitions (`.proto`), generated code (`_pb2.py`), and RPC client helpers (`cluster_connect.py`, `auth.py`)
- `dashboard/` — Vue 3 frontend (Vite + Tailwind)

## Development

```bash
# Unit tests
uv run pytest lib/iris/tests/ -m "not e2e" -o "addopts="
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

## Code Conventions

- Use Connect/RPC for APIs and dashboards. Do not use `httpx` or raw HTTP.
- After changing `.proto` files, regenerate from the repo root with `uv run python lib/iris/scripts/generate_protos.py`.
- Prefer shallow, functional code that returns control quickly; avoid callback-heavy or inheritance-driven designs.
- Dashboards must be a thin UI over the RPC API, not a second implementation path.
- Use `iris.time_utils` for all time-related operations (`Timestamp`, `Duration`, `Deadline`, `Timer`, `ExponentialBackoff`) instead of raw `datetime` or `time`.
- Use `concurrent.futures.ThreadPoolExecutor` (not asyncio) for concurrent platform operations, with hard timeouts.
- Avoid `TYPE_CHECKING`. Use real imports. If you hit a cycle, prefer refactoring or use a `Protocol` at the boundary.
- Prefer spiral plans: each stage should be independently testable (proto → server stub → client wiring → end-to-end test).

## Architecture Notes

Resource model: CPU demand is fungible and can route to any group; GPU/TPU demand is non-fungible and must match device type (and optionally variant).

The controller is a plain GCE VM (or K8s Deployment on CoreWeave) with no zone affinity to workers. See `docs/coreweave.md` for CoreWeave-specific deployment topology and `docs/image-push.md` for the GHCR → AR remote repo image pipeline.
