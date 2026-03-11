# Iris Agent Notes

Distributed job orchestration replacing Ray with simpler primitives. Start with the shared instructions in `/AGENTS.md`; only Iris-specific conventions are below.

## Key Docs

- `README.md` — overview + quick start
- `OPS.md` — operating / troubleshooting a live cluster
- `TESTING.md` — testing policy, markers, and commands
- `docs/autoscaler-v2.md` — autoscaler design + terminology
- `docs/controller-flow.md`, `docs/worker-flow.md` — controller/worker lifecycle
- `docs/task-states.md` — task state machine + retry semantics
- `docs/coreweave.md` — CoreWeave platform + `runtime=kubernetes` behavior
- `docs/image-push.md` — multi-region image push/pull architecture
- `docs/constraints.md` — constraint system design

## Development

```bash
# Unit tests
uv run pytest lib/iris/tests/ -m "not e2e" -o "addopts="
```

See `TESTING.md` for the complete testing policy, E2E test commands, and markers.

## Code Conventions

- Use Connect/RPC for APIs and dashboards. Do not use `httpx` or raw HTTP.
- After changing `.proto` files, regenerate via `scripts/generate_protos.py`.
- Prefer shallow, functional code that returns control quickly; avoid callback-heavy or inheritance-driven designs.
- Dashboards must be a thin UI over the RPC API, not a second implementation path.
- Use `iris.time_utils` for all time-related operations (`Timestamp`, `Duration`, `Deadline`, `Timer`, `ExponentialBackoff`) instead of raw `datetime` or `time`.
- Use `concurrent.futures.ThreadPoolExecutor` (not asyncio) for concurrent platform operations, with hard timeouts.
- Avoid `TYPE_CHECKING`. Use real imports. If you hit a cycle, prefer refactoring or use a `Protocol` at the boundary.
- Prefer spiral plans: each stage should be independently testable (proto → server stub → client wiring → end-to-end test).

## Architecture Notes

Resource model: CPU demand is fungible and can route to any group; GPU/TPU demand is non-fungible and must match device type (and optionally variant).

The controller is a plain GCE VM (or K8s Deployment on CoreWeave) with no zone affinity to workers. See `docs/coreweave.md` for CoreWeave-specific deployment topology and `docs/image-push.md` for the GHCR → AR remote repo image pipeline.
