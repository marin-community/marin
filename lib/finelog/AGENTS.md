# Finelog Agent Notes

Standalone log store + log service. Originally lifted out of `lib/iris`
(`iris/cluster/log_store/` and `iris/log_server/`); see the design plan at
`.agents/projects/2026-04-27_finelog_lift.md` (if present) or the original
extraction PR for context.

Start with the shared instructions in `/AGENTS.md`. Finelog-specific notes:

## Operations

Finelog has no `OPS.md`. To deploy, roll out, or roll back a finelog server
(health-gated rollout with auto-rollback via `scripts/safe_deploy.py`), or to
query archived parquet that has evicted to GCS, see
`.agents/runbooks/finelog-rollout-rollback.md`.

## Source Layout

- `src/finelog/proto/logging.proto` — log-service RPC definitions (package `finelog.logging`)
- `src/finelog/proto/finelog_stats.proto` — stats-service RPC definitions (package `finelog.stats`)
- `src/finelog/rpc/` — generated `_pb2`/`_connect` modules
- `src/finelog/types.py` — shared types: `LogReadResult`, `LogWriterProtocol`, key-related constants
- `src/finelog/store/` — `MemStore` (in-memory) and `DuckDBLogStore` (Parquet + DuckDB)
- `src/finelog/server/` — `LogServiceImpl`, `StatsServiceImpl`, ASGI builder, CLI launcher
- `src/finelog/client/` — `LogClient` (single user-facing entry; covers logs and stats),
  `RemoteLogHandler`, error types in `errors.py`. `proxy.py` hosts
  `LogServiceProxy`, an internal server-side adapter used when iris mounts the
  log service as a forwarding proxy; not re-exported.
- `tests/` — store + server tests
- `deploy/` — Dockerfile, k8s manifests, GCP snippets

## Boundaries

- Finelog has no `iris.*` imports. Iris-specific helpers (`worker_log_key`,
  `task_log_key`, `build_log_source`, anything that takes `JobName`/`TaskAttempt`)
  live under `iris/cluster/log_store_helpers.py` and call into finelog with opaque
  string keys.
- Finelog ships **no auth** in its server. Deployments secure the network
  layer (k8s NetworkPolicy, GCP firewall, VPC). If iris needs auth on top,
  it composes interceptors itself when launching the server.
- Keys are opaque strings. Any structure (`/system/...`, `/user/<job>/<task>:<attempt>`)
  is iris-side convention; finelog does not parse keys.

## Packaging

Finelog ships as two PyPI dists, released in lockstep by
`finelog-release-wheels.yaml`:

- `marin-finelog` — pure Python (this directory; hatchling).
- `marin-finelog-server` — the native in-process server ext, importable as
  top-level `finelog_server` (maturin project at `rust/`; the cdylib crate is
  `rust/pyext`). Only `src/finelog/embedded.py` imports it.

`marin-finelog` does **not** depend on `marin-finelog-server` at runtime — the
pure client never needs the in-process server. Consumers that do (the iris
controller) depend on `marin-finelog-server` explicitly. Here it is only a
`dev` dependency, pulled in for the embedded-server smoke test and the
dashboard demo.

By default the extension comes from the pre-built PyPI wheel, so in-dir
`uv run` never compiles Rust. To build it from source (live Rust dev), run
`python scripts/rust_mode.py dev` at the repo root — it points
`marin-finelog-server` at the local `rust/` tree in both the root and
`lib/finelog` pyprojects. Run `python scripts/rust_mode.py user` before
committing.

## Development

```bash
cd lib/finelog
uv run --group dev pytest --tb=short tests/
```

Regenerate protos after editing `proto/logging.proto`:

```bash
cd lib/finelog && buf generate
```
