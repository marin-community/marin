# Finelog Agent Notes

Standalone log store + log service. Originally lifted out of `lib/iris`
(`iris/cluster/log_store/` and `iris/log_server/`); see the design plan at
`.agents/projects/2026-04-27_finelog_lift.md` (if present) or the original
extraction PR for context.

Start with the shared instructions in `/AGENTS.md`. Finelog-specific notes:

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

## Development

```bash
cd lib/finelog
uv run --group dev pytest -n1 --tb=short tests/
```

Regenerate protos after editing `proto/logging.proto`:

```bash
cd lib/finelog && buf generate
```
