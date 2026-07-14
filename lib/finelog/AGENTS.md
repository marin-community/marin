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
  `RemoteLogHandler`, error types in `errors.py`.
- `tests/` — store + server tests
- `deploy/` — Dockerfile, k8s manifests, GCP snippets

## Boundaries

- Finelog has no `iris.*` imports. Iris-specific helpers (`worker_log_key`,
  `task_log_key`, `build_log_source`, anything that takes `JobName`/`TaskAttempt`)
  live under `iris/cluster/log_store_helpers.py` and call into finelog with opaque
  string keys.
- Finelog's server gates every RPC with an authenticated-ingress stack
  (`rust/src/server/auth.rs`): an ordered, **default-deny** list of `cidr` and
  `jwt` layers (`FINELOG_AUTH_POLICY`), defaulting to loopback-only. The `jwt`
  layer verifies **EdDSA (Ed25519)** tokens with `aud="finelog"` against each
  cluster's inline **public** key(s) — a sending finelog signs with its private
  key, the receiving one holds only the public half. Deployments still secure the
  network layer (k8s NetworkPolicy, GCP firewall, VPC) as defense in depth. The
  policy schema lives in `deploy/config.py`.
- **The admitting layer names the caller.** A `jwt` layer's matched key binds the
  request to that key's `cluster`; a `cidr` match binds to nothing (see
  `AuthIdentity` in `rust/src/server/auth.rs`). `PushLogs` stamps each row with the
  authenticated cluster. Cross-cluster forwarding instead writes through the generic
  `WriteRows` path and stamps the origin into the row itself, so the `cluster` column
  is a label, not a trust boundary — any admitted sender can write any cluster's rows.
- Keys are opaque strings. Any structure (`/system/...`, `/user/<job>/<task>:<attempt>`)
  is iris-side convention; finelog does not parse keys.

## Cross-cluster forwarding

A per-cluster finelog ships its rows to a hub finelog itself; no other process
relays them. `forwarding:` in its deploy config names the hub, this cluster's
name, and a `rigging.secrets` reference to its Ed25519 private key; the hub adds
one `jwt` key entry per sender. Each server therefore owns a keypair, distinct
from the iris controller's signing key.

The forwarder (`rust/src/server/forwarding.rs`) forwards **every table**, not just
logs. Each poll it lists the live namespaces and, per namespace, reads the rows past
a durable per-`(target, namespace)` cursor (`forward_state` in the catalog) and ships
them through the generic `RegisterTable` + `WriteRows` (Arrow IPC) path — a namespace
the hub lacks is created there first. Rows of a table with a `cluster` column are
stamped with the origin and skipped if they already carry a foreign one, so a hub's
own relayed rows never loop. The cursor is durable, so a restart resumes rather than
replays.

Forwarding is **best-effort by construction**: the sending store holds the record,
the hub a convenience copy. A namespace that falls more than `MAX_FORWARD_LAG_SEQS`
behind skips ahead to its freshest window and logs the count it dropped, rather than
growing without bound; rows evicted before they shipped are skipped the same way. A
hub outage costs the hub rows, never the sender's memory or its own reads.

Only the k8s backend can forward — it projects the key through a Secret. The gcp
backend refuses, because its only channel to the server is world-readable
startup-script metadata.

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
