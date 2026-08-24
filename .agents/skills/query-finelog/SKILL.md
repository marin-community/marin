---
name: query-finelog
description: Discover and query Finelog namespaces for Iris logs, task and worker measurements, profiles, training telemetry, vLLM metrics, and federated forwarding diagnosis. Use when answering a question requires Finelog SQL, schema discovery, hub-versus-regional comparison, counter or gauge semantics, or query-performance triage.
---

# Query Finelog

Read the relevant section of `lib/finelog/OPS.md` before querying. For Iris namespace meanings and controller/measurement boundaries, also read `lib/iris/OPS.md` under `Stats Namespaces`.

## Discover before querying

Do not guess a namespace schema or reuse columns from a different namespace.

```bash
uv run finelog namespaces <deployment>
uv run finelog schema <deployment> <namespace>
```

Use the returned explicit and implicit columns to form the query. Until the actual schema output is available, keep column names as explicit placeholders; do not assert that a namespace currently contains remembered columns. Feed multiline SQL on stdin and quote dotted namespace names:

```bash
uv run finelog query <deployment> --format table <<'SQL'
SELECT ...
FROM "iris.task"
WHERE ...
SQL
```

Authenticate with `uv run iris --cluster=<name> login` when the deployment uses the Iris IAP endpoint.

## Choose the data source

- Use `marin` for the federated fleet view and most cross-cluster telemetry questions. Preserve the `cluster` origin in grouping and series identity.
- Query a regional deployment for peer-local truth, recent rows that may not have forwarded, or a task explicitly scoped to that cluster.
- For missing federated logs, query the exact attempt-suffixed key on both the regional store and `marin`. Regional rows with missing or partial hub rows indicate forwarding delay.
- Finelog forwarding is asynchronous and does not replace Iris task state. Use Iris `job describe` or `task describe` for liveness.

## Bound and preserve semantics

- Select a time bound on the namespace's native key. Keep `telemetry_v1.timestamp_ms` predicates numeric and use `to_timestamp_millis` only for projection or bucketing.
- Prefer structured `telemetry_v1` columns such as `run_id`, `job_id`, `execution_uid`, `region`, `node_name`, and `process_index`; inspect JSON attributes only for dimensions not promoted to columns.
- Keep every series-identity dimension until after computing deltas. Federation origin, process/attempt identity, and metric labels commonly distinguish replicas or reset epochs.
- Treat gauges/current snapshots as values. Imported Prometheus counters and histogram components are cumulative snapshots and require ordered deltas with reset handling. Native Rigging counters are already deltas and must be summed directly.
- Run `EXPLAIN ANALYZE` only after bounding a slow query. Never reset or change a shared namespace as part of read-only diagnosis.

For vLLM metric names, identity, reset handling, and example SQL, read [references/vllm.md](references/vllm.md).

Return the deployment, namespace, time window, query, row/series semantics, and any incompleteness caused by forwarding or retention.
