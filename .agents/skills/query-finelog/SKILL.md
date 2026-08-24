---
name: query-finelog
description: Query Finelog logs and telemetry for Iris tasks, workers, profiles, training, vLLM, and cross-cluster forwarding. Use for schema discovery, SQL, memory or CPU summaries, hub-versus-regional comparisons, counter semantics, and query-performance diagnosis.
---

# Query Finelog

Read `lib/finelog/OPS.md` for access and query behavior. Read `lib/iris/OPS.md` under `Stats Namespaces` for Iris namespace meanings. Use [references/examples.md](references/examples.md) for worked queries.

Discover before querying; do not assert remembered columns:

```bash
uv run finelog namespaces <deployment>
uv run finelog schema <deployment> <namespace>
uv run finelog query <deployment> --format table <<'SQL'
<bounded SQL using schema-confirmed columns>
SQL
```

Use `marin` for the federated view and a regional deployment for peer-local truth or recent rows that may not have forwarded. Preserve `cluster` and full process/label identity until after per-series delta calculations.

Bound the native time key. Keep `telemetry_v1.timestamp_ms` predicates numeric. Treat current snapshots as values, imported Prometheus counters as cumulative snapshots with `LAG` and reset handling, and native Rigging counters as deltas to `SUM` directly.

Never reset or change a shared namespace during diagnosis. Return the deployment, namespace, time window, query, series semantics, and any forwarding or retention caveat.
