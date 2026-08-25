---
name: query-finelog
description: Query Finelog logs and telemetry for Iris tasks, workers, profiles, training, vLLM, and cross-cluster forwarding. Use for schema discovery, SQL, memory or CPU summaries, hub-versus-regional comparisons, counter semantics, and query-performance diagnosis.
---

# Query Finelog

Read `lib/finelog/OPS.md` for access and query behavior. Read `lib/iris/OPS.md` under `Stats Namespaces` for Iris namespace meanings.

Discover before querying; do not assert remembered columns:

```bash
uv run finelog namespaces <deployment>
uv run finelog schema <deployment> <namespace>
uv run finelog query <deployment> --format table <<'SQL'
<bounded SQL using schema-confirmed columns>
SQL
```

`finelog query` reads SQL from stdin when the positional SQL argument is omitted.

Use `marin` for the federated view and a regional deployment for peer-local truth or recent rows that may not have forwarded. Preserve `cluster` and full process/label identity until after per-series delta calculations.

Bound the native time key. Keep `telemetry_v1.timestamp_ms` predicates numeric. Treat current snapshots as values, imported Prometheus counters as cumulative snapshots with `LAG` and reset handling, and native Rigging counters as deltas to `SUM` directly.

Never reset or change a shared namespace during diagnosis. Return the deployment, namespace, time window, query, series semantics, and any forwarding or retention caveat.

## Examples

Confirm every schema before adapting an example. Angle-bracket values are placeholders.

### Iris task memory by half-hour

Adapted from [Echo wiki 230](https://echo.oa.dev/wiki/230). The dashboard task ID includes the final task index. Select or group by `attempt_id` after retries.

```sql
SELECT date_bin(INTERVAL '30 minutes', ts,
                TIMESTAMP '1970-01-01 00:00:00') AS bucket_start_utc,
       count(*) AS samples,
       round(min(memory_mb) / 1024.0, 1) AS min_gib,
       round(median(memory_mb) / 1024.0, 1) AS median_gib,
       round(max(memory_mb) / 1024.0, 1) AS max_gib,
       round(max(memory_peak_mb) / 1024.0, 1) AS attempt_peak_gib
FROM "iris.task"
WHERE task_id = '/user/job/task'
  AND attempt_id = 0
GROUP BY bucket_start_utc
ORDER BY bucket_start_utc
```

`memory_mb` is sampled current memory; `memory_peak_mb` is the attempt's cumulative peak. Values are MiB despite the names. `count(*)` exposes partial buckets and gaps. Query the regional deployment if recent hub rows appear incomplete.

### Missing federated logs

Use the exact attempt-suffixed log key on both stores:

```sql
-- marin hub
SELECT seq, epoch_ms, source, data, cluster
FROM "log"
WHERE key = '/user/job/task:0' AND cluster = 'cw-us-east-08a'
ORDER BY seq;

-- cw-us-east-08a regional deployment
SELECT seq, epoch_ms, source, data
FROM "log"
WHERE key = '/user/job/task:0'
ORDER BY seq;
```

Regional rows with a missing or shorter hub result mean forwarding delay. Runtime task logs with no regional rows point to shipper or regional ingest. Iris `job describe` remains the liveness source.

### Native delta counter

Native `rigging.telemetry.counter(...).add(...)` rows are already increments:

```sql
SELECT sum(value)
FROM telemetry_v1
WHERE name = 'requests_completed'
  AND timestamp_ms >= <start_ms>
  AND timestamp_ms < <end_ms>
```

### Imported cumulative counter

Imported vLLM counters carry `source_temporality = 'cumulative_snapshot'`. Scan one 15-second scrape before the visible window, preserve the complete series identity, and discard reset intervals.

```sql
WITH base AS (
  SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
         service, name, resource_attributes_json, attributes_json,
         timestamp_ms, seq, value
  FROM telemetry_v1
  WHERE service = 'vllm'
    AND name = 'generation_tokens_total'
    AND json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
    AND timestamp_ms >= <start_ms - 15000>
    AND timestamp_ms < <end_ms>
), samples AS (
  SELECT *, lag(value) OVER (
    PARTITION BY origin_cluster, service, name,
                 resource_attributes_json, attributes_json
    ORDER BY timestamp_ms, seq) AS previous_value
  FROM base
), deltas AS (
  SELECT *, CASE WHEN previous_value IS NULL OR value < previous_value
                 THEN NULL ELSE value - previous_value END AS delta
  FROM samples
)
SELECT date_bin(INTERVAL '5 minutes', to_timestamp_millis(timestamp_ms)) AS bucket,
       sum(delta) AS generated_tokens
FROM deltas
WHERE timestamp_ms >= <start_ms>
GROUP BY bucket
ORDER BY bucket
```
