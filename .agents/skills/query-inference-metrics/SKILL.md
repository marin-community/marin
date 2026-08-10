---
name: query-inference-metrics
description: Investigate a vLLM serve's performance (throughput, TTFT/TPOT latency, queue depth, KV-cache usage) from durable Finelog telemetry with SQL. Use when asked how an inference or eval serve performed, or why it was slow. For kernel-level JAX profiling, use profile-training instead.
---

# Query inference telemetry

Native vLLM `/metrics` snapshots are exported directly to Finelog every 15 seconds with `service = 'vllm'`. They land in `telemetry_v1`; the redundant `vllm:` metric prefix is removed.

Run SQL from a Marin checkout:

```sh
uv run finelog query marin "<SQL>" --format table
```

Use the federated `marin` hub unless a task specifically requires a cluster-local view.

## Row shape and identity

The useful typed columns are `cluster`, `service`, `name`, `kind`, `value`, `unit`, `timestamp_ms`, and `seq`. Process identity is string JSON in `resource_attributes_json`; source labels and the two snapshot markers are string JSON in `attributes_json`. Read both with untyped `json_get`, for example:

```sql
json_get(resource_attributes_json, 'job_id')
json_get(attributes_json, 'model_name')
```

`json_get` returns `VARCHAR`. Do not use typed JSON getters. Use `to_timestamp_millis(timestamp_ms)` for projection or `date_bin`, but keep range predicates on raw `timestamp_ms` so Parquet min/max pruning applies.

A complete imported-snapshot series is identified by `(origin_cluster, service, name, resource_attributes_json, attributes_json)`. The JSON columns contain worker/attempt and metric labels, so do not aggregate them away before computing a cumulative delta.

## Snapshot and delta semantics

Imported vLLM samples are stored as telemetry gauges without changing the source value. `source_kind` preserves the Prometheus family type. Source gauges and summary quantiles carry `source_temporality = 'current_snapshot'`; aggregate them directly or select the latest value. Source counters, histogram bucket/sum/count samples, and summary sum/count samples carry `source_temporality = 'cumulative_snapshot'` and need `LAG`. Discard a negative delta at a process reset. Scan one 15-second scrape interval before the visible window so its first point has a predecessor, then filter output to the visible range.

```sql
WITH base AS (
  SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
         service, name, resource_attributes_json, attributes_json,
         timestamp_ms, seq, value
  FROM telemetry_v1
  WHERE service = 'vllm'
    AND name = 'generation_tokens_total'
    AND json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
    AND timestamp_ms >= 1785254385000 -- visible start minus one 15s scrape interval
    AND timestamp_ms < 1785258000000
), samples AS (
  SELECT *,
         LAG(value) OVER (
           PARTITION BY origin_cluster, service, name,
                        resource_attributes_json, attributes_json
           ORDER BY timestamp_ms, seq
         ) AS previous_value
  FROM base
), deltas AS (
  SELECT *, CASE
    WHEN previous_value IS NULL OR value < previous_value THEN NULL
    ELSE value - previous_value
  END AS delta
  FROM samples
)
SELECT date_bin(INTERVAL '5 minutes', to_timestamp_millis(timestamp_ms)) AS t,
       SUM(delta) AS generated_tokens
FROM deltas
WHERE timestamp_ms >= 1785254400000
GROUP BY 1 ORDER BY 1
```

This rule does not apply to native `rigging.telemetry.counter(...).add(...)` records: those rows are already deltas and must be aggregated with `SUM(value)`, never `LAG`.

Histogram buckets are cumulative snapshots. Their upper bound is `json_get(attributes_json, 'le')`; `_count` and `_sum` use the same full series identity. Derive a quantile from bucket deltas, or divide sum delta by count delta for a mean. Summary quantiles are current snapshots, while their `_count` and `_sum` components use the cumulative path.

## Triage a serve

First list the signals for one job:

```sql
SELECT name, json_get(attributes_json, 'source_kind') AS source_kind, COUNT(*) AS samples
FROM telemetry_v1
WHERE service = 'vllm'
  AND json_get(resource_attributes_json, 'job_id') = '/held/qwen3-evals-otbl-full-r2'
GROUP BY 1, 2 ORDER BY 1
```

For lifetime totals, sum the initial value and subsequent nonnegative deltas in each reset epoch, then sum replicas:

```sql
WITH lifetime_base AS (
  SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
         service, name, resource_attributes_json, attributes_json,
         timestamp_ms, seq, value
  FROM telemetry_v1
  WHERE service = 'vllm'
    AND json_get(resource_attributes_json, 'job_id') = '/held/qwen3-evals-otbl-full-r2'
    AND json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
    AND name IN ('prompt_tokens_total', 'generation_tokens_total',
                 'e2e_request_latency_seconds_count', 'e2e_request_latency_seconds_sum',
                 'num_preemptions_total')
), lifetime_samples AS (
  SELECT *,
         LAG(value) OVER (
           PARTITION BY origin_cluster, service, name,
                        resource_attributes_json, attributes_json
           ORDER BY timestamp_ms, seq
         ) AS previous_value
  FROM lifetime_base
), lifetime_increments AS (
  SELECT *, CASE
    WHEN previous_value IS NULL OR value < previous_value THEN value
    ELSE value - previous_value
  END AS increment
  FROM lifetime_samples
)
SELECT name, SUM(increment) AS total
FROM lifetime_increments
GROUP BY 1 ORDER BY 1
```

For `current_snapshot` saturation gauges, inspect peaks and averages rather than deltas:

```sql
SELECT name, ROUND(MAX(value), 3) AS peak, ROUND(AVG(value), 3) AS average
FROM telemetry_v1
WHERE service = 'vllm'
  AND json_get(resource_attributes_json, 'job_id') = '/held/qwen3-evals-otbl-full-r2'
  AND name IN ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc')
GROUP BY 1 ORDER BY 1
```

Queue depth, KV-cache usage, TTFT (`time_to_first_token_seconds_*`), TPOT (`inter_token_latency_seconds_*`), request completion labels, and token rates usually locate the bottleneck. Cross-check independent totals; disagreement commonly means replica identity was collapsed before delta calculation.
