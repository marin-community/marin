---
name: query-inference-metrics
description: Investigate a vLLM serve's performance (throughput, TTFT/TPOT latency, queue depth, KV-cache usage) from durable Finelog telemetry with SQL. Use when asked how an inference or eval serve performed, or why it was slow. For kernel-level JAX profiling, use profile-training instead.
---

# Query inference telemetry

Run from a Marin checkout against the federated `marin` hub:

```bash
uv run finelog query marin "<SQL>" --format table
```

Useful columns are `cluster`, `service`, `name`, `kind`, `value`, `unit`,
`timestamp_ms`, and `seq`. Process identity and labels are JSON strings: use
`json_get(resource_attributes_json, '<key>')` and
`json_get(attributes_json, '<key>')`, never typed JSON getters. Keep range
predicates on raw `timestamp_ms`; use `to_timestamp_millis` for display/binning.

Imported vLLM identity is `(origin_cluster, service, name,
resource_attributes_json, attributes_json)`, with
`origin_cluster = COALESCE(NULLIF(cluster, ''), 'local')`. Preserve this full
identity through `LAG`; worker/attempt/metric labels must not be collapsed.

Gauges and summary quantiles have `source_temporality = 'current_snapshot'` and
are selected/aggregated directly. Counters, histogram buckets, and summary
`_sum`/`_count` have `cumulative_snapshot`: use `LAG` per full series, discard a
negative delta on reset, and scan one 15-second scrape before the visible window.
Native `rigging.telemetry.counter(...).add(...)` rows are already deltas; sum
them without `LAG`. Histogram bounds are `json_get(attributes_json, 'le')`.

```sql
WITH samples AS (
  SELECT *, LAG(value) OVER (
    PARTITION BY COALESCE(NULLIF(cluster, ''), 'local'), service, name,
      resource_attributes_json, attributes_json ORDER BY timestamp_ms, seq
  ) AS previous_value
  FROM telemetry_v1
  WHERE service = 'vllm' AND name = 'generation_tokens_total'
    AND timestamp_ms >= <start_ms_minus_15000> AND timestamp_ms < <end_ms>
), deltas AS (
  SELECT *, CASE WHEN previous_value IS NULL OR value < previous_value THEN NULL
                 ELSE value - previous_value END AS delta FROM samples
)
SELECT date_bin(INTERVAL '5 minutes', to_timestamp_millis(timestamp_ms)) AS t,
       SUM(delta) AS generated_tokens
FROM deltas WHERE timestamp_ms >= <start_ms> GROUP BY 1 ORDER BY 1;
```

For a serve, list metric names/source kinds/sample counts by `job_id`; sum
lifetime initial values plus nonnegative deltas per reset epoch and then replicas.
Inspect saturation gauges (`num_requests_running`, `num_requests_waiting`,
`kv_cache_usage_perc`) by peak/average. Cross-check token totals, latency
`_sum`/`_count`, TTFT/TPOT, completions, queue depth, and rates; disagreement
usually means series identity was collapsed before delta calculation.
