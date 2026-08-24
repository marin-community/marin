# vLLM telemetry in Finelog

Native vLLM `/metrics` snapshots are exported every 15 seconds to `telemetry_v1` with `service = 'vllm'`. The exporter removes the redundant `vllm:` prefix from metric names.

Useful typed columns are `cluster`, `service`, `name`, `kind`, `value`, `unit`, `timestamp_ms`, and `seq`. Process identity remains in `resource_attributes_json`; source labels and snapshot markers remain in `attributes_json`. Read JSON strings with untyped `json_get`, for example:

```sql
json_get(resource_attributes_json, 'job_id')
json_get(attributes_json, 'model_name')
```

`json_get` returns `VARCHAR`. A complete imported series is identified by origin cluster, service, name, `resource_attributes_json`, and `attributes_json`. Preserve that identity through delta calculation.

## Snapshot semantics

The exporter stores source values as gauges and records their original semantics:

- Gauges and summary quantiles have `source_temporality = 'current_snapshot'`; select or aggregate the values directly.
- Counters, histogram bucket/sum/count samples, and summary sum/count samples have `source_temporality = 'cumulative_snapshot'`; compute ordered deltas.
- A negative cumulative delta marks a process reset. Discard that interval for a windowed rate. For a lifetime total, treat the first value of each reset epoch as its initial contribution.
- Native `rigging.telemetry.counter(...).add(...)` rows are already deltas. Sum them directly and never apply `LAG`.

Scan one 15-second scrape interval before the visible window so the first visible sample has a predecessor. Filter to the visible window only after computing deltas.

```sql
WITH base AS (
  SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
         service, name, resource_attributes_json, attributes_json,
         timestamp_ms, seq, value
  FROM telemetry_v1
  WHERE service = 'vllm'
    AND name = 'generation_tokens_total'
    AND json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
    AND timestamp_ms >= <visible_start_ms - 15000>
    AND timestamp_ms < <visible_end_ms>
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
WHERE timestamp_ms >= <visible_start_ms>
GROUP BY 1
ORDER BY 1
```

Histogram buckets are cumulative snapshots. Their upper bound is `json_get(attributes_json, 'le')`; `_count` and `_sum` use the same full series identity. Derive quantiles from bucket deltas or divide sum delta by count delta for a mean.

For serve triage, first list available metric names and `source_kind` for the exact structured `job_id` when present. Queue depth, KV-cache use, TTFT, TPOT, completion labels, token rates, and preemption counts usually distinguish queueing, cache pressure, and compute saturation. Cross-check independent totals; disagreement often means replicas were collapsed before delta calculation.
