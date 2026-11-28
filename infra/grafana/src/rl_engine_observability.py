# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact per-series counter growth for the RL run-wide engine panels."""


def rl_engine_query(bucket: str, clusters_sql: str, run_sql: str, start_ms: int, end_ms: int, max_rows: int) -> str:
    """Return SQL for engine sums, counts and clamped consecutive-sample growth."""
    origin_cluster = "COALESCE(NULLIF(cluster, ''), 'marin')"
    return f"""
WITH engine_rows AS (
    SELECT * FROM "telemetry_v1.vllm" WHERE service = 'vllm'
    UNION ALL
    SELECT * FROM "telemetry_v1.marinskyrl"
    WHERE service = 'marinskyrl' AND json_get(attributes_json, 'metric_source') = 'vllm'
), filtered AS (
    -- Filter metric names before JSON extraction and retain narrow hashed label
    -- keys for the window, while keeping each actor's series independent.
    SELECT {origin_cluster} AS origin_cluster,
           service, execution_uid, node_name, process_index, name, timestamp_ms, seq, value,
           md5(resource_attributes_json) AS resource_key,
           md5(attributes_json) AS attribute_key,
           json_get(attributes_json, 'finished_reason') AS finished_reason
    FROM engine_rows
    WHERE run_id = {run_sql}
      AND {origin_cluster} IN ({clusters_sql})
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
      AND (name IN ('generation_tokens_total', 'prompt_tokens_total', 'num_requests_running',
                    'num_requests_waiting', 'kv_cache_usage_perc', 'prefix_cache_hits_total',
                    'prefix_cache_queries_total', 'num_preemptions_total', 'request_success_total')
           OR name LIKE '%_seconds_sum' OR name LIKE '%_seconds_count')
), deltas AS (
    SELECT *, value - LAG(value) OVER (
               PARTITION BY origin_cluster, service, execution_uid, node_name, process_index,
                            name, resource_key, attribute_key
               ORDER BY timestamp_ms, seq
           ) AS delta
    FROM filtered
)
SELECT {bucket} AS t, name,
       finished_reason,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       SUM(CASE WHEN delta IS NULL THEN NULL ELSE GREATEST(delta, 0) END) AS delta_sum
FROM deltas
GROUP BY 1, 2, 3
ORDER BY t, name, finished_reason
LIMIT {max_rows + 1}
""".strip()
