# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Three bounded sources shared by the RL post-training dashboard."""

from dashboard_dataset import (
    DashboardDataset,
    SourceQuery,
    bounded_bucket_ms,
    validate_time_window,
    validate_value,
    validate_values,
)
from vllm_observability import sql_string, sql_values

RL_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
RL_MAX_POINTS = 360
RL_MIN_BUCKET_MS = 30_000
RL_MAX_CLUSTERS = 16
RL_MAX_CORE_ROWS = 100_000
RL_MAX_ENGINE_ROWS = 100_000
RL_MAX_GPU_ROWS = 50_000
RL_MAX_RESULT_ROWS = 100_000
RL_RECENT_MAX_ROWS = 20
RL_RECENT_WINDOW_PADDING_MS = 60_000

_CORE_NAMES = (
    "phase_duration_seconds",
    "policy_step",
    "ray_object_store_available_memory",
    "ray_object_store_used_memory",
    "ray_spill_manager_objects_bytes",
    "rollout_capacity",
    "rollout_queue_depth",
    "rollout_staleness_steps",
    "work_completed",
)


def rl_overview_dataset(
    clusters: tuple[str, ...], run: str, start_ms: int, end_ms: int, requested_bucket_ms: int
) -> DashboardDataset:
    """Build bounded RL-core, engine, and node-attribution sources."""
    validate_values("clusters", clusters, max_values=RL_MAX_CLUSTERS, max_length=128)
    validate_value("run", run, max_length=512)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=RL_MAX_WINDOW_MS,
        max_window_error="RL overview range must not exceed 7 days",
        min_bucket_ms=RL_MIN_BUCKET_MS,
        max_points=RL_MAX_POINTS,
    )
    bucket = f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms}"
    clusters_sql = sql_values(clusters)
    run_sql = sql_string(run)
    core_sql = f"""
WITH selected AS (
    SELECT {bucket} AS t,
           name,
           execution_uid,
           json_get(attributes_json, 'work_kind') AS work_kind,
           json_get(attributes_json, 'phase') AS phase,
           json_get(attributes_json, 'outcome') AS outcome,
           json_get(attributes_json, 'clock_domain') AS clock_domain,
           json_get(attributes_json, 'queue') AS queue,
           json_get(attributes_json, 'metric_source') AS metric_source,
           json_get(attributes_json, 'source_temporality') AS source_temporality,
           json_get(attributes_json, 'state') AS state,
           CAST(json_get(attributes_json, 'weights_step') AS DOUBLE) AS weights_step,
           value
    FROM "telemetry_v1.marinskyrl"
    WHERE run_id = {run_sql}
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({clusters_sql})
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
      AND name IN ({sql_values(_CORE_NAMES)})
), aggregates AS (
    SELECT 'aggregate' AS statistic,
       t, name, execution_uid, work_kind, phase, outcome, clock_domain,
       queue, metric_source, source_temporality, state,
       MAX(weights_step) AS weights_step,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       CAST(NULL AS DOUBLE) AS p50,
       CAST(NULL AS DOUBLE) AS p99
    FROM selected
    WHERE name <> 'rollout_staleness_steps'
    GROUP BY 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
), phase_percentiles AS (
    SELECT 'percentile' AS statistic,
       t, name,
       CAST(NULL AS VARCHAR) AS execution_uid,
       CAST(NULL AS VARCHAR) AS work_kind,
       phase,
       CAST(NULL AS VARCHAR) AS outcome,
       clock_domain,
       CAST(NULL AS VARCHAR) AS queue,
       CAST(NULL AS VARCHAR) AS metric_source,
       CAST(NULL AS VARCHAR) AS source_temporality,
       CAST(NULL AS VARCHAR) AS state,
       CAST(NULL AS DOUBLE) AS weights_step,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       approx_percentile_cont(value, 0.5) AS p50,
       approx_percentile_cont(value, 0.99) AS p99
    FROM selected
    WHERE name = 'phase_duration_seconds'
      AND clock_domain = 'critical_path'
      AND phase = 'rollout_or_inference_wait'
    GROUP BY 2, 3, 6, 8
), staleness_percentiles AS (
    SELECT 'percentile' AS statistic,
       t, name,
       CAST(NULL AS VARCHAR) AS execution_uid,
       CAST(NULL AS VARCHAR) AS work_kind,
       CAST(NULL AS VARCHAR) AS phase,
       CAST(NULL AS VARCHAR) AS outcome,
       CAST(NULL AS VARCHAR) AS clock_domain,
       CAST(NULL AS VARCHAR) AS queue,
       CAST(NULL AS VARCHAR) AS metric_source,
       CAST(NULL AS VARCHAR) AS source_temporality,
       CAST(NULL AS VARCHAR) AS state,
       CAST(NULL AS DOUBLE) AS weights_step,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       approx_percentile_cont(value, 0.5) AS p50,
       approx_percentile_cont(value, 0.99) AS p99
    FROM selected
    WHERE name = 'rollout_staleness_steps'
    GROUP BY 2, 3
)
SELECT * FROM aggregates
UNION ALL SELECT * FROM phase_percentiles
UNION ALL SELECT * FROM staleness_percentiles
ORDER BY t, name, execution_uid
LIMIT {RL_MAX_CORE_ROWS + 1}
""".strip()
    engine_sql = f"""
WITH engine_rows AS (
    SELECT * FROM "telemetry_v1.vllm" WHERE service = 'vllm'
    UNION ALL
    SELECT * FROM "telemetry_v1.marinskyrl"
    WHERE service = 'marinskyrl' AND json_get(attributes_json, 'metric_source') = 'vllm'
), selected AS (
    SELECT timestamp_ms, seq, service, run_id, execution_uid, node_name, process_index,
           name, resource_attributes_json, attributes_json, value,
           value - LAG(value) OVER (
               PARTITION BY service, run_id, execution_uid, node_name, process_index,
                            name, resource_attributes_json, attributes_json
               ORDER BY timestamp_ms, seq
           ) AS delta
    FROM engine_rows
    WHERE run_id = {run_sql}
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({clusters_sql})
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
      AND (name IN ('generation_tokens_total', 'prompt_tokens_total', 'num_requests_running',
                    'num_requests_waiting', 'kv_cache_usage_perc', 'prefix_cache_hits_total',
                    'prefix_cache_queries_total', 'num_preemptions_total', 'request_success_total')
           OR name LIKE '%_seconds_sum' OR name LIKE '%_seconds_count')
)
SELECT {bucket} AS t,
       name,
       json_get(attributes_json, 'finished_reason') AS finished_reason,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       SUM(GREATEST(delta, 0)) FILTER (WHERE delta IS NOT NULL) AS delta_sum
FROM selected
GROUP BY 1, 2, 3
ORDER BY t, name, finished_reason
LIMIT {RL_MAX_ENGINE_ROWS + 1}
""".strip()
    gpu_sql = f"""
WITH run_node AS (
    SELECT origin_cluster, t, node, run
    FROM (
        SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
               {bucket} AS t,
               node_name AS node,
               run_id AS run,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(NULLIF(cluster, ''), 'marin'), {bucket}, node_name
                   ORDER BY COUNT(*) DESC, run_id
               ) AS rn
        FROM "telemetry_v1.marinskyrl"
        WHERE service = 'marinskyrl' AND node_name <> ''
          AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({clusters_sql})
          AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
        GROUP BY 1, 2, 3, 4
    ) WHERE rn = 1
), gpu AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           {bucket} AS t,
           node_name AS node,
           json_get(attributes_json, 'gpu_uuid') AS gpu,
           AVG(value) AS utilization
    FROM "telemetry_v1.node_agent"
    WHERE name = 'gpu_utilization_percent'
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({clusters_sql})
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
    GROUP BY 1, 2, 3, 4
)
SELECT gpu.t, run_node.run AS series, AVG(gpu.utilization) AS value
FROM gpu JOIN run_node USING (origin_cluster, t, node)
WHERE run_node.run = {run_sql}
GROUP BY 1, 2 ORDER BY 1
LIMIT {RL_MAX_GPU_ROWS + 1}
""".strip()
    views = {
        "policy_step": (
            """
SELECT t, 'trainer step · ' || execution_uid AS series, MAX(max_value) AS value
FROM core WHERE name = 'policy_step' GROUP BY 1, 2
UNION ALL
SELECT t, 'producing policy · ' || execution_uid AS series, MAX(weights_step) AS value
FROM core WHERE name = 'work_completed' AND weights_step IS NOT NULL GROUP BY 1, 2
ORDER BY 1
""".strip()
        ),
        "rollout_progress": (
            """
SELECT t,
       SUM(CASE WHEN work_kind = 'rollout' THEN sum_value END) AS rollouts,
       SUM(CASE WHEN work_kind = 'sample' THEN sum_value END) AS samples,
       SUM(CASE WHEN work_kind = 'generated_token' THEN sum_value END) AS generated_tokens
FROM core WHERE name = 'work_completed' GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "critical_path": (
            """
SELECT t, phase || ' · ' || outcome AS series,
       SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value
FROM core
WHERE statistic = 'aggregate'
  AND name = 'phase_duration_seconds' AND clock_domain = 'critical_path'
  AND phase IN ('rollout_or_inference_wait', 'train_step')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "buffer": (
            """
SELECT t,
       SUM(CASE WHEN name = 'rollout_queue_depth' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'rollout_queue_depth' THEN sample_count END), 0) AS depth,
       SUM(CASE WHEN name = 'rollout_capacity' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'rollout_capacity' THEN sample_count END), 0) AS capacity
FROM core WHERE queue = 'rollout_buffer' GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "gpu_utilization": "SELECT * FROM gpu ORDER BY t",
        "engine_tokens": (
            f"""
SELECT t,
       SUM(CASE WHEN name = 'generation_tokens_total' THEN delta_sum END) / ({bucket_ms} / 1000.0) AS generation,
       SUM(CASE WHEN name = 'prompt_tokens_total' THEN delta_sum END) / ({bucket_ms} / 1000.0) AS prompt
FROM engine WHERE delta_sum IS NOT NULL GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "engine_queue": (
            """
SELECT t,
       SUM(CASE WHEN name = 'num_requests_running' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'num_requests_running' THEN sample_count END), 0) AS running,
       SUM(CASE WHEN name = 'num_requests_waiting' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'num_requests_waiting' THEN sample_count END), 0) AS waiting,
       SUM(CASE WHEN name = 'kv_cache_usage_perc' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'kv_cache_usage_perc' THEN sample_count END), 0) AS kv_cache
FROM engine GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "engine_latency": (
            """
SELECT t, replace(replace(name, '_sum', ''), '_count', '') AS series,
       SUM(CASE WHEN name LIKE '%_sum' THEN delta_sum END)
           / NULLIF(SUM(CASE WHEN name LIKE '%_count' THEN delta_sum END), 0) AS value
FROM engine WHERE delta_sum IS NOT NULL AND (name LIKE '%_seconds_sum' OR name LIKE '%_seconds_count')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "engine_prefix_cache": (
            """
SELECT t, 'hit rate' AS series,
       SUM(CASE WHEN name = 'prefix_cache_hits_total' THEN delta_sum END)
           / NULLIF(SUM(CASE WHEN name = 'prefix_cache_queries_total' THEN delta_sum END), 0) AS value
FROM engine WHERE delta_sum IS NOT NULL GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "engine_preemptions": (
            "SELECT t, 'preemptions' AS series, SUM(delta_sum) AS value FROM engine "
            "WHERE name = 'num_preemptions_total' AND delta_sum IS NOT NULL GROUP BY 1, 2 ORDER BY 1"
        ),
        "straggler": (
            "SELECT t, 'p99 / p50' AS series, p99 / NULLIF(p50, 0) AS value FROM core "
            "WHERE statistic = 'percentile' AND name = 'phase_duration_seconds' "
            "AND clock_domain = 'critical_path' AND phase = 'rollout_or_inference_wait' ORDER BY 1"
        ),
        "staleness": (
            """
SELECT t, 'p50' AS series, p50 AS value FROM core
WHERE statistic = 'percentile' AND name = 'rollout_staleness_steps'
UNION ALL
SELECT t, 'p99' AS series, p99 AS value FROM core
WHERE statistic = 'percentile' AND name = 'rollout_staleness_steps'
ORDER BY 1
""".strip()
        ),
        "engine_finish": (
            "SELECT t, finished_reason AS series, SUM(delta_sum) AS value FROM engine "
            "WHERE name = 'request_success_total' AND finished_reason IS NOT NULL "
            "AND delta_sum IS NOT NULL GROUP BY 1, 2 ORDER BY 1"
        ),
        "ray_object_store": (
            """
SELECT t,
       SUM(CASE WHEN name = 'ray_object_store_used_memory' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name IN ('ray_object_store_used_memory',
                                           'ray_object_store_available_memory')
                             THEN sum_value END), 0)
           AS object_store_used_fraction
FROM core WHERE metric_source = 'ray' AND source_temporality = 'current_snapshot'
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "ray_spill": (
            """
SELECT t, COALESCE(state, 'unlabelled') AS series,
       SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value
FROM core
WHERE name = 'ray_spill_manager_objects_bytes' AND metric_source = 'ray'
  AND source_temporality = 'current_snapshot'
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
    }
    return DashboardDataset(
        name="RL overview",
        cache_key=(clusters, run, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("core", core_sql, RL_MAX_CORE_ROWS),
            SourceQuery("engine", engine_sql, RL_MAX_ENGINE_ROWS),
            SourceQuery("gpu", gpu_sql, RL_MAX_GPU_ROWS),
        ),
        setup_sql=(),
        views=views,
        max_result_rows=RL_MAX_RESULT_ROWS,
    )


def recent_rl_runs_dataset(start_ms: int, end_ms: int) -> DashboardDataset:
    """Build a bounded table of recent RL runs and their dashboard link windows."""
    validate_time_window(
        start_ms,
        end_ms,
        max_window_ms=RL_MAX_WINDOW_MS,
        max_window_error="recent RL run range must not exceed 7 days",
    )
    sql = f"""
SELECT run_id AS run,
       COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
       MAX(value) AS step,
       COUNT(DISTINCT execution_uid) AS attempts,
       MIN(timestamp_ms) - {RL_RECENT_WINDOW_PADDING_MS} AS window_from_ms,
       MAX(timestamp_ms) + {RL_RECENT_WINDOW_PADDING_MS} AS window_to_ms,
       MAX(timestamp_ms) AS last_seen
FROM "telemetry_v1.marinskyrl"
WHERE service = 'marinskyrl' AND name = 'policy_step' AND run_id IS NOT NULL
  AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
GROUP BY 1, 2 ORDER BY last_seen DESC
LIMIT {RL_RECENT_MAX_ROWS}
""".strip()
    return DashboardDataset(
        name="Recent RL runs",
        cache_key=(start_ms, end_ms),
        sources=(SourceQuery("recent", sql, RL_RECENT_MAX_ROWS),),
        setup_sql=(),
        views={
            "recent": (
                """
SELECT run, origin_cluster AS cluster, step, attempts,
       window_from_ms, window_to_ms, last_seen AS "last seen"
FROM recent ORDER BY last_seen DESC
""".strip()
            )
        },
        max_result_rows=RL_RECENT_MAX_ROWS,
    )
