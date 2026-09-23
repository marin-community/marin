# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded datasets behind the synchronous RL post-training dashboard and its two drill-downs.

The span sources hold one row per step and phase. Finelog reduces each step's worker spans to the
step's critical rank, the rank with the longest ``policy_ppo_train``, and to a per-bucket spread
across ranks.
"""

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
RL_MAX_SPAN_ROWS = 50_000
RL_MAX_COUNTER_ROWS = 50_000
RL_MAX_RESULT_ROWS = 100_000
RL_RECENT_MAX_ROWS = 20
RL_RECENT_WINDOW_PADDING_MS = 60_000
ASYNC_RL_DASHBOARD_UID = "marin-async-rl"
SYNC_RL_DASHBOARD_UID = "marin-rl-runs"

_CORE_NAMES = (
    "phase_duration_seconds",
    "policy_step",
    "ray_object_store_available_memory",
    "ray_object_store_used_memory",
    "ray_spill_manager_objects_bytes",
    "work_completed",
)
_INCLUSIVE_CLOCKS = "('inclusive_wall', 'inclusive_launch')"
_EXCLUSIVE_CLOCKS = "('exclusive_wall', 'exclusive_launch')"
_BARRIER_PHASES = (
    "('policy_entry_barrier', 'policy_final_barrier', 'policy_metric_allreduce', 'policy_entropy_allreduce')"
)
_ROLLOUT_COUNTER_NAMES = ("rollout_wait_seconds", "rollout_count")
# DCGM gauges plotted per bucket, and per-device gauges and cumulative fault counters summarised
# over the window.
_DCGM_SERIES = (
    "gpu_sm_active_ratio",
    "gpu_tensor_active_ratio",
    "gpu_memory_used_bytes",
    "gpu_nvlink_receive_bytes_per_second",
    "gpu_pcie_receive_bytes_per_second",
)
_DCGM_DEVICE = ("gpu_power_watts", "gpu_nvlink_errors", "gpu_pcie_replay_errors")


def _rl_bucket_ms(
    clusters: tuple[str, ...], run: str, start_ms: int, end_ms: int, requested_bucket_ms: int, label: str
) -> int:
    validate_values("clusters", clusters, max_values=RL_MAX_CLUSTERS, max_length=128)
    validate_value("run", run, max_length=512)
    return bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=RL_MAX_WINDOW_MS,
        max_window_error=f"{label} range must not exceed 7 days",
        min_bucket_ms=RL_MIN_BUCKET_MS,
        max_points=RL_MAX_POINTS,
    )


def _bucket_sql(start_ms: int, bucket_ms: int) -> str:
    return f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms}"


def _run_scope(clusters: tuple[str, ...], run: str, start_ms: int, end_ms: int) -> str:
    """The MarinSkyRL rows of one run, in the selected clusters and window."""
    return f"""service = 'marinskyrl'
      AND run_id = {sql_string(run)}
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({sql_values(clusters)})
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}"""


def _run_nodes_cte(bucket: str, clusters_sql: str, start_ms: int, end_ms: int) -> str:
    """The run that wrote most MarinSkyRL rows from each node in each bucket.

    The node agent stamps no run identity, so this is the only way to attribute its rows to a run.
    """
    return f"""run_node AS (
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
)"""


def _phase_rows_cte(bucket: str, scope: str) -> str:
    return f"""phase_rows AS (
    SELECT {bucket} AS t,
           json_get(attributes_json, 'role') AS role,
           json_get(attributes_json, 'step') AS step,
           json_get(attributes_json, 'rank') AS worker_rank,
           json_get(attributes_json, 'phase') AS phase,
           json_get(attributes_json, 'parent') AS parent,
           json_get(attributes_json, 'root') AS root,
           json_get(attributes_json, 'clock_domain') AS clock_domain,
           json_get(attributes_json, 'outcome') AS outcome,
           value
    FROM "telemetry_v1.marinskyrl"
    WHERE {scope}
      AND name = 'phase_duration_seconds'
)"""


# Worker spans tagged with their step's critical rank r*: the rank whose policy_ppo_train ran
# longest, with ties going to the rank id that sorts first. parent_seconds is the step's longest
# policy_ppo_train. covered_seconds sums the exclusive spans each rank published under
# policy_ppo_train, except the producer's own residual.
_CRITICAL_RANK_CTE = f"""tagged AS (
    SELECT t, step, worker_rank, phase, parent, clock_domain, value,
           FIRST_VALUE(worker_rank) OVER (
               PARTITION BY step
               ORDER BY CASE WHEN phase = 'policy_ppo_train' AND clock_domain IN {_INCLUSIVE_CLOCKS}
                             THEN value ELSE -1 END DESC,
                        worker_rank
               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
           ) AS critical_rank,
           MAX(CASE WHEN phase = 'policy_ppo_train' AND clock_domain IN {_INCLUSIVE_CLOCKS} THEN value END)
               OVER (PARTITION BY step) AS parent_seconds,
           SUM(CASE WHEN parent = 'policy_ppo_train' AND clock_domain IN {_EXCLUSIVE_CLOCKS}
                         AND phase <> 'policy_span_residual' THEN value END)
               OVER (PARTITION BY step, worker_rank) AS covered_seconds
    FROM phase_rows
    WHERE role = 'worker'
)"""


def rl_overview_dataset(
    clusters: tuple[str, ...], run: str, start_ms: int, end_ms: int, requested_bucket_ms: int
) -> DashboardDataset:
    """Build bounded RL-core, engine, node-attribution and span sources."""
    bucket_ms = _rl_bucket_ms(clusters, run, start_ms, end_ms, requested_bucket_ms, "RL overview")
    bucket = _bucket_sql(start_ms, bucket_ms)
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
       metric_source, source_temporality, state,
       MAX(weights_step) AS weights_step,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       CAST(NULL AS DOUBLE) AS p50,
       CAST(NULL AS DOUBLE) AS p99
    FROM selected
    GROUP BY 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
), phase_percentiles AS (
    SELECT 'percentile' AS statistic,
       t, name,
       CAST(NULL AS VARCHAR) AS execution_uid,
       CAST(NULL AS VARCHAR) AS work_kind,
       phase,
       CAST(NULL AS VARCHAR) AS outcome,
       clock_domain,
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
)
SELECT * FROM aggregates
UNION ALL SELECT * FROM phase_percentiles
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
WITH {_run_nodes_cte(bucket, clusters_sql, start_ms, end_ms)}, gpu AS (
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
    scope = _run_scope(clusters, run, start_ms, end_ms)
    spans_sql = f"""
WITH {_phase_rows_cte(bucket, scope)}, {_CRITICAL_RANK_CTE}, terminal AS (
    SELECT json_get(attributes_json, 'role') AS role,
           json_get(body_json, 'status') AS status,
           json_get(body_json, 'reason') AS reason,
           MAX(CAST(json_get(body_json, 'export_lost_records') AS BIGINT)) AS lost_records,
           MAX(CAST(json_get(body_json, 'export_queued_records') AS BIGINT)) AS queued_records
    FROM "telemetry_v1.marinskyrl"
    WHERE {scope}
      AND name = 'terminal'
    GROUP BY 1, 2, 3
)
SELECT 'driver' AS statistic, t, step,
       CAST(NULL AS VARCHAR) AS role,
       phase, parent, clock_domain,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       CAST(NULL AS BIGINT) AS ranks,
       CAST(NULL AS BIGINT) AS steps,
       CAST(NULL AS BIGINT) AS failed_steps,
       CAST(NULL AS VARCHAR) AS status,
       CAST(NULL AS VARCHAR) AS reason,
       CAST(NULL AS BIGINT) AS lost_records,
       CAST(NULL AS BIGINT) AS queued_records
FROM phase_rows
WHERE role = 'trainer'
  AND ((clock_domain = 'inclusive_wall' AND root = 'step') OR phase = 'generate_span_residual')
GROUP BY t, step, phase, parent, clock_domain
UNION ALL
SELECT 'critical_rank' AS statistic, t, step,
       CAST(NULL AS VARCHAR) AS role,
       phase, parent, clock_domain,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       CAST(NULL AS BIGINT) AS ranks,
       CAST(NULL AS BIGINT) AS steps,
       CAST(NULL AS BIGINT) AS failed_steps,
       CAST(NULL AS VARCHAR) AS status,
       CAST(NULL AS VARCHAR) AS reason,
       CAST(NULL AS BIGINT) AS lost_records,
       CAST(NULL AS BIGINT) AS queued_records
FROM tagged
WHERE worker_rank = critical_rank AND phase = 'policy_span_residual'
GROUP BY t, step, phase, parent, clock_domain
UNION ALL
SELECT 'coverage' AS statistic,
       CAST(NULL AS BIGINT) AS t,
       CAST(NULL AS VARCHAR) AS step,
       role,
       CAST(NULL AS VARCHAR) AS phase,
       CAST(NULL AS VARCHAR) AS parent,
       clock_domain,
       CAST(NULL AS DOUBLE) AS sum_value,
       CAST(NULL AS BIGINT) AS sample_count,
       CAST(NULL AS DOUBLE) AS max_value,
       NULLIF(COUNT(DISTINCT worker_rank), 0) AS ranks,
       COUNT(DISTINCT step) AS steps,
       CASE WHEN COUNT(outcome) = 0 THEN NULL
            ELSE COUNT(DISTINCT CASE WHEN outcome = 'failure' THEN step END) END AS failed_steps,
       CAST(NULL AS VARCHAR) AS status,
       CAST(NULL AS VARCHAR) AS reason,
       CAST(NULL AS BIGINT) AS lost_records,
       CAST(NULL AS BIGINT) AS queued_records
FROM phase_rows
GROUP BY role, clock_domain
UNION ALL
SELECT 'terminal' AS statistic,
       CAST(NULL AS BIGINT) AS t,
       CAST(NULL AS VARCHAR) AS step,
       role,
       CAST(NULL AS VARCHAR) AS phase,
       CAST(NULL AS VARCHAR) AS parent,
       CAST(NULL AS VARCHAR) AS clock_domain,
       CAST(NULL AS DOUBLE) AS sum_value,
       CAST(NULL AS BIGINT) AS sample_count,
       CAST(NULL AS DOUBLE) AS max_value,
       CAST(NULL AS BIGINT) AS ranks,
       CAST(NULL AS BIGINT) AS steps,
       CAST(NULL AS BIGINT) AS failed_steps,
       status, reason, lost_records, queued_records
FROM terminal
ORDER BY statistic, t, step
LIMIT {RL_MAX_SPAN_ROWS + 1}
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
        "span_coverage": (
            "SELECT role, clock_domain AS clock, ranks, steps, failed_steps "
            "FROM spans WHERE statistic = 'coverage' ORDER BY 1, 2"
        ),
        "run_outcome": (
            "SELECT role, status, reason, lost_records, queued_records "
            "FROM spans WHERE statistic = 'terminal' ORDER BY 1, 2"
        ),
        # A phase's band is its wall minus its children's walls in the same step, so the bands
        # close on the step at any depth; the step's own band is what no phase accounts for.
        "step_composition": (
            """
WITH driver AS (
    SELECT * FROM spans WHERE statistic = 'driver' AND clock_domain = 'inclusive_wall'
), contained AS (
    SELECT t, step, parent AS phase, SUM(sum_value) AS child_seconds
    FROM driver WHERE parent IS NOT NULL AND parent <> '' GROUP BY 1, 2, 3
)
SELECT driver.t,
       driver.phase AS series,
       SUM(driver.sum_value - driver.sample_count * COALESCE(contained.child_seconds, 0))
           / SUM(driver.sample_count) AS value
FROM driver
LEFT JOIN contained
  ON contained.t = driver.t AND contained.step = driver.step AND contained.phase = driver.phase
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "policy_train_share": (
            """
WITH per_step AS (
    SELECT t, step,
           MAX(CASE WHEN phase = 'policy_train' THEN max_value END) AS policy_train,
           MAX(CASE WHEN phase = 'step' THEN max_value END) AS step_seconds
    FROM spans WHERE statistic = 'driver' AND clock_domain = 'inclusive_wall'
    GROUP BY 1, 2
)
SELECT t, AVG(policy_train / NULLIF(step_seconds, 0)) AS policy_train_share
FROM per_step GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "generate_residual": (
            "SELECT t, SUM(sum_value) / SUM(sample_count) AS generate_span_residual FROM spans "
            "WHERE statistic = 'driver' AND phase = 'generate_span_residual' GROUP BY 1 ORDER BY 1"
        ),
        "policy_residual": (
            "SELECT t, SUM(sum_value) / SUM(sample_count) AS policy_span_residual FROM spans "
            "WHERE statistic = 'critical_rank' AND phase = 'policy_span_residual' GROUP BY 1 ORDER BY 1"
        ),
    }
    return DashboardDataset(
        name="RL overview",
        cache_key=(clusters, run, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("core", core_sql, RL_MAX_CORE_ROWS),
            SourceQuery("engine", engine_sql, RL_MAX_ENGINE_ROWS),
            SourceQuery("gpu", gpu_sql, RL_MAX_GPU_ROWS),
            SourceQuery("spans", spans_sql, RL_MAX_SPAN_ROWS),
        ),
        setup_sql=(),
        views=views,
        max_result_rows=RL_MAX_RESULT_ROWS,
    )


def rl_sync_generation_dataset(
    clusters: tuple[str, ...], run: str, start_ms: int, end_ms: int, requested_bucket_ms: int
) -> DashboardDataset:
    """Build the driver's step spans and rollout counters, one row per step and phase or counter."""
    bucket_ms = _rl_bucket_ms(clusters, run, start_ms, end_ms, requested_bucket_ms, "RL generation")
    names = sql_values(("phase_duration_seconds", *_ROLLOUT_COUNTER_NAMES))
    driver_sql = f"""
WITH selected AS (
    SELECT {_bucket_sql(start_ms, bucket_ms)} AS t,
           name,
           json_get(attributes_json, 'step') AS step,
           json_get(attributes_json, 'phase') AS phase,
           json_get(attributes_json, 'parent') AS parent,
           json_get(attributes_json, 'root') AS root,
           json_get(attributes_json, 'clock_domain') AS clock_domain,
           json_get(attributes_json, 'counter') AS counter,
           value
    FROM "telemetry_v1.marinskyrl"
    WHERE {_run_scope(clusters, run, start_ms, end_ms)}
      AND name IN ({names})
      AND json_get(attributes_json, 'role') = 'trainer'
)
SELECT t, step, name, phase, parent, counter,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value
FROM selected
WHERE name <> 'phase_duration_seconds' OR (clock_domain = 'inclusive_wall' AND root = 'step')
GROUP BY t, step, name, phase, parent, counter
ORDER BY t, step, name
LIMIT {RL_MAX_SPAN_ROWS + 1}
""".strip()
    # The rollout counters are sums over concurrent coroutines and exceed the step itself, so
    # every view divides them before plotting.
    setup_sql = (
        f"""
CREATE VIEW rollout_steps AS
SELECT t, step,
       MAX(CASE WHEN counter = 'rollout_trajectory_count' THEN max_value END) AS trajectories,
       MAX(CASE WHEN counter = 'rollout_engine_await_seconds_sum' THEN max_value END) AS engine_seconds,
       MAX(CASE WHEN counter = 'rollout_engine_await_seconds_max' THEN max_value END) AS slowest,
       MAX(CASE WHEN counter = 'rollout_env_await_seconds_sum' THEN max_value END) AS env_seconds,
       MAX(CASE WHEN counter = 'rollout_env_queue_seconds_sum' THEN max_value END) AS queued,
       MAX(CASE WHEN counter = 'rollout_env_exec_seconds_sum' THEN max_value END) AS executed,
       MAX(CASE WHEN counter = 'rollout_env_resume_seconds_sum' THEN max_value END) AS resumed
FROM driver WHERE name IN ({sql_values(_ROLLOUT_COUNTER_NAMES)})
GROUP BY 1, 2
""".strip(),
    )
    views = {
        "generation_vs_training": (
            "SELECT t, phase AS series, SUM(sum_value) / SUM(sample_count) AS value FROM driver "
            "WHERE name = 'phase_duration_seconds' AND phase IN ('generate', 'policy_train') "
            "GROUP BY 1, 2 ORDER BY 1"
        ),
        "generate_breakdown": (
            """
WITH spans AS (
    SELECT * FROM driver WHERE name = 'phase_duration_seconds'
), per_step AS (
    SELECT t, step,
           MAX(CASE WHEN phase = 'generate' THEN max_value END) AS generate_seconds,
           SUM(CASE WHEN parent = 'generate' THEN sum_value END) AS child_seconds
    FROM spans GROUP BY 1, 2
), banded AS (
    SELECT spans.t, spans.phase AS band,
           spans.sum_value / NULLIF(per_step.generate_seconds, 0) AS share_sum,
           spans.sample_count AS samples
    FROM spans JOIN per_step ON per_step.t = spans.t AND per_step.step = spans.step
    WHERE spans.parent = 'generate'
    UNION ALL
    SELECT t, 'generate_span_residual', (generate_seconds - child_seconds) / NULLIF(generate_seconds, 0), 1
    FROM per_step WHERE child_seconds IS NOT NULL
)
SELECT t, band AS series,
       SUM(share_sum) / SUM(CASE WHEN share_sum IS NOT NULL THEN samples END) AS value
FROM banded GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "trajectory_wait": (
            """
SELECT t,
       AVG(engine_seconds / NULLIF(trajectories, 0)) AS engine_wait_per_trajectory,
       AVG(env_seconds / NULLIF(trajectories, 0)) AS env_wait_per_trajectory,
       AVG(slowest) AS engine_await_max
FROM rollout_steps GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "tail_over_mean": (
            "SELECT t, AVG(slowest / NULLIF(engine_seconds / NULLIF(trajectories, 0), 0)) AS tail_over_mean "
            "FROM rollout_steps GROUP BY 1 ORDER BY 1"
        ),
        "environment_split": (
            """
WITH banded AS (
    SELECT t, 'rollout_env_queue' AS band, queued / NULLIF(env_seconds, 0) AS share FROM rollout_steps
    UNION ALL
    SELECT t, 'rollout_env_exec', executed / NULLIF(env_seconds, 0) FROM rollout_steps
    UNION ALL
    SELECT t, 'rollout_env_resume', resumed / NULLIF(env_seconds, 0) FROM rollout_steps
    UNION ALL
    SELECT t, 'remainder', (env_seconds - queued - executed - resumed) / NULLIF(env_seconds, 0)
    FROM rollout_steps
)
SELECT t, band AS series, AVG(share) AS value FROM banded GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
    }
    return DashboardDataset(
        name="RL generation",
        cache_key=(clusters, run, start_ms, end_ms, bucket_ms),
        sources=(SourceQuery("driver", driver_sql, RL_MAX_SPAN_ROWS),),
        setup_sql=setup_sql,
        views=views,
        max_result_rows=RL_MAX_RESULT_ROWS,
    )


def rl_sync_train_step_dataset(
    clusters: tuple[str, ...], run: str, start_ms: int, end_ms: int, requested_bucket_ms: int
) -> DashboardDataset:
    """Build critical-rank worker spans, per-step worker counters, and DCGM on the run's nodes."""
    bucket_ms = _rl_bucket_ms(clusters, run, start_ms, end_ms, requested_bucket_ms, "RL train step")
    bucket = _bucket_sql(start_ms, bucket_ms)
    scope = _run_scope(clusters, run, start_ms, end_ms)
    clusters_sql = sql_values(clusters)
    spans_sql = f"""
WITH {_phase_rows_cte(bucket, scope)}, {_CRITICAL_RANK_CTE}
SELECT 'critical_rank' AS statistic, t, step, phase, parent, clock_domain,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       CAST(NULL AS DOUBLE) AS min_value,
       CAST(NULL AS DOUBLE) AS p50,
       CAST(NULL AS DOUBLE) AS p95,
       MAX(parent_seconds) AS parent_seconds,
       MAX(covered_seconds) AS covered_seconds
FROM tagged
WHERE worker_rank = critical_rank
GROUP BY t, step, phase, parent, clock_domain
UNION ALL
SELECT 'rank_spread' AS statistic, t,
       CAST(NULL AS VARCHAR) AS step,
       'policy_ppo_train' AS phase,
       CAST(NULL AS VARCHAR) AS parent,
       CAST(NULL AS VARCHAR) AS clock_domain,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       MIN(value) AS min_value,
       approx_percentile_cont(value, 0.50) AS p50,
       approx_percentile_cont(value, 0.95) AS p95,
       CAST(NULL AS DOUBLE) AS parent_seconds,
       CAST(NULL AS DOUBLE) AS covered_seconds
FROM phase_rows
WHERE role = 'worker' AND phase = 'policy_ppo_train' AND clock_domain IN {_INCLUSIVE_CLOCKS}
GROUP BY t
UNION ALL
SELECT 'driver' AS statistic, t, step, phase,
       CAST(NULL AS VARCHAR) AS parent,
       clock_domain,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value,
       CAST(NULL AS DOUBLE) AS min_value,
       CAST(NULL AS DOUBLE) AS p50,
       CAST(NULL AS DOUBLE) AS p95,
       CAST(NULL AS DOUBLE) AS parent_seconds,
       CAST(NULL AS DOUBLE) AS covered_seconds
FROM phase_rows
WHERE role = 'trainer' AND clock_domain = 'inclusive_wall' AND phase = 'policy_train'
GROUP BY t, step, phase, clock_domain
ORDER BY statistic, t, step
LIMIT {RL_MAX_SPAN_ROWS + 1}
""".strip()
    # Each counter is first reduced to its per-rank maximum within a step, as the worker publishes
    # one value per rank. Padding averages the per-rank fractions; the worker branch keeps the
    # rank sum and rank maximum of every counter.
    counters_sql = f"""
WITH samples AS (
    SELECT {bucket} AS t,
           json_get(attributes_json, 'step') AS step,
           json_get(attributes_json, 'rank') AS worker_rank,
           json_get(attributes_json, 'role') AS role,
           name,
           json_get(attributes_json, 'counter') AS counter,
           MAX(value) AS value
    FROM "telemetry_v1.marinskyrl"
    WHERE {scope}
      AND name IN ('policy_train_count', 'policy_train_bytes')
    GROUP BY 1, 2, 3, 4, 5, 6
), per_rank AS (
    SELECT t, step, worker_rank,
           MAX(CASE WHEN counter IN ('rank_tokens_real', 'tokens_real') THEN value END) AS tokens_real,
           MAX(CASE WHEN counter IN ('rank_tokens_padded', 'tokens_padded') THEN value END) AS tokens_padded,
           MAX(CASE WHEN counter = 'attention_work_ratio' THEN value END) AS attention_work,
           MAX(CASE WHEN counter = 'micro_step_count' THEN value END) AS micro_steps
    FROM samples WHERE name = 'policy_train_count'
    GROUP BY 1, 2, 3
), worker AS (
    SELECT t, step, worker_rank, counter, MAX(value) AS value
    FROM samples WHERE role = 'worker'
    GROUP BY 1, 2, 3, 4
)
SELECT 'per_rank' AS statistic, t, step, 'padded_fraction' AS counter,
       SUM(1.0 - tokens_real / NULLIF(tokens_padded, 0)) AS sum_value,
       COUNT(1.0 - tokens_real / NULLIF(tokens_padded, 0)) AS sample_count,
       CAST(NULL AS DOUBLE) AS max_value
FROM per_rank GROUP BY t, step
UNION ALL
SELECT 'per_rank' AS statistic, t, step, 'attention_work_ratio' AS counter,
       SUM(attention_work) AS sum_value,
       COUNT(attention_work) AS sample_count,
       CAST(NULL AS DOUBLE) AS max_value
FROM per_rank GROUP BY t, step
UNION ALL
SELECT 'per_rank' AS statistic, t, step, 'micro_step_count' AS counter,
       CAST(NULL AS DOUBLE) AS sum_value,
       CAST(NULL AS BIGINT) AS sample_count,
       MAX(micro_steps) AS max_value
FROM per_rank GROUP BY t, step
UNION ALL
SELECT 'worker' AS statistic, t, step, counter,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MAX(value) AS max_value
FROM worker GROUP BY t, step, counter
ORDER BY statistic, t, step, counter
LIMIT {RL_MAX_COUNTER_ROWS + 1}
""".strip()
    gpu_sql = f"""
WITH {_run_nodes_cte(bucket, clusters_sql, start_ms, end_ms)}, gpu AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           {bucket} AS t,
           node_name AS node,
           json_get(attributes_json, 'gpu_uuid') AS gpu,
           name,
           AVG(value) AS mean_value,
           MAX(value) AS max_value,
           MIN(value) AS min_value
    FROM "telemetry_v1.node_agent"
    WHERE name IN ({sql_values(_DCGM_SERIES + _DCGM_DEVICE)})
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({clusters_sql})
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
    GROUP BY 1, 2, 3, 4, 5
), attributed AS (
    SELECT gpu.* FROM gpu JOIN run_node USING (origin_cluster, t, node)
    WHERE run_node.run = {sql_string(run)}
)
SELECT CASE WHEN GROUPING(node) = 1 THEN 'series' ELSE 'device' END AS statistic,
       t, name, node, gpu,
       CASE WHEN GROUPING(node) = 1 THEN AVG(mean_value) END AS mean_value,
       CASE WHEN GROUPING(node) = 1 THEN SUM(mean_value) END AS total_value,
       MAX(max_value) AS max_value,
       CASE WHEN GROUPING(node) = 0 THEN MIN(min_value) END AS min_value
FROM attributed
GROUP BY GROUPING SETS ((t, name), (name, node, gpu))
HAVING (GROUPING(node) = 1 AND name IN ({sql_values(_DCGM_SERIES)}))
    OR (GROUPING(node) = 0 AND name IN ({sql_values(_DCGM_DEVICE)}))
ORDER BY statistic, t, name
LIMIT {RL_MAX_GPU_ROWS + 1}
""".strip()
    setup_sql = (
        f"""
CREATE VIEW critical_steps AS
SELECT t, step,
       MAX(CASE WHEN phase = 'policy_forward' THEN max_value END) AS forward_seconds,
       MAX(CASE WHEN phase = 'policy_backward' THEN max_value END) AS backward_seconds,
       SUM(CASE WHEN phase IN {_BARRIER_PHASES} THEN sum_value END) AS waiting_seconds,
       MAX(parent_seconds) AS total_seconds
FROM spans
WHERE statistic = 'critical_rank' AND clock_domain IN {_EXCLUSIVE_CLOCKS}
GROUP BY 1, 2
""".strip(),
    )
    views = {
        # Every band is read from r*'s own rows. A per-phase maximum across ranks takes barrier and
        # compute from different ranks, which are anti-correlated, and sums past the parent.
        "ppo_decomposition": (
            f"""
WITH banded AS (
    SELECT t,
           CASE WHEN clock_domain IN {_INCLUSIVE_CLOCKS} THEN 'policy_span_residual' ELSE phase END AS band,
           CASE WHEN clock_domain IN {_INCLUSIVE_CLOCKS}
                THEN (parent_seconds - covered_seconds) * sample_count ELSE sum_value END AS seconds,
           sample_count
    FROM spans
    WHERE statistic = 'critical_rank'
      AND ((parent = 'policy_ppo_train' AND clock_domain IN {_EXCLUSIVE_CLOCKS}
            AND phase <> 'policy_span_residual')
           OR (clock_domain IN {_INCLUSIVE_CLOCKS} AND phase = 'policy_ppo_train'))
)
SELECT t, band AS series,
       SUM(seconds) / SUM(CASE WHEN seconds IS NOT NULL THEN sample_count END) AS value
FROM banded GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "rank_skew": (
            "SELECT t, max_value AS slowest_rank, p95, p50, min_value AS fastest_rank "
            "FROM spans WHERE statistic = 'rank_spread' ORDER BY 1"
        ),
        "micro_steps": (
            """
WITH driver AS (
    SELECT t, step, max_value AS seconds FROM spans WHERE statistic = 'driver'
), micro AS (
    SELECT step, MAX(max_value) AS micro_steps FROM counters
    WHERE statistic = 'per_rank' AND counter = 'micro_step_count' GROUP BY 1
)
SELECT driver.t,
       AVG(driver.seconds / NULLIF(micro.micro_steps, 0)) AS seconds_per_micro_step,
       AVG(micro.micro_steps) AS micro_steps
FROM driver LEFT JOIN micro ON driver.step = micro.step
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "backward_over_forward": (
            "SELECT t, AVG(backward_seconds / NULLIF(forward_seconds, 0)) AS backward_over_forward "
            "FROM critical_steps GROUP BY 1 ORDER BY 1"
        ),
        # A rank that published no barrier span waited for an unknown time, so the share stays
        # NULL. Zero would read as a measurement that it never waited.
        "waiting_share": (
            "SELECT t, AVG(waiting_seconds / NULLIF(total_seconds, 0)) AS waiting_share "
            "FROM critical_steps GROUP BY 1 ORDER BY 1"
        ),
        "padding": (
            """
SELECT t,
       SUM(CASE WHEN counter = 'padded_fraction' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN counter = 'padded_fraction' THEN sample_count END), 0) AS padded_fraction,
       SUM(CASE WHEN counter = 'attention_work_ratio' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN counter = 'attention_work_ratio' THEN sample_count END), 0)
           AS attention_work_ratio
FROM counters WHERE statistic = 'per_rank' GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "allocator": (
            """
SELECT t,
       MAX(CASE WHEN counter = 'peak_reserved_bytes' THEN max_value END) AS peak_reserved_bytes,
       MAX(CASE WHEN counter = 'peak_allocated_bytes' THEN max_value END) AS peak_allocated_bytes,
       SUM(CASE WHEN counter = 'alloc_retries' THEN sum_value END) AS alloc_retries,
       SUM(CASE WHEN counter = 'alloc_ooms' THEN sum_value END) AS alloc_ooms
FROM counters WHERE statistic = 'worker' GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "sm_activity": (
            "SELECT t, CASE name WHEN 'gpu_sm_active_ratio' THEN 'SM active' ELSE 'tensor pipe active' END AS series, "
            "mean_value * 100.0 AS value FROM gpu WHERE statistic = 'series' "
            "AND name IN ('gpu_sm_active_ratio', 'gpu_tensor_active_ratio') ORDER BY 1"
        ),
        "gpu_memory": (
            "SELECT t, mean_value AS mean_used_bytes, max_value AS peak_used_bytes FROM gpu "
            "WHERE statistic = 'series' AND name = 'gpu_memory_used_bytes' ORDER BY 1"
        ),
        "fabric_receive": (
            "SELECT t, CASE name WHEN 'gpu_nvlink_receive_bytes_per_second' THEN 'NVLink receive' "
            "ELSE 'PCIe receive' END AS series, total_value AS value FROM gpu WHERE statistic = 'series' "
            "AND name IN ('gpu_nvlink_receive_bytes_per_second', 'gpu_pcie_receive_bytes_per_second') ORDER BY 1"
        ),
        # Only a GPU whose fault counters rose appears; an empty table is the healthy result.
        "link_faults": (
            """
SELECT node, gpu,
       MAX(CASE WHEN name = 'gpu_power_watts' THEN max_value END) AS peak_power_watts,
       MAX(CASE WHEN name = 'gpu_nvlink_errors' THEN max_value - min_value END) AS nvlink_error_increase,
       MAX(CASE WHEN name = 'gpu_pcie_replay_errors' THEN max_value - min_value END) AS pcie_replay_increase
FROM gpu WHERE statistic = 'device'
GROUP BY 1, 2
HAVING MAX(CASE WHEN name IN ('gpu_nvlink_errors', 'gpu_pcie_replay_errors')
                THEN max_value - min_value END) > 0
ORDER BY 4 DESC, 5 DESC
""".strip()
        ),
    }
    return DashboardDataset(
        name="RL train step",
        cache_key=(clusters, run, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("spans", spans_sql, RL_MAX_SPAN_ROWS),
            SourceQuery("counters", counters_sql, RL_MAX_COUNTER_ROWS),
            SourceQuery("gpu", gpu_sql, RL_MAX_GPU_ROWS),
        ),
        setup_sql=setup_sql,
        views=views,
        max_result_rows=RL_MAX_RESULT_ROWS,
    )


def recent_rl_runs_dataset(start_ms: int, end_ms: int) -> DashboardDataset:
    """Build a bounded table of recent RL runs, their dashboard link windows, and the dashboard
    whose run picker offers each run: the async view for a run whose trainer stamps
    training_loop 'async', the sync view for every other run."""
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
       MAX(timestamp_ms) AS last_seen,
       MAX(CASE WHEN json_get(resource_attributes_json, 'training_loop') = 'async' THEN 1 ELSE 0 END) AS is_async
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
                f"""
SELECT run, origin_cluster AS cluster, step, attempts,
       window_from_ms, window_to_ms, last_seen AS "last seen",
       CASE WHEN is_async = 1 THEN {sql_string(ASYNC_RL_DASHBOARD_UID)}
            ELSE {sql_string(SYNC_RL_DASHBOARD_UID)} END AS dashboard
FROM recent ORDER BY last_seen DESC
""".strip()
            )
        },
        max_result_rows=RL_RECENT_MAX_ROWS,
    )
