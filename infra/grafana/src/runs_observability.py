# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two bounded sources shared by the multi-run training dashboard."""

from dashboard_dataset import DashboardDataset, SourceQuery, bounded_bucket_ms, validate_values
from vllm_observability import sql_values

RUNS_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
RUNS_MAX_POINTS = 360
RUNS_MIN_BUCKET_MS = 15_000
RUNS_MAX_RUNS = 128
RUNS_MAX_CLUSTERS = 16
RUNS_MAX_METRIC_ROWS = 100_000
RUNS_MAX_POWER_ROWS = 50_000
RUNS_MAX_RESULT_ROWS = 50_000
RUNS_IDENTITY_MAX_LENGTH = 512

_METRIC_NAMES = (
    "eval_dropless_loss",
    "eval_dropless_macro_loss",
    "eval_dropless_paloma_macro_loss",
    "eval_loss",
    "eval_macro_loss",
    "eval_paloma_macro_loss",
    "throughput_duration",
    "throughput_hook_time",
    "throughput_loading_time",
    "throughput_mfu",
    "throughput_tokens_per_second",
    "train_loss",
)


def runs_overview_dataset(
    clusters: tuple[str, ...],
    runs: tuple[str, ...],
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
) -> DashboardDataset:
    """Build bounded multi-run metric and GPU-attribution sources."""
    validate_values("clusters", clusters, max_values=RUNS_MAX_CLUSTERS, max_length=RUNS_IDENTITY_MAX_LENGTH)
    validate_values("runs", runs, max_values=RUNS_MAX_RUNS, max_length=RUNS_IDENTITY_MAX_LENGTH)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=RUNS_MAX_WINDOW_MS,
        max_window_error="Runs overview range must not exceed 7 days",
        min_bucket_ms=RUNS_MIN_BUCKET_MS,
        max_points=RUNS_MAX_POINTS,
    )
    bucket = f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms}"
    metrics_bucket = (
        f"{start_ms} + (metrics.timestamp_ms - {start_ms}) - (metrics.timestamp_ms - {start_ms}) % {bucket_ms}"
    )
    cluster_values = sql_values(clusters)
    run_values = sql_values(runs)
    metric_names = sql_values(_METRIC_NAMES)
    metrics_sql = f"""
WITH filtered AS (
    SELECT {bucket} AS t,
           run_id AS run,
           COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           name, value, step, node_name, process_index, timestamp_ms
    FROM "levanter.metrics"
    WHERE run_id IN ({run_values})
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND timestamp_ms >= {start_ms}
      AND timestamp_ms < {end_ms}
      AND name IN ({metric_names})
), active_cardinality AS (
    SELECT run, origin_cluster,
           COUNT(DISTINCT node_name) AS nodes,
           COUNT(DISTINCT process_index) AS processes
    FROM filtered
    WHERE name IN ('throughput_mfu', 'throughput_tokens_per_second', 'train_loss')
    GROUP BY 1, 2
), bucketed AS (
    SELECT t, run, origin_cluster, name,
       SUM(value) AS sum_value,
       COUNT(value) AS sample_count,
       MIN(value) AS min_value,
       MAX(value) AS max_value,
       MAX(step) AS step,
       MAX(timestamp_ms) AS last_ms
    FROM filtered
    GROUP BY 1, 2, 3, 4
)
SELECT bucketed.*, active_cardinality.nodes, active_cardinality.processes
FROM bucketed LEFT JOIN active_cardinality USING (run, origin_cluster)
ORDER BY t, run, origin_cluster, name
LIMIT {RUNS_MAX_METRIC_ROWS + 1}
""".strip()
    power_sql = f"""
WITH selected_nodes AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           {bucket} AS t,
           node_name AS node
    FROM "levanter.metrics"
    WHERE step IS NOT NULL
      AND run_id IN ({run_values})
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND timestamp_ms >= {start_ms}
      AND timestamp_ms < {end_ms}
    GROUP BY 1, 2, 3
), run_raw AS (
    SELECT COALESCE(NULLIF(metrics.cluster, ''), 'marin') AS origin_cluster,
           {metrics_bucket} AS t,
           metrics.node_name AS node,
           metrics.run_id AS run
    FROM "levanter.metrics" AS metrics
    JOIN selected_nodes
      ON selected_nodes.origin_cluster = COALESCE(NULLIF(metrics.cluster, ''), 'marin')
     AND selected_nodes.t = {metrics_bucket}
     AND selected_nodes.node = metrics.node_name
    WHERE metrics.step IS NOT NULL
      AND COALESCE(NULLIF(metrics.cluster, ''), 'marin') IN ({cluster_values})
      AND metrics.timestamp_ms >= {start_ms}
      AND metrics.timestamp_ms < {end_ms}
), run_node AS (
    SELECT origin_cluster, t, node, run
    FROM (
        SELECT origin_cluster, t, node, run,
               ROW_NUMBER() OVER (PARTITION BY origin_cluster, t, node ORDER BY COUNT(*) DESC, run) AS rn
        FROM run_raw GROUP BY 1, 2, 3, 4
    ) WHERE rn = 1
), gpu AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           {bucket} AS t,
           node_name AS node,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           AVG(value) AS watts
    FROM "telemetry_v1.node_agent"
    WHERE service = 'iris-node-agent'
      AND name = 'gpu_power_watts'
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND timestamp_ms >= {start_ms}
      AND timestamp_ms < {end_ms}
    GROUP BY 1, 2, 3, 4
)
SELECT gpu.t, run_node.run, SUM(gpu.watts) / 1000.0 AS value
FROM gpu JOIN run_node USING (origin_cluster, t, node)
WHERE run_node.run IN ({run_values})
GROUP BY 1, 2 ORDER BY 1, 2
LIMIT {RUNS_MAX_POWER_ROWS + 1}
""".strip()
    average = "SUM(sum_value) / NULLIF(SUM(sample_count), 0)"
    views = {
        "active": (
            f"""
SELECT run, origin_cluster AS cluster,
       MAX(nodes) AS nodes, MAX(processes) AS processes, MAX(step) AS step,
       SUM(CASE WHEN name = 'throughput_mfu' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_mfu' THEN sample_count END), 0) AS mfu,
       SUM(CASE WHEN name = 'throughput_tokens_per_second' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_tokens_per_second' THEN sample_count END), 0)
           AS tokens_per_second,
       MIN(CASE WHEN name = 'train_loss' THEN min_value END) AS best_loss,
       ({end_ms} - MAX(last_ms)) / 1000.0 AS sample_age_seconds
FROM metrics
WHERE name IN ('throughput_mfu', 'throughput_tokens_per_second', 'train_loss')
GROUP BY 1, 2 ORDER BY step DESC
""".strip()
        ),
        "loss": (
            f"SELECT t, run AS series, {average} AS value "
            "FROM metrics WHERE name = 'train_loss' GROUP BY 1, 2 ORDER BY 1"
        ),
        "mfu": (
            f"SELECT t, run AS series, {average} AS value "
            "FROM metrics WHERE name = 'throughput_mfu' GROUP BY 1, 2 ORDER BY 1"
        ),
        "tokens": (
            f"SELECT t, run AS series, {average} AS value "
            "FROM metrics WHERE name = 'throughput_tokens_per_second' GROUP BY 1, 2 ORDER BY 1"
        ),
        "step": (
            "SELECT t, run AS series, MAX(step) AS value FROM metrics WHERE step IS NOT NULL GROUP BY 1, 2 ORDER BY 1"
        ),
        "step_time": (
            """
SELECT t,
       SUM(CASE WHEN name = 'throughput_duration' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_duration' THEN sample_count END), 0) AS step,
       SUM(CASE WHEN name = 'throughput_loading_time' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_loading_time' THEN sample_count END), 0) AS loading,
       SUM(CASE WHEN name = 'throughput_hook_time' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_hook_time' THEN sample_count END), 0) AS hooks
FROM metrics
WHERE name IN ('throughput_duration', 'throughput_loading_time', 'throughput_hook_time')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "power": "SELECT t, run AS series, value FROM power ORDER BY t, run",
        "evaluation": (
            f"""
SELECT t, run || ' · ' || name AS series, {average} AS value
FROM metrics WHERE name LIKE 'eval_%'
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
    }
    return DashboardDataset(
        name="Runs overview",
        cache_key=(clusters, runs, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("metrics", metrics_sql, RUNS_MAX_METRIC_ROWS),
            SourceQuery("power", power_sql, RUNS_MAX_POWER_ROWS),
        ),
        setup_sql=(),
        views=views,
        max_result_rows=RUNS_MAX_RESULT_ROWS,
    )
