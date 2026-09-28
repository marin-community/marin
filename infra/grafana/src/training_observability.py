# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Three bounded sources shared by the Training run dashboard."""

from dashboard_dataset import DashboardDataset, SourceQuery, bounded_bucket_ms, validate_value
from hero_health import DROP_FRACTION_MAX, ROUTER_BIAS_MAX, ROUTER_ENTROPY_MIN
from vllm_observability import sql_string, sql_values

TRAINING_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
TRAINING_MAX_POINTS = 360
TRAINING_MIN_BUCKET_MS = 15_000
TRAINING_MAX_METRIC_ROWS = 100_000
TRAINING_MAX_RESULT_ROWS = 25_000
TRAINING_MAX_RUN_LENGTH = 512

_METRIC_NAMES = (
    "eval_dropless_loss",
    "eval_dropless_macro_loss",
    "eval_dropless_paloma_macro_loss",
    "eval_loss",
    "eval_macro_loss",
    "eval_paloma_macro_loss",
    "grad_norm_total",
    "memory_in_use_gib",
    "memory_limit_gib",
    "memory_peak_gib",
    "moe_drop_fraction",
    "moe_receiver_drop_fraction",
    "moe_sender_drop_fraction",
    "optim_learning_rate",
    "optim_skip_step_loss_threshold",
    "optim_skipped_step",
    "params_norm_stacked_blocks_stacked_mlp_router_bias",
    "params_norm_total",
    "phase",
    "progress_time_seconds",
    "run_progress",
    "throughput_duration",
    "throughput_hook_time",
    "throughput_loading_time",
    "throughput_mfu",
    "throughput_tokens_per_second",
    "throughput_total_tokens",
    "train_loss",
    "train_router_bias_max",
    "train_router_bias_min",
    "train_router_margin_max",
    "train_router_margin_min",
    "train_router_routing_entropy_mean",
)


def _current_execution_sql(run: str, end_ms: int) -> str:
    return f"""
WITH recent_phase AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           job_id,
           execution_uid,
           timestamp_ms,
           seq
    FROM "levanter.metrics"
    WHERE name = 'phase'
      AND process_index = 0
      AND run_id = {sql_string(run)}
      AND job_id IS NOT NULL
      AND execution_uid IS NOT NULL
      AND timestamp_ms >= {max(0, end_ms - 90 * 60 * 1000)}
      AND timestamp_ms < {end_ms}
), current_execution AS (
    SELECT origin_cluster, job_id, execution_uid
    FROM recent_phase
    ORDER BY timestamp_ms DESC, seq DESC, origin_cluster, job_id, execution_uid
    LIMIT 1
)
""".strip()


def training_overview_dataset(
    run: str,
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
) -> DashboardDataset:
    """Build bounded Training sources and fixed local projections."""
    validate_value("run", run, max_length=TRAINING_MAX_RUN_LENGTH)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=TRAINING_MAX_WINDOW_MS,
        max_window_error="Training overview range must not exceed 7 days",
        min_bucket_ms=TRAINING_MIN_BUCKET_MS,
        max_points=TRAINING_MAX_POINTS,
    )
    bucket = f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms}"
    attempts_start_ms = max(0, end_ms - TRAINING_MAX_WINDOW_MS)
    metrics_sql = f"""
WITH metric_rows AS (
    SELECT {bucket} AS t,
           COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           execution_uid,
           CAST(NULL AS VARCHAR) AS job_id,
           name,
           SUM(value) / NULLIF(COUNT(value), 0) AS value,
           MIN(value) AS min_value,
           MAX(value) AS max_value,
           SUM(value) AS sum_value,
           COUNT(value) AS sample_count,
           MAX(step) AS step,
           MIN(timestamp_ms) AS first_ms,
           MAX(timestamp_ms) AS last_ms
    FROM "levanter.metrics"
    WHERE run_id = {sql_string(run)}
      AND name IN ({sql_values(_METRIC_NAMES)})
      AND timestamp_ms >= {start_ms}
      AND timestamp_ms < {end_ms}
    GROUP BY 1, 2, 3, 5
), attempt_rows AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           execution_uid, job_id, value, step, timestamp_ms, seq
    FROM "levanter.metrics"
    WHERE run_id = {sql_string(run)}
      AND name = 'phase'
      AND process_index = 0
      AND job_id IS NOT NULL
      AND execution_uid IS NOT NULL
      AND timestamp_ms >= {attempts_start_ms}
      AND timestamp_ms < {end_ms}
), attempts AS (
    SELECT MAX(timestamp_ms) AS t,
           origin_cluster,
           execution_uid,
           job_id,
           '__attempt' AS name,
           MAX(value) AS value,
           MIN(value) AS min_value,
           MAX(value) AS max_value,
           SUM(value) AS sum_value,
           COUNT(value) AS sample_count,
           MAX(step) AS step,
           MIN(timestamp_ms) AS first_ms,
           MAX(timestamp_ms) AS last_ms
    FROM attempt_rows
    GROUP BY 2, 3, 4
), current_phase AS (
    SELECT origin_cluster, execution_uid, job_id, value
    FROM attempt_rows
    ORDER BY timestamp_ms DESC, seq DESC, origin_cluster, job_id, execution_uid
    LIMIT 1
), attempt_output AS (
    SELECT attempts.t, attempts.origin_cluster, attempts.execution_uid, attempts.job_id, attempts.name,
           COALESCE(current_phase.value, attempts.value) AS value,
           attempts.min_value, attempts.max_value, attempts.sum_value, attempts.sample_count,
           attempts.step, attempts.first_ms, attempts.last_ms
    FROM attempts
    LEFT JOIN current_phase USING (origin_cluster, execution_uid, job_id)
)
SELECT * FROM metric_rows
UNION ALL
SELECT * FROM attempt_output
ORDER BY t, name, execution_uid
LIMIT {TRAINING_MAX_METRIC_ROWS + 1}
""".strip()

    current_execution = _current_execution_sql(run, end_ms)
    task_state_sql = f"""
{current_execution}, roots AS (
    SELECT state.ts,
           state.pending,
           state.assigned,
           state.building,
           state.running,
           ROW_NUMBER() OVER (ORDER BY state.ts DESC) AS rn
    FROM "iris.task_state" AS state
    JOIN current_execution
      ON COALESCE(NULLIF(state.cluster, ''), 'marin') = current_execution.origin_cluster
     AND (current_execution.job_id = state.root_job_id
          OR current_execution.job_id LIKE CONCAT(state.root_job_id, '/%'))
    WHERE state.ts >= to_timestamp_millis({max(0, end_ms - 15 * 60 * 1000)})
      AND state.ts < to_timestamp_millis({end_ms})
)
SELECT pending + assigned + building + running AS active_tasks,
       running,
       pending,
       assigned,
       building,
       ({end_ms} - CAST(EXTRACT(EPOCH FROM ts) * 1000 AS BIGINT)) / 1000.0 AS task_state_age_seconds
FROM roots WHERE rn = 1
""".strip()
    task_event_sql = f"""
{current_execution}, retry_events AS (
    SELECT event.ts
    FROM "iris.task_event" AS event
    JOIN current_execution
      ON COALESCE(NULLIF(event.cluster, ''), 'marin') = current_execution.origin_cluster
     AND event.task_id LIKE CONCAT(current_execution.job_id, '/%')
    WHERE event.reason IN ('TaskRetryScheduled', 'CoscheduledSiblingRequeued')
      AND event.ts >= to_timestamp_millis({attempts_start_ms})
      AND event.ts < to_timestamp_millis({end_ms})
)
SELECT CAST(COUNT(*) AS BIGINT) AS retry_events,
       ({end_ms} - CAST(EXTRACT(EPOCH FROM MAX(ts)) * 1000 AS BIGINT)) / 1000.0 AS since_last_retry
FROM retry_events
""".strip()

    weighted_average = "SUM(sum_value) / NULLIF(SUM(sample_count), 0)"
    views = {
        "status": (
            f"""
SELECT MAX(CASE WHEN name = 'phase' THEN max_value END) AS phase,
       MAX(step) AS step,
       MAX(CASE WHEN name = 'run_progress' THEN max_value END) AS progress,
       ({end_ms} - CAST(MAX(CASE WHEN name = 'progress_time_seconds' THEN max_value END) * 1000 AS BIGINT))
           / 1000.0 AS since_last_step,
       SUM(CASE WHEN name = 'throughput_duration' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_duration' THEN sample_count END), 0) AS step_time,
       SUM(CASE WHEN name = 'throughput_mfu' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_mfu' THEN sample_count END), 0) AS mfu,
       SUM(CASE WHEN name = 'throughput_tokens_per_second' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_tokens_per_second' THEN sample_count END), 0)
           AS tokens_per_second,
       MAX(CASE WHEN name = 'throughput_total_tokens' THEN max_value END) AS total_tokens,
       SUM(CASE WHEN name = 'train_loss' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'train_loss' THEN sample_count END), 0) AS train_loss,
       ({end_ms} - MAX(last_ms)) / 1000.0 AS sample_age_seconds
FROM training_rows WHERE name <> '__attempt'
""".strip()
        ),
        "loss": (
            """
SELECT t, execution_uid AS series, SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS train_loss
FROM training_rows WHERE name = 'train_loss' AND execution_uid IS NOT NULL
GROUP BY 1, 2 ORDER BY 1, 2
""".strip()
        ),
        "loss_health": (
            """
SELECT t,
       MAX(CASE WHEN name = 'train_loss' THEN max_value END) AS peak_loss,
       SUM(CASE WHEN name = 'optim_skip_step_loss_threshold' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'optim_skip_step_loss_threshold' THEN sample_count END), 0)
           AS reject_threshold,
       SUM(CASE WHEN name = 'optim_skipped_step' THEN sum_value END) AS skipped_steps
FROM training_rows WHERE name IN ('train_loss', 'optim_skip_step_loss_threshold', 'optim_skipped_step')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "step_time": (
            """
SELECT t,
       SUM(CASE WHEN name = 'throughput_duration' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_duration' THEN sample_count END), 0) AS step,
       MAX(CASE WHEN name = 'throughput_duration' THEN max_value END) AS slowest,
       SUM(CASE WHEN name = 'throughput_loading_time' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_loading_time' THEN sample_count END), 0) AS loading,
       SUM(CASE WHEN name = 'throughput_hook_time' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_hook_time' THEN sample_count END), 0) AS hooks
FROM training_rows
WHERE name IN ('throughput_duration', 'throughput_loading_time', 'throughput_hook_time')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "throughput": (
            """
SELECT t,
       SUM(CASE WHEN name = 'throughput_mfu' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_mfu' THEN sample_count END), 0) AS mfu,
       SUM(CASE WHEN name = 'throughput_tokens_per_second' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'throughput_tokens_per_second' THEN sample_count END), 0)
           AS tokens_per_second
FROM training_rows WHERE name IN ('throughput_mfu', 'throughput_tokens_per_second')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "progress": (
            """
SELECT t,
       MAX(CASE WHEN name = 'run_progress' THEN max_value END) AS progress,
       MAX(CASE WHEN name = 'throughput_total_tokens' THEN max_value END) AS total_tokens
FROM training_rows WHERE name IN ('run_progress', 'throughput_total_tokens')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "optimizer": (
            """
SELECT t,
       SUM(CASE WHEN name = 'grad_norm_total' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'grad_norm_total' THEN sample_count END), 0) AS grad_norm,
       SUM(CASE WHEN name = 'optim_learning_rate' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'optim_learning_rate' THEN sample_count END), 0) AS learning_rate
FROM training_rows WHERE name IN ('grad_norm_total', 'optim_learning_rate')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "evaluation_loss": (
            f"""
SELECT t, name AS series, {weighted_average} AS value
FROM training_rows
WHERE name IN ('eval_loss', 'eval_macro_loss', 'eval_paloma_macro_loss', 'eval_dropless_loss',
               'eval_dropless_macro_loss', 'eval_dropless_paloma_macro_loss')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "device_memory": (
            """
SELECT t,
       SUM(CASE WHEN name = 'memory_in_use_gib' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'memory_in_use_gib' THEN sample_count END), 0) AS in_use,
       MAX(CASE WHEN name = 'memory_peak_gib' THEN max_value END) AS peak,
       MAX(CASE WHEN name = 'memory_limit_gib' THEN max_value END) AS limit_gib
FROM training_rows WHERE name IN ('memory_in_use_gib', 'memory_peak_gib', 'memory_limit_gib')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "execution_attempt": (
            f"""
SELECT ({end_ms} - first_ms) / 1000.0 AS attempt_age_seconds,
       CASE WHEN value = 0 THEN ({end_ms} - first_ms) / 1000.0 END AS initialization_age_seconds
FROM training_rows WHERE name = '__attempt'
ORDER BY last_ms DESC LIMIT 1
""".strip()
        ),
        "execution_tasks": "SELECT * FROM task_state",
        "execution_retries": "SELECT * FROM task_events",
        "token_drops": (
            f"""
SELECT t,
       SUM(CASE WHEN name = 'moe_drop_fraction' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'moe_drop_fraction' THEN sample_count END), 0) AS drop_fraction,
       SUM(CASE WHEN name = 'moe_sender_drop_fraction' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'moe_sender_drop_fraction' THEN sample_count END), 0)
           AS sender_drop_fraction,
       SUM(CASE WHEN name = 'moe_receiver_drop_fraction' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'moe_receiver_drop_fraction' THEN sample_count END), 0)
           AS receiver_drop_fraction,
       CAST({DROP_FRACTION_MAX} AS DOUBLE) AS alert_threshold
FROM training_rows
WHERE name IN ('moe_drop_fraction', 'moe_sender_drop_fraction', 'moe_receiver_drop_fraction')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "router_health": (
            f"""
SELECT t,
       SUM(CASE WHEN name = 'train_router_routing_entropy_mean' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'train_router_routing_entropy_mean' THEN sample_count END), 0)
           AS routing_entropy,
       MAX(CASE WHEN name = 'train_router_bias_max' THEN max_value END) AS router_bias_max,
       MIN(CASE WHEN name = 'train_router_bias_min' THEN min_value END) AS router_bias_min,
       MAX(CASE WHEN name = 'train_router_margin_max' THEN max_value END) AS margin_max,
       MIN(CASE WHEN name = 'train_router_margin_min' THEN min_value END) AS margin_min,
       CAST({ROUTER_ENTROPY_MIN} AS DOUBLE) AS entropy_minimum,
       CAST({ROUTER_BIAS_MAX} AS DOUBLE) AS bias_upper_limit,
       CAST({-ROUTER_BIAS_MAX} AS DOUBLE) AS bias_lower_limit
FROM training_rows
WHERE name IN ('train_router_routing_entropy_mean', 'train_router_bias_max', 'train_router_bias_min',
               'train_router_margin_max', 'train_router_margin_min')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "parameter_norms": (
            """
SELECT t,
       SUM(CASE WHEN name = 'params_norm_total' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'params_norm_total' THEN sample_count END), 0) AS total_norm,
       SUM(CASE WHEN name = 'params_norm_stacked_blocks_stacked_mlp_router_bias' THEN sum_value END)
           / NULLIF(SUM(CASE WHEN name = 'params_norm_stacked_blocks_stacked_mlp_router_bias'
                             THEN sample_count END), 0) AS router_bias_norm
FROM training_rows
WHERE name IN ('params_norm_total', 'params_norm_stacked_blocks_stacked_mlp_router_bias')
GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "attempts": (
            """
SELECT first_ms AS started_ms,
       origin_cluster AS cluster,
       job_id AS job,
       (last_ms - first_ms) / 1000.0 AS active_seconds,
       CASE WHEN origin_cluster = 'marin' THEN 'local' ELSE origin_cluster END AS iris_cluster
FROM training_rows WHERE name = '__attempt'
ORDER BY started_ms DESC
""".strip()
        ),
    }
    return DashboardDataset(
        name="Training overview",
        cache_key=(run, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("training_rows", metrics_sql, TRAINING_MAX_METRIC_ROWS),
            SourceQuery("task_state", task_state_sql, 1),
            SourceQuery("task_events", task_event_sql, 1),
        ),
        setup_sql=(),
        views=views,
        max_result_rows=TRAINING_MAX_RESULT_ROWS,
    )
