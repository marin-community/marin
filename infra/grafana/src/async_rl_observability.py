# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded sources shared by the asynchronous RL post-training dashboard."""

from dashboard_dataset import DashboardDataset, SourceQuery, bounded_bucket_ms, validate_value, validate_values
from vllm_observability import sql_string, sql_values

ASYNC_RL_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
ASYNC_RL_MAX_POINTS = 360
ASYNC_RL_MIN_BUCKET_MS = 30_000
ASYNC_RL_MAX_CLUSTERS = 16
ASYNC_RL_MAX_EXECUTIONS = 32
ASYNC_RL_MAX_IDENTITY_LENGTH = 512
ASYNC_RL_MAX_CORE_ROWS = 100_000
ASYNC_RL_MAX_METRIC_ROWS = 100_000
ASYNC_RL_MAX_SPAN_ROWS = 50_000
ASYNC_RL_MAX_STEP_ROWS = 50_000
ASYNC_RL_MAX_PROCESS_ROWS = 10_000
ASYNC_RL_MAX_RESULT_ROWS = 200_000
MEGATRON_DETAIL_ROWS = 5_000
GIB = 1073741824
FINITE_VALUE_LIMIT_SQL = "1e308"
M2_REFERENCE = 0.04
MARINSKYRL_TABLE = '"telemetry_v1.marinskyrl"'

# Loop events aggregated per display bucket and execution. Gauges keep their latest observation
# per process; the dwell, staleness and rollout-call names carry percentiles as well.
_CORE_NAMES = (
    "event_loop_lag_seconds",
    "phase_duration_seconds",
    "policy_step",
    "rollout_buffer_dwell_seconds",
    "rollout_capacity",
    "rollout_group_tokens",
    "rollout_groups",
    "rollout_queue_depth",
    "rollout_staleness_steps",
    "rollout_wait_seconds",
    "rollout_waits",
    "weight_sync_completed",
    "work_completed",
)
_GAUGE_NAMES = ("rollout_capacity", "rollout_queue_depth")
_LIFECYCLE_NAMES = ("lifecycle", "terminal")
_EXPORTER_NAMES = ("telemetry_lost_records", "telemetry_rejected_records", "training_nonfinite_values")
_PERCENTILE_NAMES = ("rollout_buffer_dwell_seconds", "rollout_staleness_steps")

_REWARD_METRICS = ("reward/avg_raw_reward", "reward/avg_pass_at_4", "reward/informative_group_fraction")
_LENGTH_STOP_METRICS = ("consumed/length_stop_fraction", "consumed/stop_reason_coverage")
_OPTIMIZER_METRICS = (
    "policy/policy_entropy",
    "policy/raw_grad_norm",
    "policy/policy_kl",
    "policy/final_loss",
    "policy/policy_loss",
)
_DRIFT_METRICS = (
    "policy/mismatch/pooled/log_ratio_mean",
    "policy/mismatch/pooled/log_ratio_abs_mean",
    "policy/mismatch/pooled/log_ratio_abs_p95",
    "policy/mismatch/pooled/log_ratio_abs_p99",
    "policy/mismatch/pooled/log_ratio_abs_max",
)
_CLIP_PRESSURE_METRICS = ("policy/mismatch/pooled/lower_clip_pressure", "policy/mismatch/pooled/upper_clip_pressure")
_DRIFT_COVERAGE_METRICS = (
    "policy/mismatch/pooled/finite_fraction",
    "policy/mismatch/pooled/missing_behavior",
    "policy/mismatch/pooled/ess_fraction",
)
_LOG_RATIO_SQUARED_METRICS = ("policy/mismatch/pooled/log_ratio_mean_squared",)
_TIS_SKIP_METRICS = ("tis/batch_skipped_no_logprobs", "tis/skipped_fraction")
_CORE_CYCLE_METRICS = (
    "async/performance/core_seconds",
    "async/performance/cycle_seconds",
    "async/performance/outside_core_seconds",
)
_LOSS_TOKEN_RATE_METRICS = (
    "async/performance/consumed_loss_tokens_per_core_second",
    "async/performance/consumed_loss_tokens_per_cycle_second",
)
_CORE_FRACTION_METRICS = (
    "async/performance/buffer_wait_fraction",
    "async/performance/training_fraction",
    "async/performance/weight_sync_fraction",
)
_ROLE_GPU_TOKEN_METRICS = (
    "async/performance/loss_tokens_per_configured_policy_gpu_second",
    "async/performance/response_tokens_per_configured_inference_gpu_second",
)
_ROLE_GPU_METRICS = ("async/performance/configured_policy_gpus", "async/performance/configured_inference_gpus")
_CORE_GPU_HOUR_METRICS = ("async/performance/core_seconds", *_ROLE_GPU_METRICS)
_UNIFORM_STALENESS_METRICS = (
    "async/staleness_min",
    "async/staleness_max",
    "async/performance/consumed_loss_tokens",
    "async/performance/consumed_response_tokens",
    "consumed/sequences",
    "policy/mismatch/pooled/log_ratio_mean_squared",
    "policy/mismatch/pooled/ess_fraction",
    "reward/avg_raw_reward",
    "policy/mismatch/pooled/finite_fraction",
    "policy/mismatch/pooled/missing_behavior",
    "policy/policy_loss",
)
_MISMATCH_STALENESS0_METRICS = (
    "policy/mismatch/staleness0/log_ratio_abs_mean",
    "policy/mismatch/staleness0/log_ratio_abs_p99",
    "policy/mismatch/staleness0/log_ratio_abs_p999",
    "policy/mismatch/staleness0/frac_outside_0_5_2",
    "policy/mismatch/staleness0/ess_fraction",
    "policy/mismatch/staleness0/kl_k3",
    "policy/mismatch/staleness0/chi2",
)
_MISMATCH_BY_STALENESS_METRICS = tuple(
    f"policy/mismatch/staleness{bucket}/log_ratio_abs_mean" for bucket in ("0", "1", "2", "3", "4-7", "8+")
)
_LEARNER_DRIFT_METRICS = (
    "policy/log_ratio_mean",
    "policy/log_ratio_abs_mean",
    "policy/log_ratio_abs_p99",
    "policy/log_ratio_abs_p999",
    "policy/log_ratio_ess_fraction",
    "policy/log_ratio_kl_k3",
    "policy/log_ratio_chi2",
)
_POSITION_METRICS = (
    "policy/mismatch/pooled/pos_first256/log_ratio_abs_mean",
    "policy/mismatch/pooled/pos_last256/log_ratio_abs_mean",
    "policy/mismatch/pooled/pos_middle/log_ratio_abs_mean",
    "policy/log_ratio_pos_first256/log_ratio_abs_mean",
    "policy/log_ratio_pos_last256/log_ratio_abs_mean",
    "policy/log_ratio_pos_middle/log_ratio_abs_mean",
)
_GRADIENT_METRICS = (
    "policy/grad_cosine",
    "policy/grad_cosine_min",
    "policy/grad_cosine_max",
    "policy/grad_norm_reduced",
    "policy/raw_grad_norm",
)
_CORRECTION_METRICS = (
    "policy/offpolicy_mask/masked_fraction",
    "policy/offpolicy_mask/vetoed_sequence_fraction",
    "policy/m2_mask/m2_before",
    "policy/m2_mask/masked_fraction",
    "policy/ppo_clip_ratio",
    "policy/tis/imp_ratio_capped_fraction",
)
_METRIC_NAMES = tuple(
    sorted(
        {
            *_REWARD_METRICS,
            *_LENGTH_STOP_METRICS,
            *_OPTIMIZER_METRICS,
            *_DRIFT_METRICS,
            *_CLIP_PRESSURE_METRICS,
            *_DRIFT_COVERAGE_METRICS,
            *_LOG_RATIO_SQUARED_METRICS,
            *_TIS_SKIP_METRICS,
            *_CORE_CYCLE_METRICS,
            *_LOSS_TOKEN_RATE_METRICS,
            *_CORE_FRACTION_METRICS,
            *_ROLE_GPU_TOKEN_METRICS,
            *_ROLE_GPU_METRICS,
            *_UNIFORM_STALENESS_METRICS,
            *_MISMATCH_STALENESS0_METRICS,
            *_MISMATCH_BY_STALENESS_METRICS,
            *_LEARNER_DRIFT_METRICS,
            *_POSITION_METRICS,
            *_GRADIENT_METRICS,
            *_CORRECTION_METRICS,
        }
    )
)
_EVALUATION_LENGTH_SUFFIXES = (
    "response_tokens_mean",
    "response_tokens_max",
    "length_stop_fraction",
    "completed_stop_fraction",
    "stop_reason_coverage",
)
_EVALUATION_CONTRIBUTION_SUFFIXES = ("avg_score", "length_stop_score_contribution", "completed_stop_score_contribution")

# Optimizer batches whose every diagnostic is present, single-valued, finite and integral where a
# count is expected; the uniform-staleness table and its coverage stat share this classification.
_UNIFORM_STALENESS_BATCHES = f"""
WITH observed AS (
    SELECT job_id, execution_uid, CAST(step AS BIGINT) AS step, metric,
           MIN(value) AS minimum, MAX(value) AS maximum,
           COUNT(*) AS samples, COUNT(value) AS values_present
    FROM metrics
    WHERE metric IN ({sql_values(_UNIFORM_STALENESS_METRICS)}) AND payload_kind = 'train'
      AND CAST(step AS BIGINT) > 0
    GROUP BY 1, 2, 3, 4
), batches AS (
    SELECT job_id, execution_uid, step,
           MAX(CASE WHEN metric = 'async/staleness_min' THEN maximum END) AS staleness_min,
           MAX(CASE WHEN metric = 'async/staleness_max' THEN maximum END) AS staleness_max,
           MAX(CASE WHEN metric = 'async/performance/consumed_loss_tokens' THEN maximum END) AS loss_tokens,
           MAX(CASE WHEN metric = 'async/performance/consumed_response_tokens' THEN maximum END) AS response_tokens,
           MAX(CASE WHEN metric = 'consumed/sequences' THEN maximum END) AS sequences,
           MAX(CASE WHEN metric = 'policy/mismatch/pooled/log_ratio_mean_squared' THEN maximum END) AS mslr,
           MAX(CASE WHEN metric = 'policy/mismatch/pooled/ess_fraction' THEN maximum END) AS ess,
           MAX(CASE WHEN metric = 'reward/avg_raw_reward' THEN maximum END) AS reward,
           MAX(CASE WHEN metric = 'policy/mismatch/pooled/finite_fraction' THEN maximum END) AS finite_fraction,
           MAX(CASE WHEN metric = 'policy/mismatch/pooled/missing_behavior' THEN maximum END) AS missing_behavior,
           MAX(CASE WHEN metric = 'policy/policy_loss' THEN maximum END) AS policy_loss,
           SUM(CASE WHEN minimum <> maximum OR samples <> values_present THEN 1 ELSE 0 END) AS conflicts,
           SUM(CASE WHEN metric = 'async/performance/consumed_loss_tokens'
                     AND (samples <> values_present OR minimum <> maximum OR minimum < 0
                          OR maximum >= {FINITE_VALUE_LIMIT_SQL} OR maximum <> FLOOR(maximum))
                    THEN 1 ELSE 0 END) AS loss_conflicts
    FROM observed GROUP BY 1, 2, 3
), classified AS (
    SELECT *,
           CASE WHEN conflicts = 0 AND staleness_min = staleness_max AND staleness_min >= 0
                     AND staleness_min = FLOOR(staleness_min)
                     AND loss_tokens > 0 AND loss_tokens < {FINITE_VALUE_LIMIT_SQL}
                     AND loss_tokens = FLOOR(loss_tokens)
                     AND response_tokens >= 0 AND response_tokens < {FINITE_VALUE_LIMIT_SQL}
                     AND response_tokens = FLOOR(response_tokens)
                     AND sequences > 0 AND sequences < {FINITE_VALUE_LIMIT_SQL}
                     AND sequences = FLOOR(sequences)
                     AND mslr >= 0 AND mslr < {FINITE_VALUE_LIMIT_SQL} AND ess > 0 AND ess <= 1
                     AND finite_fraction = 1 AND missing_behavior = 0
                     AND reward BETWEEN -{FINITE_VALUE_LIMIT_SQL} AND {FINITE_VALUE_LIMIT_SQL}
                     AND policy_loss BETWEEN -{FINITE_VALUE_LIMIT_SQL} AND {FINITE_VALUE_LIMIT_SQL}
                THEN 1 ELSE 0 END AS eligible
    FROM batches
)
""".strip()


def _metric_points_with_series(predicate: str, series: str) -> str:
    return (
        f"SELECT timestamp_ms AS t, {series} AS series, value FROM metrics WHERE {predicate} ORDER BY timestamp_ms, seq"
    )


def _metric_points(predicate: str) -> str:
    """Return training metric points labelled by metric and execution."""
    return _metric_points_with_series(predicate, "metric || ' · ' || execution_uid")


def _payload_metric_points(predicate: str) -> str:
    """Return training metric points labelled by metric, payload kind, and execution."""
    return _metric_points_with_series(predicate, "metric || ' ' || payload_kind || ' · ' || execution_uid")


def _train_metric_points(metrics: tuple[str, ...]) -> str:
    return _metric_points(f"metric IN ({sql_values(metrics)}) AND payload_kind = 'train'")


def _evaluation_points(suffixes: tuple[str, ...]) -> str:
    patterns = " OR ".join(f"metric LIKE 'eval/%/{suffix}'" for suffix in suffixes)
    return _metric_points(f"metric LIKE 'eval/%' AND ({patterns})")


def _percentile_series(name: str) -> str:
    return f"""
SELECT t, summary || ' · ' || execution_uid AS series,
       CASE WHEN summary = 'p50' THEN p50 WHEN summary = 'p95' THEN p95 ELSE max_value END AS value
FROM core CROSS JOIN (VALUES ('p50'), ('p95'), ('max')) AS summaries(summary)
WHERE statistic = 'percentile' AND name = {sql_string(name)}
ORDER BY 1
""".strip()


def async_rl_overview_dataset(
    clusters: tuple[str, ...],
    run: str,
    job: str,
    executions: tuple[str, ...],
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int = ASYNC_RL_MIN_BUCKET_MS,
) -> DashboardDataset:
    """Build the bounded sources behind every panel of the asynchronous RL dashboard."""
    validate_values("clusters", clusters, max_values=ASYNC_RL_MAX_CLUSTERS, max_length=128)
    validate_value("run", run, max_length=ASYNC_RL_MAX_IDENTITY_LENGTH)
    validate_value("job", job, max_length=ASYNC_RL_MAX_IDENTITY_LENGTH)
    validate_values(
        "executions",
        executions,
        max_values=ASYNC_RL_MAX_EXECUTIONS,
        max_length=ASYNC_RL_MAX_IDENTITY_LENGTH,
    )
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=ASYNC_RL_MAX_WINDOW_MS,
        max_window_error="async RL overview range must not exceed 7 days",
        min_bucket_ms=ASYNC_RL_MIN_BUCKET_MS,
        max_points=ASYNC_RL_MAX_POINTS,
    )
    bucket = f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms}"
    identity = f"""service = 'marinskyrl'
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({sql_values(clusters)})
      AND run_id = {sql_string(run)} AND job_id = {sql_string(job)}
      AND execution_uid IN ({sql_values(executions)})
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}"""
    core_sql = f"""
WITH selected AS (
    SELECT {bucket} AS t,
           name,
           execution_uid,
           resource_attributes_json,
           timestamp_ms,
           seq,
           json_get(attributes_json, 'work_kind') AS work_kind,
           json_get(attributes_json, 'phase') AS phase,
           json_get(attributes_json, 'outcome') AS outcome,
           json_get(attributes_json, 'role') AS role,
           json_get(attributes_json, 'rank') AS rank,
           json_get(attributes_json, 'wait') AS wait,
           json_get(attributes_json, 'stat') AS stat,
           json_get(attributes_json, 'disposition') AS disposition,
           CAST(json_get(body_json, 'model_version_step') AS DOUBLE) AS weights_step,
           value
    FROM {MARINSKYRL_TABLE}
    WHERE {identity}
      AND name IN ({sql_values(_CORE_NAMES)})
), aggregates AS (
    SELECT 'aggregate' AS statistic,
           t, name, execution_uid, work_kind, phase, outcome, role, rank, wait, stat, disposition,
           MAX(weights_step) AS weights_step,
           SUM(value) AS sum_value,
           COUNT(value) AS sample_count,
           MIN(value) AS min_value,
           MAX(value) AS max_value,
           CAST(NULL AS DOUBLE) AS p50,
           CAST(NULL AS DOUBLE) AS p95
    FROM selected
    WHERE name NOT IN ({sql_values(_GAUGE_NAMES + _PERCENTILE_NAMES)})
    GROUP BY 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
), latest AS (
    SELECT 'latest' AS statistic,
           t, name, execution_uid,
           CAST(NULL AS VARCHAR) AS work_kind,
           CAST(NULL AS VARCHAR) AS phase,
           CAST(NULL AS VARCHAR) AS outcome,
           CAST(NULL AS VARCHAR) AS role,
           CAST(NULL AS VARCHAR) AS rank,
           CAST(NULL AS VARCHAR) AS wait,
           CAST(NULL AS VARCHAR) AS stat,
           CAST(NULL AS VARCHAR) AS disposition,
           CAST(NULL AS DOUBLE) AS weights_step,
           value AS sum_value,
           CAST(1 AS BIGINT) AS sample_count,
           value AS min_value,
           value AS max_value,
           CAST(NULL AS DOUBLE) AS p50,
           CAST(NULL AS DOUBLE) AS p95
    FROM (
        SELECT t, name, execution_uid, value,
               ROW_NUMBER() OVER (
                   PARTITION BY t, execution_uid, resource_attributes_json, name
                   ORDER BY timestamp_ms DESC, seq DESC
               ) AS newest
        FROM selected WHERE name IN ({sql_values(_GAUGE_NAMES)})
    ) WHERE newest = 1
), percentiles AS (
    SELECT 'percentile' AS statistic,
           t, name, execution_uid,
           CAST(NULL AS VARCHAR) AS work_kind,
           CAST(NULL AS VARCHAR) AS phase,
           CAST(NULL AS VARCHAR) AS outcome,
           CAST(NULL AS VARCHAR) AS role,
           CAST(NULL AS VARCHAR) AS rank,
           CAST(NULL AS VARCHAR) AS wait,
           CAST(NULL AS VARCHAR) AS stat,
           CAST(NULL AS VARCHAR) AS disposition,
           CAST(NULL AS DOUBLE) AS weights_step,
           SUM(value) AS sum_value,
           COUNT(value) AS sample_count,
           MIN(value) AS min_value,
           MAX(value) AS max_value,
           approx_percentile_cont(value, 0.5) AS p50,
           approx_percentile_cont(value, 0.95) AS p95
    FROM selected
    WHERE name = 'rollout_staleness_steps'
       OR (name = 'rollout_buffer_dwell_seconds' AND disposition = 'consumed')
       OR (name = 'phase_duration_seconds' AND phase = 'rollout_call' AND outcome = 'success')
    GROUP BY 2, 3, 4
)
SELECT * FROM aggregates
UNION ALL SELECT * FROM latest
UNION ALL SELECT * FROM percentiles
ORDER BY t, name, execution_uid
LIMIT {ASYNC_RL_MAX_CORE_ROWS + 1}
""".strip()
    metrics_sql = f"""
SELECT timestamp_ms, seq, job_id, execution_uid,
       json_get(attributes_json, 'metric') AS metric,
       json_get(attributes_json, 'payload_kind') AS payload_kind,
       json_get(attributes_json, 'role') AS role,
       json_get(attributes_json, 'step') AS step,
       value
FROM {MARINSKYRL_TABLE}
WHERE {identity}
  AND name = 'training_metric_value'
  AND (json_get(attributes_json, 'metric') IN ({sql_values(_METRIC_NAMES)})
       OR json_get(attributes_json, 'metric') LIKE 'eval/%')
ORDER BY timestamp_ms, seq
LIMIT {ASYNC_RL_MAX_METRIC_ROWS + 1}
""".strip()
    staleness_sql = f"""
SELECT job_id, execution_uid, name, step, staleness,
       COUNT(*) AS groups,
       SUM(tokens) AS tokens,
       MAX(timestamp_ms) AS observed_ms
FROM (
    SELECT job_id, execution_uid, name, timestamp_ms,
           json_get(attributes_json, 'step') AS step,
           CASE WHEN name = 'rollout_staleness_steps' THEN CAST(value AS BIGINT)
                ELSE CAST(json_get(body_json, 'staleness') AS BIGINT) END AS staleness,
           CAST(json_get(body_json, 'response_tokens') AS BIGINT) AS tokens
    FROM {MARINSKYRL_TABLE}
    WHERE {identity}
      AND name IN ('rollout_staleness_steps', 'consumed_staleness')
      AND json_get(attributes_json, 'role') = 'trainer'
)
GROUP BY 1, 2, 3, 4, 5
ORDER BY observed_ms, execution_uid, name, staleness
LIMIT {ASYNC_RL_MAX_STEP_ROWS + 1}
""".strip()
    process = (
        "COALESCE(json_get(resource_attributes_json, 'actor_uid'), json_get(resource_attributes_json, 'ray_task_id'), "
        "json_get(resource_attributes_json, 'host'))"
    )
    processes_sql = f"""
WITH events AS (
    SELECT execution_uid, resource_attributes_json, name, body_json, timestamp_ms,
           ROW_NUMBER() OVER (
               PARTITION BY execution_uid, resource_attributes_json
               ORDER BY timestamp_ms DESC, seq DESC
           ) AS newest
    FROM {MARINSKYRL_TABLE}
    WHERE {identity}
      AND name IN ({sql_values(_LIFECYCLE_NAMES)})
), lifecycle AS (
    SELECT execution_uid,
           json_get(resource_attributes_json, 'role') AS role,
           {process} AS process,
           name,
           COALESCE(json_get(body_json, 'status'), json_get(body_json, 'state')) AS status,
           json_get(body_json, 'reason') AS reason,
           CAST(NULL AS DOUBLE) AS observed_value,
           timestamp_ms AS last_record_ms
    FROM events WHERE newest = 1
), exporter AS (
    SELECT execution_uid,
           json_get(resource_attributes_json, 'role') AS role,
           {process} AS process,
           name,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS VARCHAR) AS reason,
           CASE WHEN name = 'training_nonfinite_values' THEN SUM(value) ELSE MAX(value) END AS observed_value,
           MAX(timestamp_ms) AS last_record_ms
    FROM {MARINSKYRL_TABLE}
    WHERE {identity}
      AND name IN ({sql_values(_EXPORTER_NAMES)})
    GROUP BY execution_uid, resource_attributes_json, name
)
SELECT * FROM lifecycle
UNION ALL SELECT * FROM exporter
ORDER BY last_record_ms DESC, execution_uid, name
LIMIT {ASYNC_RL_MAX_PROCESS_ROWS + 1}
""".strip()
    memory_sql = f"""
WITH m AS (
    SELECT execution_uid, resource_attributes_json,
           json_get(attributes_json, 'rank') AS rank,
           json_get(attributes_json, 'gpu_uuid') AS gpu,
           json_get(attributes_json, 'phase') AS phase,
           body_json
    FROM {MARINSKYRL_TABLE}
    WHERE {identity}
      AND name = 'cuda_memory_observation'
      AND json_get(attributes_json, 'worker_role') = 'policy'
)
SELECT execution_uid AS execution,
       json_get(resource_attributes_json, 'host') AS host,
       rank, gpu, phase,
       COUNT(*) AS observations,
       MAX(CAST(json_get(body_json, 'peak_allocated_bytes') AS DOUBLE)) / {GIB} AS peak_allocated_gib,
       MAX(CAST(json_get(body_json, 'peak_reserved_bytes') AS DOUBLE)) / {GIB} AS peak_reserved_gib,
       MAX(CAST(json_get(body_json, 'allocated_bytes') AS DOUBLE)) / {GIB} AS sampled_allocated_gib,
       MIN(CAST(json_get(body_json, 'device_free_bytes') AS DOUBLE)) / {GIB} AS sampled_free_gib,
       MAX(CAST(json_get(body_json, 'device_total_bytes') AS DOUBLE)) / {GIB} AS device_total_gib
FROM m
GROUP BY execution_uid, resource_attributes_json, rank, gpu, phase
ORDER BY execution, rank, phase
LIMIT {ASYNC_RL_MAX_PROCESS_ROWS + 1}
""".strip()
    megatron_sql = f"""
SELECT execution_uid, timestamp_ms, seq,
       CAST(json_get(attributes_json, 'step') AS BIGINT) AS step,
       json_get(attributes_json, 'rank') AS rank,
       json_get(attributes_json, 'phase') AS phase,
       json_get(attributes_json, 'outcome') AS outcome,
       json_get(attributes_json, 'backend') AS backend,
       value AS seconds
FROM {MARINSKYRL_TABLE}
WHERE {identity}
  AND name = 'phase_duration_seconds'
  AND (json_get(attributes_json, 'backend') = 'megatron' OR json_get(attributes_json, 'phase') = 'ppo_train')
ORDER BY timestamp_ms, seq
LIMIT {ASYNC_RL_MAX_SPAN_ROWS + 1}
""".strip()
    windows_sql = f"""
SELECT execution_uid, name,
       json_get(attributes_json, 'phase') AS phase,
       CASE WHEN name = 'weight_sync_completed' THEN timestamp_ms
            ELSE CAST(json_get(body_json, 'started_unix_ms') AS BIGINT) END AS start_ms,
       CASE WHEN name = 'weight_sync_completed' THEN timestamp_ms + 1
            ELSE CAST(json_get(body_json, 'finished_unix_ms') AS BIGINT) END AS finish_ms
FROM {MARINSKYRL_TABLE}
WHERE {identity}
  AND name IN ('async_phase_window', 'weight_sync_completed')
  AND (name = 'weight_sync_completed' OR json_get(attributes_json, 'phase') IN ('training', 'weight_sync'))
ORDER BY start_ms, execution_uid
LIMIT {ASYNC_RL_MAX_SPAN_ROWS + 1}
""".strip()
    overlap_sql = f"""
WITH r AS (
    SELECT * FROM {MARINSKYRL_TABLE}
    WHERE {identity}
      AND name IN ('async_phase_window', 'rollout_call')
), p AS (
    SELECT execution_uid,
           CAST(json_get(attributes_json, 'step') AS BIGINT) AS step,
           CAST(json_get(body_json, 'started_unix_ms') AS BIGINT) AS started,
           CAST(json_get(body_json, 'finished_unix_ms') AS BIGINT) AS finished
    FROM r
    WHERE name = 'async_phase_window' AND json_get(attributes_json, 'phase') = 'training'
      AND json_get(attributes_json, 'outcome') = 'success'
), c AS (
    SELECT execution_uid,
           json_get(body_json, 'call_id') AS call_id,
           CAST(json_get(body_json, 'started_unix_ms') AS BIGINT) AS started,
           CAST(json_get(body_json, 'finished_unix_ms') AS BIGINT) AS finished,
           CAST(json_get(body_json, 'response_tokens') AS BIGINT) AS tokens
    FROM r
    WHERE name = 'rollout_call' AND json_get(attributes_json, 'outcome') = 'success'
), coverage AS (
    SELECT execution_uid, COUNT(*) AS calls FROM c GROUP BY 1
)
SELECT p.execution_uid, p.step,
       CASE WHEN MIN(p.started) < {start_ms} THEN 'partial interval'
            WHEN MAX(coverage.calls) IS NULL THEN 'no rollout records'
            ELSE 'observed' END AS coverage,
       CASE WHEN MIN(p.started) >= {start_ms} AND MAX(coverage.calls) > 0
            THEN COUNT(DISTINCT c.call_id) END AS completed_calls,
       CASE WHEN MIN(p.started) >= {start_ms} AND MAX(coverage.calls) > 0
            THEN COALESCE(SUM(c.tokens), 0) END AS returned_tokens
FROM p
LEFT JOIN coverage ON p.execution_uid = coverage.execution_uid
LEFT JOIN c ON p.execution_uid = c.execution_uid
           AND c.started < p.finished AND c.finished >= p.started AND c.finished < p.finished
GROUP BY p.execution_uid, p.step
ORDER BY p.step, p.execution_uid
LIMIT {ASYNC_RL_MAX_STEP_ROWS + 1}
""".strip()
    service_sql = f"""
WITH r AS (
    SELECT * FROM {MARINSKYRL_TABLE}
    WHERE {identity}
      AND name IN ('async_phase_window', 'generation_tokens_total', 'prompt_tokens_total')
), windows AS (
    SELECT execution_uid, resource_attributes_json, seq,
           json_get(attributes_json, 'phase') AS phase,
           CAST(json_get(body_json, 'started_unix_ms') AS BIGINT) AS started_ms,
           CAST(json_get(body_json, 'finished_unix_ms') AS BIGINT) AS finished_ms
    FROM r
    WHERE name = 'async_phase_window' AND json_get(attributes_json, 'outcome') = 'success'
      AND CAST(json_get(body_json, 'finished_unix_ms') AS BIGINT)
          > CAST(json_get(body_json, 'started_unix_ms') AS BIGINT)
      AND ABS((CAST(json_get(body_json, 'finished_unix_ms') AS DOUBLE)
               - CAST(json_get(body_json, 'started_unix_ms') AS DOUBLE)) / 1000
              - CAST(json_get(body_json, 'duration_seconds') AS DOUBLE)) < 0.005
), counters AS (
    SELECT execution_uid, resource_attributes_json, name,
           json_get(attributes_json, 'engine') AS engine,
           COALESCE(json_get(attributes_json, 'model_name'), '') AS model,
           timestamp_ms, seq, value
    FROM r
    WHERE name IN ('generation_tokens_total', 'prompt_tokens_total')
      AND json_get(attributes_json, 'metric_source') = 'vllm'
      AND json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
      AND json_get(attributes_json, 'engine') IS NOT NULL
), lagged AS (
    SELECT *,
           LAG(timestamp_ms) OVER (
               PARTITION BY execution_uid, resource_attributes_json, name, engine, model
               ORDER BY timestamp_ms, seq
           ) AS previous_ms,
           LAG(value) OVER (
               PARTITION BY execution_uid, resource_attributes_json, name, engine, model
               ORDER BY timestamp_ms, seq
           ) AS previous_value
    FROM counters
), engines AS (
    SELECT DISTINCT execution_uid, resource_attributes_json, name, engine, model FROM counters
), per_window AS (
    SELECT w.execution_uid, w.resource_attributes_json, w.seq, w.phase, e.name, e.engine, e.model,
           (w.finished_ms - w.started_ms) / 1000.0 AS window_seconds,
           SUM(l.value - l.previous_value) AS tokens,
           SUM((l.timestamp_ms - l.previous_ms) / 1000.0) AS covered_seconds,
           COUNT(l.timestamp_ms) AS intervals
    FROM windows w
    JOIN engines e ON w.execution_uid = e.execution_uid AND w.resource_attributes_json = e.resource_attributes_json
    LEFT JOIN lagged l ON e.execution_uid = l.execution_uid AND e.resource_attributes_json = l.resource_attributes_json
                      AND e.name = l.name AND e.engine = l.engine AND e.model = l.model
                      AND l.previous_ms >= w.started_ms AND l.timestamp_ms <= w.finished_ms
                      AND l.timestamp_ms > l.previous_ms AND l.value >= l.previous_value
    GROUP BY w.execution_uid, w.resource_attributes_json, w.seq, w.phase, e.name, e.engine, e.model,
             w.started_ms, w.finished_ms
)
SELECT execution_uid AS execution,
       json_get(resource_attributes_json, 'host') AS collector,
       phase, engine, model, name AS counter,
       SUM(tokens) / NULLIF(SUM(covered_seconds), 0) AS tokens_per_sampled_second,
       COALESCE(SUM(covered_seconds), 0) / NULLIF(SUM(window_seconds), 0) AS coverage_fraction,
       SUM(intervals) AS intervals,
       SUM(window_seconds) AS phase_seconds
FROM per_window
GROUP BY execution_uid, resource_attributes_json, phase, engine, model, name
ORDER BY execution, phase, engine, counter
LIMIT {ASYNC_RL_MAX_PROCESS_ROWS + 1}
""".strip()
    views = {
        "policy_step": (
            """
SELECT t, name || ' · ' || execution_uid AS series,
       MAX(CASE WHEN name = 'policy_step' THEN max_value ELSE weights_step END) AS value
FROM core
WHERE statistic = 'aggregate' AND name IN ('policy_step', 'weight_sync_completed')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "token_rates": (
            f"""
SELECT t, work_kind || ' · ' || execution_uid AS series, SUM(sum_value) * 1000.0 / {bucket_ms} AS value
FROM core
WHERE statistic = 'aggregate' AND name = 'work_completed'
  AND work_kind IN ('generated_token', 'consumed_response_token', 'consumed_loss_token')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "lifecycle": (
            f"""
SELECT execution_uid, role, process, name AS event, status, reason, last_record_ms
FROM processes WHERE name IN ({sql_values(_LIFECYCLE_NAMES)}) ORDER BY last_record_ms DESC
""".strip()
        ),
        "driver_walls": (
            """
SELECT t, phase || ' · ' || execution_uid AS series,
       SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value
FROM core
WHERE statistic = 'aggregate' AND name = 'phase_duration_seconds'
  AND phase IN ('step', 'wait_for_generation_buffer', 'run_training', 'fwd_logprobs_values_reward',
                'train_critic_and_policy')
  AND COALESCE(outcome, 'success') = 'success' AND role = 'trainer'
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "worker_waits": (
            """
WITH g AS (
    SELECT t, wait, CASE WHEN wait IS NULL THEN NULL ELSE execution_uid END AS execution_uid,
           SUM(CASE WHEN name = 'rollout_wait_seconds' AND stat = 'sum' THEN sum_value END) AS wait_seconds,
           SUM(CASE WHEN name = 'rollout_waits' THEN sum_value END) AS waits,
           MAX(CASE WHEN name = 'rollout_wait_seconds' AND stat = 'max' THEN max_value END) AS longest_wait,
           SUM(CASE WHEN name = 'rollout_wait_seconds' AND stat = 'max' THEN sample_count ELSE 0 END)
               AS longest_samples
    FROM core
    WHERE statistic = 'aggregate' AND name IN ('rollout_wait_seconds', 'rollout_waits')
    GROUP BY 1, 2, 3
)
SELECT t, wait || ' ' || summary || ' · ' || execution_uid AS series,
       CASE WHEN summary = 'mean' THEN wait_seconds / NULLIF(waits, 0) ELSE longest_wait END AS value
FROM g CROSS JOIN (VALUES ('mean'), ('max')) AS summaries(summary)
WHERE summary = 'mean' OR longest_samples > 0
ORDER BY 1
""".strip()
        ),
        "buffer": (
            "SELECT t, name || ' · ' || execution_uid AS series, max_value AS value "
            "FROM core WHERE statistic = 'latest' ORDER BY 1"
        ),
        "buffer_dwell": _percentile_series("rollout_buffer_dwell_seconds"),
        "policy_staleness": _percentile_series("rollout_staleness_steps"),
        "rollout_latency": _percentile_series("phase_duration_seconds"),
        "sync_walls": (
            """
SELECT t, phase || ' ' || COALESCE(outcome, 'success') || ' · ' || execution_uid AS series,
       SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value
FROM core
WHERE statistic = 'aggregate' AND name = 'phase_duration_seconds'
  AND phase IN ('sync_weights', 'init_weight_sync_state', 'offload_policy_model_to_cpu',
                'update_ref_with_policy')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "event_loop_lag": (
            "SELECT t, 'max lag · ' || execution_uid AS series, MAX(max_value) AS value "
            "FROM core WHERE statistic = 'aggregate' AND name = 'event_loop_lag_seconds' "
            "GROUP BY 1, 2 ORDER BY 1"
        ),
        "residuals": (
            """
SELECT t, phase || ' rank ' || COALESCE(rank, 'driver') || ' · ' || execution_uid AS series,
       MIN(min_value) AS value
FROM core
WHERE statistic = 'aggregate' AND name = 'phase_duration_seconds'
  AND phase IN ('rollout_call_residual', 'ppo_train_residual') AND outcome = 'success'
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "training_overlap": (
            "SELECT execution_uid, step, coverage, completed_calls, returned_tokens FROM overlap ORDER BY step"
        ),
        "group_dispositions": (
            "SELECT t, disposition || ' · ' || execution_uid AS series, SUM(sum_value) AS value "
            "FROM core WHERE statistic = 'aggregate' AND name = 'rollout_groups' GROUP BY 1, 2 ORDER BY 1"
        ),
        "group_tokens": (
            "SELECT t, disposition || ' · ' || execution_uid AS series, SUM(sum_value) AS value "
            "FROM core WHERE statistic = 'aggregate' AND name = 'rollout_group_tokens' GROUP BY 1, 2 ORDER BY 1"
        ),
        "reward": _payload_metric_points(f"metric IN ({sql_values(_REWARD_METRICS)})"),
        "evaluation": _payload_metric_points("metric LIKE 'eval/%'"),
        "length_stops": (
            f"""
WITH s AS (
    SELECT execution_uid, step, MAX(timestamp_ms) AS timestamp_ms,
           MAX(CASE WHEN metric = 'consumed/length_stop_fraction' THEN value END) AS length_fraction,
           MAX(CASE WHEN metric = 'consumed/stop_reason_coverage' THEN value END) AS coverage
    FROM metrics
    WHERE metric IN ({sql_values(_LENGTH_STOP_METRICS)}) AND payload_kind = 'train' AND role = 'trainer'
    GROUP BY 1, 2
)
SELECT timestamp_ms AS t, metric || ' · ' || execution_uid AS series,
       CASE WHEN metric = 'consumed/length_stop_fraction' THEN CASE WHEN coverage = 1 THEN length_fraction END
            ELSE coverage END AS value
FROM s CROSS JOIN (VALUES ('consumed/length_stop_fraction'), ('consumed/stop_reason_coverage')) AS names(metric)
ORDER BY t, series
""".strip()
        ),
        "optimizer": _payload_metric_points(f"metric IN ({sql_values(_OPTIMIZER_METRICS)})"),
        "megatron_policy_wall": (
            """
SELECT timestamp_ms AS t, 'rank ' || rank || ' ' || outcome || ' · ' || execution_uid AS series, seconds AS value
FROM megatron WHERE phase = 'ppo_train' ORDER BY timestamp_ms, seq
""".strip()
        ),
        "megatron_phases": (
            f"""
SELECT execution_uid, step, rank, phase, outcome, seconds
FROM megatron WHERE backend = 'megatron' ORDER BY step DESC, rank, phase LIMIT {MEGATRON_DETAIL_ROWS}
""".strip()
        ),
        "exporter": (
            f"""
SELECT execution_uid, process || '/' || role AS process, name, observed_value, last_record_ms
FROM processes WHERE name IN ({sql_values(_EXPORTER_NAMES)}) ORDER BY 3, 1, 2
""".strip()
        ),
        "drift": _train_metric_points(_DRIFT_METRICS),
        "clip_pressure": _train_metric_points(_CLIP_PRESSURE_METRICS),
        "drift_coverage": _train_metric_points(_DRIFT_COVERAGE_METRICS),
        "log_ratio_squared": _train_metric_points(_LOG_RATIO_SQUARED_METRICS),
        "tis_skips": _train_metric_points(_TIS_SKIP_METRICS),
        "core_cycle": _train_metric_points(_CORE_CYCLE_METRICS),
        "loss_token_rates": _train_metric_points(_LOSS_TOKEN_RATE_METRICS),
        "core_fractions": _train_metric_points(_CORE_FRACTION_METRICS),
        "role_gpu_tokens": _train_metric_points(_ROLE_GPU_TOKEN_METRICS),
        "role_gpus": _train_metric_points(_ROLE_GPU_METRICS),
        "learner_memory": (
            """
SELECT execution, host, rank, gpu, phase, observations, peak_allocated_gib, peak_reserved_gib,
       sampled_allocated_gib, sampled_free_gib, device_total_gib
FROM memory ORDER BY execution, rank, phase
""".strip()
        ),
        "inference_service": (
            """
SELECT execution, collector, phase, engine, model, counter, tokens_per_sampled_second, coverage_fraction,
       intervals, phase_seconds
FROM service ORDER BY execution, phase, engine, counter
""".strip()
        ),
        "core_gpu_hours": (
            f"""
WITH steps AS (
    SELECT execution_uid, step, MAX(timestamp_ms) AS timestamp_ms,
           MAX(CASE WHEN metric = 'async/performance/core_seconds' THEN value END) AS seconds,
           MAX(CASE WHEN metric = 'async/performance/configured_policy_gpus' THEN value END) AS policy_gpus,
           MAX(CASE WHEN metric = 'async/performance/configured_inference_gpus' THEN value END) AS inference_gpus
    FROM metrics
    WHERE metric IN ({sql_values(_CORE_GPU_HOUR_METRICS)}) AND payload_kind = 'train'
    GROUP BY 1, 2
)
SELECT timestamp_ms AS t, 'core GPU-hours · ' || execution_uid AS series,
       SUM(seconds * (policy_gpus + inference_gpus) / 3600) OVER (
           PARTITION BY execution_uid ORDER BY CAST(step AS BIGINT)
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS value
FROM steps
WHERE seconds IS NOT NULL AND policy_gpus IS NOT NULL AND inference_gpus IS NOT NULL
ORDER BY timestamp_ms
""".strip()
        ),
        "uniform_staleness": (
            f"""
{_UNIFORM_STALENESS_BATCHES}, grouped AS (
    SELECT 'Uniform-staleness batches' AS status, execution_uid AS execution,
           CAST(staleness_min AS BIGINT) AS staleness,
           COUNT(*) AS updates,
           SUM(response_tokens) / SUM(sequences) AS mean_response_tokens,
           SUM(mslr * loss_tokens) / SUM(loss_tokens) AS token_weighted_mslr,
           MIN(ess) AS minimum_ess_fraction,
           AVG(reward) AS mean_update_raw_reward
    FROM classified WHERE eligible = 1 GROUP BY job_id, execution_uid, staleness_min
)
SELECT COALESCE(grouped.status, 'No qualifying uniform-staleness batches') AS status,
       grouped.execution, grouped.staleness, grouped.updates, grouped.mean_response_tokens,
       grouped.token_weighted_mslr, grouped.minimum_ess_fraction, grouped.mean_update_raw_reward
FROM (VALUES (1)) AS selection(present) LEFT JOIN grouped ON TRUE
ORDER BY execution, staleness
""".strip()
        ),
        "uniform_staleness_coverage": (
            f"""
{_UNIFORM_STALENESS_BATCHES}
SELECT CASE WHEN SUM(loss_conflicts) = 0
            THEN SUM(CASE WHEN eligible = 1 THEN loss_tokens ELSE 0 END) / NULLIF(SUM(loss_tokens), 0)
            END AS uniform_staleness_token_fraction,
       CASE WHEN SUM(loss_conflicts) = 0 THEN SUM(loss_tokens) END AS observed_loss_tokens,
       SUM(CASE WHEN eligible = 1 THEN loss_tokens ELSE 0 END) AS represented_loss_tokens,
       SUM(CASE WHEN staleness_min < staleness_max AND staleness_min >= 0
                     AND staleness_min = FLOOR(staleness_min) AND staleness_max = FLOOR(staleness_max)
                THEN loss_tokens ELSE 0 END) AS mixed_staleness_loss_tokens,
       SUM(CASE WHEN eligible = 1 THEN 1 ELSE 0 END) AS uniform_updates,
       SUM(CASE WHEN eligible = 0 THEN 1 ELSE 0 END) AS excluded_updates
FROM classified
""".strip()
        ),
        "evaluation_lengths": _evaluation_points(_EVALUATION_LENGTH_SUFFIXES),
        "evaluation_contributions": _evaluation_points(_EVALUATION_CONTRIBUTION_SUFFIXES),
        "staleness_groups": (
            """
SELECT MAX(observed_ms) OVER (PARTITION BY job_id, execution_uid, step) AS t,
       'staleness ' || CAST(staleness AS VARCHAR) || ' · ' || execution_uid AS series,
       groups AS value
FROM staleness WHERE name = 'rollout_staleness_steps' ORDER BY t, series
""".strip()
        ),
        "staleness_tokens": (
            """
SELECT MAX(observed_ms) OVER (PARTITION BY job_id, execution_uid, step) AS t,
       'staleness ' || CAST(staleness AS VARCHAR) || ' · ' || execution_uid AS series,
       tokens AS value
FROM staleness WHERE name = 'consumed_staleness' ORDER BY t, series
""".strip()
        ),
        "weight_sync_stages": (
            """
SELECT start_ms AS start, finish_ms AS finish, execution_uid AS execution,
       CASE WHEN name = 'weight_sync_completed' THEN 'weights synced'
            WHEN phase = 'weight_sync' THEN 'weight sync'
            ELSE 'training' END AS state
FROM windows ORDER BY start, execution
""".strip()
        ),
        "mismatch_staleness0": _train_metric_points(_MISMATCH_STALENESS0_METRICS),
        "mismatch_by_staleness": _train_metric_points(_MISMATCH_BY_STALENESS_METRICS),
        "learner_drift": _train_metric_points(_LEARNER_DRIFT_METRICS),
        "position_dependence": _train_metric_points(_POSITION_METRICS),
        "gradient_direction": _train_metric_points(_GRADIENT_METRICS),
        "corrections": (
            f"""
WITH m AS (
    SELECT timestamp_ms, execution_uid, metric, value FROM metrics
    WHERE metric IN ({sql_values(_CORRECTION_METRICS)}) AND payload_kind = 'train'
)
SELECT timestamp_ms AS t,
       CASE WHEN kind = 'reference' THEN 'M2 reference {M2_REFERENCE}' ELSE metric END
            || ' · ' || execution_uid AS series,
       CASE WHEN kind = 'reference' THEN {M2_REFERENCE} ELSE value END AS value
FROM m CROSS JOIN (VALUES ('observation'), ('reference')) AS kinds(kind)
WHERE kind = 'observation' OR metric = 'policy/m2_mask/m2_before'
ORDER BY t, series
""".strip()
        ),
    }
    return DashboardDataset(
        name="async RL overview",
        cache_key=(clusters, run, job, executions, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("core", core_sql, ASYNC_RL_MAX_CORE_ROWS),
            SourceQuery("metrics", metrics_sql, ASYNC_RL_MAX_METRIC_ROWS),
            SourceQuery("staleness", staleness_sql, ASYNC_RL_MAX_STEP_ROWS),
            SourceQuery("processes", processes_sql, ASYNC_RL_MAX_PROCESS_ROWS),
            SourceQuery("memory", memory_sql, ASYNC_RL_MAX_PROCESS_ROWS),
            SourceQuery("megatron", megatron_sql, ASYNC_RL_MAX_SPAN_ROWS),
            SourceQuery("windows", windows_sql, ASYNC_RL_MAX_SPAN_ROWS),
            SourceQuery("overlap", overlap_sql, ASYNC_RL_MAX_STEP_ROWS),
            SourceQuery("service", service_sql, ASYNC_RL_MAX_PROCESS_ROWS),
        ),
        setup_sql=(),
        views=views,
        max_result_rows=ASYNC_RL_MAX_RESULT_ROWS,
    )
