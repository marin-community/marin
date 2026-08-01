# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded Finelog query for one centralized Marin vLLM serve."""

import math
from dataclasses import dataclass
from enum import StrEnum

VLLM_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
VLLM_MAX_POINTS = 720
VLLM_MAX_RESULT_ROWS = 10_000
VLLM_MIN_BUCKET_MS = 15_000
VLLM_SCRAPE_INTERVAL_MS = 15_000
VLLM_SNAPSHOT_LOOKBACK_MS = 3 * VLLM_SCRAPE_INTERVAL_MS
VLLM_FRESHNESS_THRESHOLD_MS = 3 * VLLM_SCRAPE_INTERVAL_MS
VLLM_MAX_FRESHNESS_DETAILS = 128
VLLM_MAX_IDENTITY_LENGTH = 512
VLLM_OVERVIEW_SECTIONS = frozenset(
    {
        "counter_total",
        "freshness",
        "freshness_detail",
        "latency",
        "request_outcome",
        "saturation",
        "saturation_summary",
        "token_rate",
    }
)


class VllmIdentityField(StrEnum):
    """Canonical resource attributes accepted by the dashboard query."""

    JOB_ID = "job_id"
    ROOT_RUN_UID = "root_run_uid"
    EXECUTION_UID = "execution_uid"


@dataclass(frozen=True)
class VllmOverviewQuery:
    """Validated SQL and canonical parameters for one vLLM overview."""

    sql: str
    identity_field: VllmIdentityField
    identity: str
    start_ms: int
    end_ms: int
    bucket_ms: int


_TOKEN_COUNTERS = ("prompt_tokens_total", "generation_tokens_total")
_PREEMPTION_COUNTERS = ("num_preemptions_total",)
_OUTCOME_COUNTERS = (
    "request_success_total",
    "request_failure_total",
    "request_failures_total",
    "request_timeout_total",
    "request_timeouts_total",
)
_GAUGES = (
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "gpu_cache_usage_perc",
)
_HISTOGRAM_FAMILIES = (
    ("time_to_first_token_seconds", "ttft"),
    ("inter_token_latency_seconds", "tpot"),
    ("time_per_output_token_seconds", "tpot"),
    ("request_queue_time_seconds", "queue"),
    ("e2e_request_latency_seconds", "e2e"),
)
_HISTOGRAM_COMPONENTS = ("bucket", "count", "sum")
_HISTOGRAM_NAMES = tuple(
    f"{family}_{component}" for family, _ in _HISTOGRAM_FAMILIES for component in _HISTOGRAM_COMPONENTS
)
_METRIC_NAMES = (*_TOKEN_COUNTERS, *_PREEMPTION_COUNTERS, *_OUTCOME_COUNTERS, *_GAUGES, *_HISTOGRAM_NAMES)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_values(values: tuple[str, ...]) -> str:
    return ", ".join(_sql_string(value) for value in values)


def _case_for(mapping: tuple[tuple[str, str], ...], expression: str) -> str:
    cases = " ".join(f"WHEN {_sql_string(source)} THEN {_sql_string(target)}" for source, target in mapping)
    return f"CASE {expression} {cases} END"


def _histogram_name_mapping() -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{family}_{component}", canonical)
        for family, canonical in _HISTOGRAM_FAMILIES
        for component in _HISTOGRAM_COMPONENTS
    )


def _histogram_component_mapping() -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{family}_{component}", component) for family, _ in _HISTOGRAM_FAMILIES for component in _HISTOGRAM_COMPONENTS
    )


def _histogram_source_mapping() -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{family}_{component}", family) for family, _ in _HISTOGRAM_FAMILIES for component in _HISTOGRAM_COMPONENTS
    )


def _validate_identity(identity: str) -> None:
    if not identity:
        raise ValueError("identity must not be empty")
    if len(identity) > VLLM_MAX_IDENTITY_LENGTH:
        raise ValueError(f"identity exceeds {VLLM_MAX_IDENTITY_LENGTH} characters")
    if any(ord(character) < 32 for character in identity):
        raise ValueError("identity must not contain control characters")


def _bounded_bucket_ms(start_ms: int, end_ms: int, requested_bucket_ms: int) -> int:
    if requested_bucket_ms <= 0:
        raise ValueError("bucket_ms must be positive")
    minimum_for_result_cap = math.ceil((end_ms - start_ms) / VLLM_MAX_POINTS)
    return min(end_ms - start_ms, max(requested_bucket_ms, VLLM_MIN_BUCKET_MS, minimum_for_result_cap))


def vllm_overview_query(
    identity_field: VllmIdentityField,
    identity: str,
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
) -> VllmOverviewQuery:
    """Render the fixed vLLM overview query after validating its safety bounds."""
    _validate_identity(identity)
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("to must be later than from and both times must be nonnegative")
    if end_ms - start_ms > VLLM_MAX_WINDOW_MS:
        raise ValueError("vLLM overview range must not exceed 7 days")

    bucket_ms = _bounded_bucket_ms(start_ms, end_ms, requested_bucket_ms)
    scan_start_ms = max(0, start_ms - VLLM_SNAPSHOT_LOOKBACK_MS)
    identity_literal = _sql_string(identity)
    metric_names = _sql_values(_METRIC_NAMES)
    token_counters = _sql_values(_TOKEN_COUNTERS)
    preemption_counters = _sql_values(_PREEMPTION_COUNTERS)
    outcome_counters = _sql_values(_OUTCOME_COUNTERS)
    gauges = _sql_values(_GAUGES)
    histogram_names = _sql_values(_HISTOGRAM_NAMES)
    histogram_family = _case_for(_histogram_name_mapping(), "name")
    histogram_component = _case_for(_histogram_component_mapping(), "name")
    histogram_source_family = _case_for(_histogram_source_mapping(), "name")

    sql = f"""
WITH base AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
           service,
           name,
           kind,
           value,
           resource_attributes_json,
           attributes_json,
           timestamp_ms,
           seq
    FROM "telemetry_v1"
    WHERE service = 'vllm'
      AND json_get(resource_attributes_json, '{identity_field.value}') = {identity_literal}
      AND name IN ({metric_names})
      AND timestamp_ms >= {scan_start_ms}
      AND timestamp_ms < {end_ms}
), cumulative_samples AS (
    SELECT *,
           LAG(value) OVER (
               PARTITION BY origin_cluster,
                            service,
                            name,
                            resource_attributes_json,
                            attributes_json
               ORDER BY timestamp_ms, seq
           ) AS previous_value
    FROM base
    WHERE json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
), increments AS (
    SELECT origin_cluster,
           service,
           name,
           resource_attributes_json,
           attributes_json,
           timestamp_ms,
           CASE
               WHEN previous_value IS NULL OR value < previous_value THEN NULL
               ELSE value - previous_value
           END AS delta
    FROM cumulative_samples
    WHERE timestamp_ms >= {start_ms}

    UNION ALL

    SELECT origin_cluster,
           service,
           name,
           resource_attributes_json,
           attributes_json,
           timestamp_ms,
           value AS delta
    FROM base
    WHERE timestamp_ms >= {start_ms}
      AND kind = 'counter'
      AND COALESCE(json_get(attributes_json, 'source_temporality'), '') <> 'cumulative_snapshot'
), token_bins AS (
    SELECT {start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms} AS t,
           name,
           SUM(delta) AS delta
    FROM increments
    WHERE name IN ({token_counters})
      AND delta IS NOT NULL
    GROUP BY 1, 2
), token_rates AS (
    SELECT t,
           name,
           delta / (CASE
               WHEN t + {bucket_ms} <= {end_ms} THEN {bucket_ms}
               ELSE {end_ms} - t
           END / 1000.0) AS value
    FROM token_bins
), counter_totals AS (
    SELECT name, SUM(delta) AS value
    FROM increments
    WHERE name IN ({token_counters}, {preemption_counters})
      AND delta IS NOT NULL
    GROUP BY 1
), canonical_gauge_samples AS (
    SELECT timestamp_ms,
           CASE
               WHEN name IN ('kv_cache_usage_perc', 'gpu_cache_usage_perc') THEN 'kv_cache_usage'
               ELSE name
           END AS name,
           origin_cluster,
           service,
           resource_attributes_json,
           attributes_json,
           value
    FROM base
    WHERE timestamp_ms >= {start_ms}
      AND name IN ({gauges})
      AND json_get(attributes_json, 'source_temporality') = 'current_snapshot'
), gauge_replica_bins AS (
    SELECT {start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms} AS t,
           name,
           origin_cluster,
           service,
           resource_attributes_json,
           attributes_json,
           AVG(value) AS value
    FROM canonical_gauge_samples
    GROUP BY 1, 2, 3, 4, 5, 6
), canonical_gauge_bins AS (
    SELECT t,
           name,
           CASE
               WHEN name = 'kv_cache_usage' THEN AVG(value)
               ELSE SUM(value)
           END AS value
    FROM gauge_replica_bins
    GROUP BY 1, 2
), raw_gauge_peaks AS (
    SELECT name, MAX(value) AS peak
    FROM canonical_gauge_samples
    GROUP BY 1
), gauge_stats AS (
    SELECT bins.name,
           AVG(bins.value) AS average,
           CASE WHEN bins.name = 'kv_cache_usage' THEN raw.peak ELSE MAX(bins.value) END AS peak
    FROM canonical_gauge_bins AS bins
    JOIN raw_gauge_peaks AS raw USING (name)
    GROUP BY 1, raw.peak
), histogram_component_samples AS (
    SELECT origin_cluster,
           service,
           resource_attributes_json,
           attributes_json,
           timestamp_ms,
           name,
           {histogram_source_family} AS source_family,
           {histogram_family} AS family,
           {histogram_component} AS component,
           json_get(attributes_json, 'le') AS upper_bound,
           CASE
               WHEN previous_value IS NULL OR value < previous_value THEN NULL
               ELSE value - previous_value
           END AS delta,
           CASE WHEN previous_value IS NULL OR value < previous_value THEN 1 ELSE 0 END AS invalid_component
    FROM cumulative_samples
    WHERE name IN ({histogram_names})
), histogram_series AS (
    SELECT DISTINCT origin_cluster,
           service,
           resource_attributes_json,
           source_family,
           name,
           attributes_json
    FROM histogram_component_samples
), histogram_expected_series AS (
    SELECT origin_cluster,
           service,
           resource_attributes_json,
           source_family,
           COUNT(*) AS expected_series
    FROM histogram_series
    GROUP BY 1, 2, 3, 4
), histogram_sample_validity AS (
    SELECT samples.origin_cluster,
           samples.service,
           samples.resource_attributes_json,
           samples.source_family,
           samples.timestamp_ms,
           CASE
               WHEN MAX(samples.invalid_component) = 1 OR COUNT(*) < MAX(expected.expected_series) THEN 0
               ELSE 1
           END AS valid_sample
    FROM histogram_component_samples AS samples
    JOIN histogram_expected_series AS expected
      ON samples.origin_cluster = expected.origin_cluster
     AND samples.service = expected.service
     AND samples.resource_attributes_json = expected.resource_attributes_json
     AND samples.source_family = expected.source_family
    WHERE samples.timestamp_ms >= {start_ms}
    GROUP BY 1, 2, 3, 4, 5
), coherent_histogram_increments AS (
    SELECT samples.family,
           samples.component,
           samples.upper_bound,
           samples.delta
    FROM histogram_component_samples AS samples
    JOIN histogram_sample_validity AS validity
      ON samples.origin_cluster = validity.origin_cluster
     AND samples.service = validity.service
     AND samples.resource_attributes_json = validity.resource_attributes_json
     AND samples.source_family = validity.source_family
     AND samples.timestamp_ms = validity.timestamp_ms
    WHERE validity.valid_sample = 1
), histogram_means AS (
    SELECT family,
           SUM(CASE WHEN component = 'sum' THEN delta ELSE 0 END)
               / NULLIF(SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END), 0) AS mean
    FROM coherent_histogram_increments
    GROUP BY 1
), histogram_buckets AS (
    SELECT family, upper_bound, SUM(delta) AS bucket_count
    FROM coherent_histogram_increments
    WHERE component = 'bucket'
    GROUP BY 1, 2
), histogram_ranked_buckets AS (
    SELECT family,
           upper_bound,
           bucket_count,
           MAX(bucket_count) OVER (PARTITION BY family) AS total_count
    FROM histogram_buckets
), histogram_quantiles AS (
    SELECT family,
           MIN(CASE
               WHEN upper_bound NOT IN ('+Inf', 'Inf') AND bucket_count >= total_count * 0.50
               THEN CAST(upper_bound AS DOUBLE)
           END) AS p50,
           MIN(CASE
               WHEN upper_bound NOT IN ('+Inf', 'Inf') AND bucket_count >= total_count * 0.90
               THEN CAST(upper_bound AS DOUBLE)
           END) AS p90,
           MIN(CASE
               WHEN upper_bound NOT IN ('+Inf', 'Inf') AND bucket_count >= total_count * 0.99
               THEN CAST(upper_bound AS DOUBLE)
           END) AS p99
    FROM histogram_ranked_buckets
    GROUP BY 1
), histogram_stats AS (
    SELECT means.family, means.mean, quantiles.p50, quantiles.p90, quantiles.p99
    FROM histogram_means AS means
    LEFT JOIN histogram_quantiles AS quantiles USING (family)
), histogram_evidence AS (
    SELECT family, 'mean' AS stat, mean AS value FROM histogram_stats
    UNION ALL
    SELECT family, 'p50', p50 FROM histogram_stats
    UNION ALL
    SELECT family, 'p90', p90 FROM histogram_stats
    UNION ALL
    SELECT family, 'p99', p99 FROM histogram_stats
), outcome_totals AS (
    SELECT COALESCE(
               json_get(attributes_json, 'finished_reason'),
               json_get(attributes_json, 'finish_reason'),
               json_get(attributes_json, 'outcome'),
               json_get(attributes_json, 'status'),
               name
           ) AS outcome,
           SUM(delta) AS value
    FROM increments
    WHERE name IN ({outcome_counters})
      AND delta IS NOT NULL
    GROUP BY 1
), producer_samples AS (
    SELECT DISTINCT origin_cluster, service, resource_attributes_json, timestamp_ms
    FROM base
), producer_ordered AS (
    SELECT *,
           LAG(timestamp_ms) OVER (
               PARTITION BY origin_cluster, service, resource_attributes_json
               ORDER BY timestamp_ms
           ) AS previous_timestamp_ms
    FROM producer_samples
), producer_freshness_data AS (
    SELECT origin_cluster,
           service,
           resource_attributes_json,
           MAX(timestamp_ms) AS latest_timestamp_ms,
           SUM(CASE WHEN timestamp_ms >= {start_ms} THEN 1 ELSE 0 END) AS samples,
           MAX(CASE
               WHEN timestamp_ms >= {start_ms} THEN timestamp_ms - previous_timestamp_ms
           END) / 1000.0 AS gap_seconds
    FROM producer_ordered
    GROUP BY 1, 2, 3
), producer_freshness AS (
    SELECT *,
           CASE
               WHEN {end_ms} - latest_timestamp_ms > {VLLM_FRESHNESS_THRESHOLD_MS} THEN 'stale_or_stopped'
               WHEN gap_seconds * 1000 > {VLLM_FRESHNESS_THRESHOLD_MS} THEN 'export_or_scrape_gap'
               ELSE 'fresh'
           END AS freshness_status
    FROM producer_freshness_data
), ranked_freshness AS (
    SELECT *,
           ROW_NUMBER() OVER (
               ORDER BY CASE freshness_status
                            WHEN 'stale_or_stopped' THEN 3
                            WHEN 'export_or_scrape_gap' THEN 2
                            ELSE 1
                        END DESC,
                        {end_ms} - latest_timestamp_ms DESC,
                        gap_seconds DESC,
                        origin_cluster,
                        service,
                        resource_attributes_json
           ) AS freshness_rank
    FROM producer_freshness
), freshness_summary AS (
    SELECT * FROM ranked_freshness WHERE freshness_rank = 1
), output AS (
    SELECT t,
           'token_rate' AS section,
           CASE name
               WHEN 'prompt_tokens_total' THEN 'prompt_tokens'
               ELSE 'generated_tokens'
           END AS metric,
           'rate' AS stat,
           CASE name
               WHEN 'prompt_tokens_total' THEN 'prompt tokens/s'
               ELSE 'generated tokens/s'
           END AS series,
           value,
           'tokens/s' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM token_rates

    UNION ALL

    SELECT CAST(NULL AS BIGINT),
           'counter_total',
           CASE name
               WHEN 'prompt_tokens_total' THEN 'prompt_tokens'
               WHEN 'generation_tokens_total' THEN 'generated_tokens'
               ELSE 'preemptions'
           END,
           'total',
           CASE name
               WHEN 'prompt_tokens_total' THEN 'prompt tokens'
               WHEN 'generation_tokens_total' THEN 'generated tokens'
               ELSE 'preemptions'
           END,
           value,
           CASE WHEN name IN ({token_counters}) THEN 'tokens' ELSE 'requests' END,
           CAST(NULL AS VARCHAR),
           CAST(NULL AS BIGINT),
           CAST(NULL AS DOUBLE)
    FROM counter_totals

    UNION ALL

    SELECT t,
           'saturation',
           name,
           'value',
           name,
           value,
           CASE WHEN name = 'kv_cache_usage' THEN 'ratio' ELSE 'requests' END,
           CAST(NULL AS VARCHAR),
           CAST(NULL AS BIGINT),
           CAST(NULL AS DOUBLE)
    FROM canonical_gauge_bins

    UNION ALL

    SELECT CAST(NULL AS BIGINT),
           'saturation_summary',
           name,
           'average',
           name,
           average,
           CASE WHEN name = 'kv_cache_usage' THEN 'ratio' ELSE 'requests' END,
           CAST(NULL AS VARCHAR),
           CAST(NULL AS BIGINT),
           CAST(NULL AS DOUBLE)
    FROM gauge_stats

    UNION ALL

    SELECT CAST(NULL AS BIGINT),
           'saturation_summary',
           name,
           'peak',
           name,
           peak,
           CASE WHEN name = 'kv_cache_usage' THEN 'ratio' ELSE 'requests' END,
           CAST(NULL AS VARCHAR),
           CAST(NULL AS BIGINT),
           CAST(NULL AS DOUBLE)
    FROM gauge_stats

    UNION ALL

    SELECT CAST(NULL AS BIGINT),
           'latency',
           family,
           stat,
           family,
           value,
           's',
           CAST(NULL AS VARCHAR),
           CAST(NULL AS BIGINT),
           CAST(NULL AS DOUBLE)
    FROM histogram_evidence

    UNION ALL

    SELECT CAST(NULL AS BIGINT),
           'request_outcome',
           'requests',
           'total',
           outcome,
           value,
           'requests',
           CAST(NULL AS VARCHAR),
           CAST(NULL AS BIGINT),
           CAST(NULL AS DOUBLE)
    FROM outcome_totals

    UNION ALL

    SELECT latest_timestamp_ms,
           'freshness',
           'telemetry',
           'latest_sample_age',
           origin_cluster || ':' || service || ':' || resource_attributes_json,
           ({end_ms} - latest_timestamp_ms) / 1000.0,
           's',
           freshness_status,
           CAST(samples AS BIGINT),
           gap_seconds
    FROM freshness_summary

    UNION ALL

    SELECT latest_timestamp_ms,
           'freshness_detail',
           'telemetry',
           'latest_sample_age',
           origin_cluster || ':' || service || ':' || resource_attributes_json,
           ({end_ms} - latest_timestamp_ms) / 1000.0,
           's',
           freshness_status,
           CAST(samples AS BIGINT),
           gap_seconds
    FROM ranked_freshness
    WHERE freshness_rank <= {VLLM_MAX_FRESHNESS_DETAILS}

    UNION ALL

    SELECT CAST(NULL AS BIGINT),
           'freshness',
           'telemetry',
           'latest_sample_age',
           'telemetry',
           CAST(NULL AS DOUBLE),
           's',
           'no_data',
           CAST(0 AS BIGINT),
           CAST(NULL AS DOUBLE)
    WHERE NOT EXISTS (SELECT 1 FROM producer_freshness)
)
SELECT t, section, metric, stat, series, value, unit, status, samples, gap_seconds
FROM output
ORDER BY section, metric, stat, series, t
LIMIT {VLLM_MAX_RESULT_ROWS}
""".strip()
    return VllmOverviewQuery(
        sql=sql,
        identity_field=identity_field,
        identity=identity,
        start_ms=start_ms,
        end_ms=end_ms,
        bucket_ms=bucket_ms,
    )
