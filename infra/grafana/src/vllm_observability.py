# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded Finelog query for one standalone or MarinSkyRL-embedded vLLM serve."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum

import duckdb
import pyarrow as pa
from dashboard_dataset import bounded_bucket_ms, projection_database, validate_table_budget, validate_value
from finelog.errors import QueryResultTooLargeError

VLLM_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
# Leave room for latency/outcome series and bounded producer tables in the shared result.
VLLM_MAX_POINTS = 360
VLLM_MAX_RESULT_ROWS = 10_000
VLLM_MAX_SAMPLES = 1_000_000
VLLM_MAX_SERIES = 50_000
VLLM_DETAIL_MAX_WINDOW_MS = 7 * 60 * 60 * 1000
VLLM_MAX_SUMMARY_ROWS = 1_000
VLLM_MIN_BUCKET_MS = 15_000
VLLM_SCRAPE_INTERVAL_MS = 60_000
VLLM_HISTOGRAM_COHERENCE_MS = 15_000
VLLM_SNAPSHOT_LOOKBACK_MS = 3 * VLLM_SCRAPE_INTERVAL_MS
VLLM_FRESHNESS_THRESHOLD_MS = 3 * VLLM_SCRAPE_INTERVAL_MS
VLLM_MAX_FRESHNESS_DETAILS = 128
VLLM_MAX_IDENTITY_LENGTH = 512
VLLM_OVERVIEW_SECTIONS = frozenset(
    {
        "counter_total",
        "engine_summary",
        "freshness",
        "freshness_detail",
        "latency",
        "length_finish_fraction",
        "output_length_distribution",
        "request_outcome",
        "request_rate",
        "run_summary",
        "run_timeline",
        "diagnostic_status",
        "saturation",
        "saturation_summary",
        "telemetry_health",
        "token_rate",
        "workload",
    }
)


class VllmIdentityField(StrEnum):
    """Structured resource dimensions accepted by the dashboard query."""

    JOB_ID = "job_id"
    RUN_ID = "run_id"
    EXECUTION_UID = "execution_uid"


@dataclass(frozen=True)
class VllmOverviewQuery:
    """Validated SQL and canonical parameters for one vLLM overview."""

    sql: str
    samples_sql: str
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
    ("request_time_per_output_token_seconds", "tpot"),
    ("inter_token_latency_seconds", "inter_token_latency"),
    ("request_queue_time_seconds", "queue"),
    ("request_prefill_time_seconds", "prefill"),
    ("request_decode_time_seconds", "decode"),
    ("e2e_request_latency_seconds", "e2e"),
    ("request_generation_tokens", "output_tokens"),
    ("iteration_tokens_total", "iteration_tokens"),
)
_HISTOGRAM_COMPONENTS = ("bucket", "count", "sum")
_HISTOGRAM_NAMES = tuple(
    f"{family}_{component}" for family, _ in _HISTOGRAM_FAMILIES for component in _HISTOGRAM_COMPONENTS
)
_HISTOGRAM_BASE_NAMES = tuple(family for family, _ in _HISTOGRAM_FAMILIES)
_SERVING_METRIC_NAMES = (
    *_TOKEN_COUNTERS,
    *_PREEMPTION_COUNTERS,
    *_OUTCOME_COUNTERS,
    *_GAUGES,
    *_HISTOGRAM_NAMES,
    *_HISTOGRAM_BASE_NAMES,
)
_HEALTH_METRIC_NAMES = (
    "prometheus_source_available",
    "prometheus_stage_failures",
    "prometheus_dropped_samples",
    "metric_publication_dropped_records",
)
_METRIC_NAMES = (*_SERVING_METRIC_NAMES, *_HEALTH_METRIC_NAMES)
_HISTOGRAM_BOUND_ORDER_SQL = "CASE WHEN upper_bound IN ('+Inf', 'Inf') THEN 1e308 ELSE CAST(upper_bound AS DOUBLE) END"
_PUBLICATION_ID_JSON_FIELD_RE = r'"histogram_publication_id":"(?:\\.|[^"\\])*"'
_SUMMARY_METRIC_NAMES = (
    *_TOKEN_COUNTERS,
    *_PREEMPTION_COUNTERS,
    *_OUTCOME_COUNTERS,
    "time_to_first_token_seconds_count",
    "time_to_first_token_seconds_sum",
    "inter_token_latency_seconds_sum",
    "inter_token_latency_seconds_count",
    "time_to_first_token_seconds",
    "inter_token_latency_seconds",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "gpu_cache_usage_perc",
)


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_values(values: tuple[str, ...]) -> str:
    return ", ".join(sql_string(value) for value in values)


def _vllm_samples_query(
    identity_field: VllmIdentityField,
    identity: str,
    scan_start_ms: int,
    end_ms: int,
    names: tuple[str, ...],
    *,
    compact_histograms: bool = False,
) -> str:
    identity_literal = sql_string(identity)
    metric_names = sql_values(names)
    histogram_fields = "'body_json', body_json, 'publication_id', publication_id"
    if compact_histograms:
        histogram_fields = """'body_json', CAST(NULL AS VARCHAR), 'publication_id', publication_id,
           'histogram_count', CAST(json_get(body_json, 'count') AS BIGINT),
           'histogram_sum', CAST(json_get(body_json, 'sum') AS DOUBLE),
           'histogram_bounds', json_get(body_json, 'explicit_bounds'),
           'producer_epoch', json_get(body_json, 'producer_epoch'),
           'source_sequence', CAST(json_get(body_json, 'sequence') AS BIGINT)"""
    # Finelog serializes attributes as compact, sorted JSON and supports regexp_replace,
    # but not json_merge_patch. Handle a publication token in either key position.
    leading_token = sql_string(r"^\{" + _PUBLICATION_ID_JSON_FIELD_RE + ",")
    other_token = sql_string("," + _PUBLICATION_ID_JSON_FIELD_RE)
    attributes_without_token = (
        f"regexp_replace(regexp_replace(attributes_json, {leading_token}, '{{'), {other_token}, '')"
    )
    return f"""
WITH base AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
           service, name, kind, value, body_json, resource_attributes_json, attributes_json, timestamp_ms, seq
    FROM "telemetry_v1.vllm"
    WHERE service = 'vllm'
      AND {identity_field.value} = {identity_literal}
      AND name IN ({metric_names})
      AND timestamp_ms >= {scan_start_ms} AND timestamp_ms < {end_ms}
    UNION ALL
    SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
           service, name, kind, value, body_json, resource_attributes_json, attributes_json, timestamp_ms, seq
    FROM "telemetry_v1.marinskyrl"
    WHERE service = 'marinskyrl'
      AND json_get(attributes_json, 'metric_source') = 'vllm'
      AND {identity_field.value} = {identity_literal}
      AND name IN ({metric_names})
      AND timestamp_ms >= {scan_start_ms} AND timestamp_ms < {end_ms}
), bounded_samples AS (
    SELECT * FROM base LIMIT {VLLM_MAX_SAMPLES + 1}
), normalized AS (
    SELECT origin_cluster, service, name, kind, value, body_json, resource_attributes_json,
           CASE WHEN json_get(attributes_json, 'histogram_publication_id') IS NOT NULL
                THEN {attributes_without_token}
                ELSE attributes_json END AS attributes_json,
           json_get(attributes_json, 'histogram_publication_id') AS publication_id,
           timestamp_ms, seq
    FROM bounded_samples
)
SELECT origin_cluster, service, name, kind, resource_attributes_json, attributes_json,
       array_agg(named_struct('timestamp_ms', timestamp_ms, 'seq', seq, 'value', value,
                              {histogram_fields})) AS points
FROM normalized
GROUP BY 1, 2, 3, 4, 5, 6
LIMIT {VLLM_MAX_SERIES + 1}
""".strip()


def _case_for(mapping: tuple[tuple[str, str], ...], expression: str) -> str:
    cases = " ".join(f"WHEN {sql_string(source)} THEN {sql_string(target)}" for source, target in mapping)
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


def _histogram_base_mapping() -> tuple[tuple[str, str], ...]:
    return _HISTOGRAM_FAMILIES


def vllm_overview_query(
    identity_field: VllmIdentityField,
    identity: str,
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
) -> VllmOverviewQuery:
    """Render the fixed vLLM overview query after validating its safety bounds."""
    validate_value("identity", identity, max_length=VLLM_MAX_IDENTITY_LENGTH)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=VLLM_MAX_WINDOW_MS,
        max_window_error="vLLM overview range must not exceed 7 days",
        min_bucket_ms=VLLM_MIN_BUCKET_MS,
        max_points=VLLM_MAX_POINTS,
    )
    standalone_bucket_ms = max(bucket_ms, VLLM_SCRAPE_INTERVAL_MS)
    scan_start_ms = max(0, start_ms - VLLM_SNAPSHOT_LOOKBACK_MS)
    serving_metric_names = sql_values(_SERVING_METRIC_NAMES)
    token_counters = sql_values(_TOKEN_COUNTERS)
    preemption_counters = sql_values(_PREEMPTION_COUNTERS)
    outcome_counters = sql_values(_OUTCOME_COUNTERS)
    gauges = sql_values(_GAUGES)
    histogram_names = sql_values(_HISTOGRAM_NAMES)
    histogram_base_names = sql_values(_HISTOGRAM_BASE_NAMES)
    histogram_family = _case_for(_histogram_name_mapping(), "samples.name")
    histogram_component = _case_for(_histogram_component_mapping(), "samples.name")
    histogram_source_family = _case_for(_histogram_source_mapping(), "samples.name")
    histogram_base_family = _case_for(_histogram_base_mapping(), "name")

    samples_sql = _vllm_samples_query(identity_field, identity, scan_start_ms, end_ms, _METRIC_NAMES)
    # Retain the whole leading coherence interval and one predecessor per
    # series. Older points can affect neither an in-window delta nor freshness.
    coherence_start_ms = start_ms - start_ms % VLLM_HISTOGRAM_COHERENCE_MS
    sql = f"""
WITH base AS MATERIALIZED (
    SELECT origin_cluster, service, name, kind, resource_attributes_json, attributes_json,
           point.timestamp_ms AS timestamp_ms, point.seq AS seq, point.value AS value,
           point.body_json AS body_json, point.publication_id AS publication_id
    FROM (
        SELECT * EXCLUDE (points), unnest(list_concat(
            list_slice(list_filter(points, p -> p.timestamp_ms < {coherence_start_ms}), -1, -1),
            list_filter(points, p -> p.timestamp_ms >= {coherence_start_ms})
        )) AS point
        FROM (SELECT * REPLACE (
            list_sort(points) AS points,
            CAST(resource_attributes_json AS resource_labels) AS resource_attributes_json,
            CAST(attributes_json AS metric_labels) AS attributes_json
        ) FROM series)
    )
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
      -- Reject legacy mixed-engine histograms before computing their unused deltas.
      AND name NOT IN ({histogram_names}, {histogram_base_names})
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
    SELECT {start_ms} + (timestamp_ms - {start_ms})
               - (timestamp_ms - {start_ms}) % CASE
                   WHEN service = 'vllm' THEN {standalone_bucket_ms}
                   ELSE {bucket_ms}
               END AS t,
           name,
           origin_cluster,
           service,
           resource_attributes_json,
           COALESCE(json_get(attributes_json, 'engine'), resource_attributes_json) AS producer_identity,
           SUM(delta) AS delta
    FROM increments
    WHERE name IN ({token_counters})
      AND delta IS NOT NULL
    GROUP BY 1, 2, 3, 4, 5, 6
), token_source_rates AS (
    SELECT t,
           name,
           origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           delta / (CASE WHEN service = 'vllm' THEN {standalone_bucket_ms} ELSE {bucket_ms} END / 1000.0) AS value
    FROM token_bins
), token_rates AS (
    SELECT t, name, SUM(value) AS value
    FROM token_source_rates
    GROUP BY 1, 2
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
           COALESCE(json_get(attributes_json, 'engine'), attributes_json) AS producer_identity,
           COALESCE(json_get(attributes_json, 'engine'), resource_attributes_json) AS engine_identity,
           value
    FROM base
    WHERE timestamp_ms >= {start_ms}
      AND name IN ({gauges})
      AND json_get(attributes_json, 'source_temporality') = 'current_snapshot'
), gauge_replica_bins AS (
    SELECT {start_ms} + (timestamp_ms - {start_ms})
               - (timestamp_ms - {start_ms}) % CASE
                   WHEN service = 'vllm' THEN {standalone_bucket_ms}
                   ELSE {bucket_ms}
               END AS t,
           name,
           origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
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
), request_in_flight_bins AS (
    SELECT t, SUM(value) AS value
    FROM canonical_gauge_bins
    WHERE name IN ('num_requests_running', 'num_requests_waiting')
    GROUP BY 1
    HAVING COUNT(*) = 2
), raw_gauge_peaks AS (
    SELECT name, MAX(value) AS peak
    FROM canonical_gauge_samples
    GROUP BY 1
), kv_peak_bins AS (
    SELECT {start_ms} + (timestamp_ms - {start_ms})
               - (timestamp_ms - {start_ms}) % CASE
                   WHEN service = 'vllm' THEN {standalone_bucket_ms}
                   ELSE {bucket_ms}
               END AS t,
           MAX(value) AS value
    FROM canonical_gauge_samples
    WHERE name = 'kv_cache_usage'
    GROUP BY 1
), engine_stats AS (
    SELECT engine_identity || ' @ ' || origin_cluster || ':' || service || ':' || resource_attributes_json AS series,
           CASE name
               WHEN 'num_requests_running' THEN 'running_mean'
               WHEN 'num_requests_waiting' THEN 'waiting_mean'
               ELSE 'kv_cache_peak'
           END AS metric,
           CASE WHEN name = 'kv_cache_usage' THEN MAX(value) ELSE AVG(value) END AS value,
           CASE WHEN name = 'kv_cache_usage' THEN 'ratio' ELSE 'requests' END AS unit,
           COUNT(*) AS samples
    FROM canonical_gauge_samples
    GROUP BY 1, 2, 4, name

    UNION ALL

    SELECT producer_identity || ' @ ' || origin_cluster || ':' || service || ':' || resource_attributes_json AS series,
           'generated_tokens_per_second' AS metric,
           AVG(value) AS value,
           'tokens/s' AS unit,
           COUNT(*) AS samples
    FROM token_source_rates
    WHERE name = 'generation_tokens_total'
    GROUP BY 1
), ranked_engine_stats AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY metric ORDER BY series) AS producer_rank
    FROM engine_stats
), gauge_stats AS (
    SELECT bins.name,
           AVG(bins.value) AS average,
           CASE WHEN bins.name = 'kv_cache_usage' THEN raw.peak ELSE MAX(bins.value) END AS peak
    FROM canonical_gauge_bins AS bins
    JOIN raw_gauge_peaks AS raw USING (name)
    GROUP BY 1, raw.peak
), histogram_structured_samples AS MATERIALIZED (
    SELECT origin_cluster, service, resource_attributes_json, attributes_json,
           name AS source_family, {histogram_base_family} AS family,
           timestamp_ms, seq,
           CAST(json_extract(body_json, '$.explicit_bounds') AS DOUBLE[]) AS bounds,
           CAST(json_extract(body_json, '$.bucket_counts') AS BIGINT[]) AS bins,
           CAST(json_extract(body_json, '$.count') AS BIGINT) AS count,
           CAST(json_extract(body_json, '$.sum') AS DOUBLE) AS total,
           json_get(body_json, 'producer_epoch') AS producer_epoch,
           CAST(json_extract(body_json, '$.sequence') AS BIGINT) AS source_sequence,
           CAST(json_extract(body_json, '$.explicit_bounds') AS VARCHAR) AS schema_key,
           json_get(body_json, 'producer_epoch') || ':' || json_get(body_json, 'sequence') AS publication_id
    FROM base
    WHERE name IN ({histogram_base_names})
      AND kind = 'histogram' AND body_json IS NOT NULL
      AND (service = 'vllm' OR json_get(attributes_json, 'engine_index') IS NOT NULL)
    WINDOW publication AS (
        PARTITION BY origin_cluster, service, name, resource_attributes_json, attributes_json,
                     json_get(body_json, 'producer_epoch'), json_get(body_json, 'sequence')
    )
    QUALIFY MIN(body_json) OVER publication = MAX(body_json) OVER publication
        AND MIN(timestamp_ms) OVER publication = MAX(timestamp_ms) OVER publication
        AND ROW_NUMBER() OVER (publication ORDER BY seq DESC) = 1
), histogram_scalar_samples AS (
    SELECT samples.origin_cluster, samples.service, samples.resource_attributes_json,
           CAST(json_merge_patch(CAST(CAST(samples.attributes_json AS VARCHAR) AS JSON), '{{"le":null}}') AS VARCHAR)
               AS attributes_json,
           {histogram_source_family} AS source_family,
           {histogram_family} AS family,
           {histogram_component} AS component,
           json_get(samples.attributes_json, 'le') AS upper_bound,
           samples.timestamp_ms, samples.seq, samples.value,
           CASE WHEN samples.name LIKE '%_sum' THEN NULL
                WHEN samples.value >= 0 AND samples.value < 9007199254740992
                     AND samples.value = FLOOR(samples.value)
                THEN CAST(samples.value AS BIGINT) ELSE NULL END AS integer_value,
           CAST(NULL AS VARCHAR) AS producer_epoch,
           CAST(NULL AS BIGINT) AS source_sequence,
           CAST(NULL AS VARCHAR) AS schema_key,
           0 AS is_structured
    FROM base AS samples
    WHERE samples.name IN ({histogram_names})
      AND json_get(samples.attributes_json, 'source_temporality') = 'cumulative_snapshot'
      AND (samples.service = 'vllm' OR json_get(samples.attributes_json, 'engine_index') IS NOT NULL)
      AND NOT EXISTS (
          SELECT 1 FROM histogram_structured_samples AS structured
          WHERE samples.publication_id IS NOT NULL
            AND structured.publication_id = samples.publication_id
            AND structured.source_family = {histogram_source_family}
            AND structured.origin_cluster = samples.origin_cluster
            AND structured.service = samples.service
            AND structured.resource_attributes_json = samples.resource_attributes_json
            AND structured.attributes_json = CAST(json_merge_patch(
                CAST(CAST(samples.attributes_json AS VARCHAR) AS JSON), '{{"le":null}}') AS VARCHAR)
      )
), histogram_structured_components AS (
    SELECT origin_cluster, service, resource_attributes_json, attributes_json,
           source_family, family, 'bucket' AS component,
           CASE WHEN bucket.i <= array_length(bounds) THEN CAST(bounds[bucket.i] AS VARCHAR)
                ELSE '+Inf' END AS upper_bound,
           timestamp_ms, seq, CAST(list_sum(list_slice(bins, 1, bucket.i)) AS DOUBLE) AS value,
           list_sum(list_slice(bins, 1, bucket.i)) AS integer_value,
           producer_epoch, source_sequence, schema_key, 1 AS is_structured
    FROM histogram_structured_samples,
         unnest(range(1, array_length(bounds) + 2)) AS bucket(i)

    UNION ALL

    SELECT origin_cluster, service, resource_attributes_json, attributes_json,
           source_family, family, 'count' AS component, CAST(NULL AS VARCHAR) AS upper_bound,
           timestamp_ms, seq, CAST(count AS DOUBLE) AS value, count AS integer_value,
           producer_epoch, source_sequence, schema_key, 1 AS is_structured
    FROM histogram_structured_samples

    UNION ALL

    SELECT origin_cluster, service, resource_attributes_json, attributes_json,
           source_family, family, 'sum' AS component, CAST(NULL AS VARCHAR) AS upper_bound,
           timestamp_ms, seq, total AS value, CAST(NULL AS BIGINT) AS integer_value,
           producer_epoch, source_sequence, schema_key, 1 AS is_structured
    FROM histogram_structured_samples
), histogram_ordered AS (
    SELECT *,
           LAG(value) OVER hist_order AS previous_value,
           LAG(integer_value) OVER hist_order AS previous_integer_value,
           LAG(producer_epoch) OVER hist_order AS previous_producer_epoch,
           LAG(schema_key) OVER hist_order AS previous_schema_key
    FROM (
        SELECT * FROM histogram_scalar_samples
        UNION ALL
        SELECT * FROM histogram_structured_components
    )
    WINDOW hist_order AS (
        PARTITION BY origin_cluster, service, resource_attributes_json, attributes_json,
                     source_family, component, upper_bound
        ORDER BY timestamp_ms, COALESCE(source_sequence, seq), seq
    )
), histogram_delta_samples AS (
    SELECT *,
           CASE
               WHEN previous_value IS NULL
                 OR (producer_epoch IS NOT NULL AND previous_producer_epoch IS NOT NULL
                     AND producer_epoch <> previous_producer_epoch)
                 OR (schema_key IS NOT NULL AND previous_schema_key IS NOT NULL
                     AND schema_key <> previous_schema_key)
               THEN NULL
               WHEN integer_value IS NOT NULL AND previous_integer_value IS NOT NULL
               THEN CASE WHEN integer_value < previous_integer_value THEN NULL
                         ELSE CAST(integer_value - previous_integer_value AS DOUBLE) END
               WHEN value < previous_value THEN NULL
               ELSE value - previous_value
           END AS delta
    FROM histogram_ordered
), histogram_component_samples AS (
    SELECT origin_cluster, service, resource_attributes_json, attributes_json,
           COALESCE(json_get(attributes_json, 'engine'), resource_attributes_json) AS producer_identity,
           timestamp_ms, timestamp_ms - timestamp_ms % {VLLM_HISTOGRAM_COHERENCE_MS} AS sample_t,
           source_family, family, component, upper_bound, schema_key, is_structured, delta,
           CASE WHEN delta IS NULL THEN 1 ELSE 0 END AS invalid_component
    FROM histogram_delta_samples
), histogram_series AS (
    SELECT DISTINCT origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           source_family,
           component,
           upper_bound,
           attributes_json
    FROM histogram_component_samples
), histogram_expected_series AS (
    SELECT origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           source_family,
           COUNT(*) AS expected_series
    FROM histogram_series
    GROUP BY 1, 2, 3, 4, 5
), histogram_sample_validity AS (
    SELECT samples.origin_cluster,
           samples.service,
           samples.resource_attributes_json,
           samples.producer_identity,
           samples.source_family,
           samples.sample_t,
           CASE
               WHEN MAX(samples.invalid_component) = 1 THEN 0
               -- One validated body is already a complete family, even if an older
               -- publication used a different set of explicit bounds.
               WHEN MIN(samples.is_structured) = 1 THEN 1
               WHEN COUNT(*) < MAX(expected.expected_series) THEN 0
               ELSE 1
           END AS valid_sample
    FROM histogram_component_samples AS samples
    JOIN histogram_expected_series AS expected
      ON samples.origin_cluster = expected.origin_cluster
     AND samples.service = expected.service
     AND samples.resource_attributes_json = expected.resource_attributes_json
     AND samples.producer_identity = expected.producer_identity
     AND samples.source_family = expected.source_family
    WHERE samples.timestamp_ms >= {start_ms}
    GROUP BY 1, 2, 3, 4, 5, 6
), coherent_histogram_increments AS (
    SELECT samples.sample_t,
           samples.origin_cluster,
           samples.service,
           samples.resource_attributes_json,
           samples.producer_identity,
           samples.family,
           samples.component,
           samples.upper_bound,
           samples.schema_key,
           samples.delta
    FROM histogram_component_samples AS samples
    JOIN histogram_sample_validity AS validity
      ON samples.origin_cluster = validity.origin_cluster
     AND samples.service = validity.service
     AND samples.resource_attributes_json = validity.resource_attributes_json
     AND samples.producer_identity = validity.producer_identity
     AND samples.source_family = validity.source_family
     AND samples.sample_t = validity.sample_t
    WHERE validity.valid_sample = 1
), engine_itl AS (
    SELECT producer_identity || ' @ ' || origin_cluster || ':' || service || ':' || resource_attributes_json AS series,
           SUM(CASE WHEN component = 'sum' THEN delta ELSE 0 END)
               / NULLIF(SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END), 0) AS value,
           SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) AS samples
    FROM coherent_histogram_increments
    WHERE family = 'inter_token_latency'
    GROUP BY 1
    HAVING SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) > 0
), ranked_engine_itl AS (
    SELECT *, ROW_NUMBER() OVER (ORDER BY series) AS producer_rank
    FROM engine_itl
), histogram_time_means AS (
    SELECT {start_ms} + (sample_t - {start_ms}) - (sample_t - {start_ms}) % {bucket_ms} AS t,
           family,
           SUM(CASE WHEN component = 'sum' THEN delta ELSE 0 END)
               / NULLIF(SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END), 0) AS mean,
           SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) AS samples
    FROM coherent_histogram_increments
    GROUP BY 1, 2
    HAVING SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) > 0
), histogram_means AS (
    SELECT family,
           SUM(CASE WHEN component = 'sum' THEN delta ELSE 0 END)
               / NULLIF(SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END), 0) AS mean,
           SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) AS samples
    FROM coherent_histogram_increments
    GROUP BY 1
    HAVING SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) > 0
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
), histogram_schema_counts AS (
    SELECT family, COUNT(DISTINCT schema_key) AS versions
    FROM coherent_histogram_increments
    WHERE schema_key IS NOT NULL
    GROUP BY family
), output_length_distribution AS (
    SELECT upper_bound,
           bucket_count - COALESCE(LAG(bucket_count) OVER (
               ORDER BY {_HISTOGRAM_BOUND_ORDER_SQL}
           ), 0) AS value,
           total_count
    FROM histogram_ranked_buckets
    LEFT JOIN histogram_schema_counts AS schemas USING (family)
    WHERE family = 'output_tokens'
      AND COALESCE(schemas.versions, 0) <= 1
), histogram_quantiles AS (
    SELECT buckets.family,
           CASE WHEN COALESCE(schemas.versions, 0) > 1 THEN NULL ELSE MIN(CASE
               WHEN upper_bound NOT IN ('+Inf', 'Inf') AND bucket_count >= total_count * 0.50
               THEN CAST(upper_bound AS DOUBLE)
           END) END AS p50,
           CASE WHEN COALESCE(schemas.versions, 0) > 1 THEN NULL ELSE MIN(CASE
               WHEN upper_bound NOT IN ('+Inf', 'Inf') AND bucket_count >= total_count * 0.90
               THEN CAST(upper_bound AS DOUBLE)
           END) END AS p90,
           CASE WHEN COALESCE(schemas.versions, 0) > 1 THEN NULL ELSE MIN(CASE
               WHEN upper_bound NOT IN ('+Inf', 'Inf') AND bucket_count >= total_count * 0.99
               THEN CAST(upper_bound AS DOUBLE)
           END) END AS p99
    FROM histogram_ranked_buckets AS buckets
    LEFT JOIN histogram_schema_counts AS schemas USING (family)
    GROUP BY buckets.family, schemas.versions
), histogram_stats AS (
    SELECT means.family, means.mean, means.samples, quantiles.p50, quantiles.p90, quantiles.p99
    FROM histogram_means AS means
    LEFT JOIN histogram_quantiles AS quantiles USING (family)
), histogram_evidence AS (
    SELECT family,
           quantile.stat,
           CASE quantile.stat
               WHEN 'mean' THEN mean
               WHEN 'p50' THEN p50
               WHEN 'p90' THEN p90
               ELSE p99
           END AS value,
           samples
    FROM histogram_stats
    -- Share the histogram pipeline across statistics.
    CROSS JOIN (VALUES ('mean'), ('p50'), ('p90'), ('p99')) AS quantile(stat)
), outcome_increments AS (
    SELECT timestamp_ms,
           service,
           name,
           COALESCE(
               json_get(attributes_json, 'finished_reason'),
               json_get(attributes_json, 'finish_reason'),
               json_get(attributes_json, 'outcome'),
               json_get(attributes_json, 'status'),
               name
           ) AS outcome,
           delta
    FROM increments
    WHERE name IN ({outcome_counters})
      AND delta IS NOT NULL
), outcome_totals AS (
    SELECT outcome, SUM(delta) AS value
    FROM outcome_increments
    GROUP BY 1
), length_finish_fraction AS (
    SELECT SUM(CASE WHEN outcome = 'length' THEN delta ELSE 0 END) / NULLIF(SUM(delta), 0) AS value,
           SUM(delta) AS samples
    FROM outcome_increments
    WHERE name = 'request_success_total'
    HAVING COUNT(delta) > 0
), outcome_source_rates AS (
    SELECT {start_ms} + (timestamp_ms - {start_ms})
               - (timestamp_ms - {start_ms}) % CASE
                   WHEN service = 'vllm' THEN {standalone_bucket_ms}
                   ELSE {bucket_ms}
               END AS t,
           outcome,
           service,
           SUM(delta) / (CASE WHEN service = 'vllm' THEN {standalone_bucket_ms} ELSE {bucket_ms} END / 1000.0) AS value
    FROM outcome_increments
    GROUP BY 1, 2, 3
), outcome_rates AS (
    SELECT t, outcome, SUM(value) AS value
    FROM outcome_source_rates
    GROUP BY 1, 2
), collector_polls AS (
    SELECT COUNT(*) AS polls,
           SUM(CASE WHEN value <= 0 THEN 1 ELSE 0 END) AS unavailable_polls,
           MAX(timestamp_ms) AS latest_timestamp_ms
    FROM base
    WHERE timestamp_ms >= {start_ms}
      AND name = 'prometheus_source_available'
      AND json_get(attributes_json, 'metric_source') = 'vllm'
), publication_health_ranked AS (
    SELECT json_get(attributes_json, 'drop_reason') AS drop_reason,
           value,
           timestamp_ms,
           COUNT(*) OVER (PARTITION BY json_get(attributes_json, 'drop_reason')) AS samples,
           ROW_NUMBER() OVER (
               PARTITION BY json_get(attributes_json, 'drop_reason')
               ORDER BY timestamp_ms DESC, seq DESC
           ) AS recency
    FROM base
    WHERE timestamp_ms >= {start_ms}
      AND service = 'marinskyrl'
      AND name = 'metric_publication_dropped_records'
      AND json_get(attributes_json, 'metric_source') = 'vllm'
      AND json_get(attributes_json, 'drop_reason') IN ('sample_limit', 'telemetry_loss')
), publication_health_reasons AS (
    SELECT drop_reason,
           samples,
           timestamp_ms AS latest_timestamp_ms,
           value AS current_value,
           CASE WHEN value > 0 THEN 1 ELSE 0 END AS positive
    FROM publication_health_ranked
    WHERE recency = 1
), publication_health AS (
    SELECT COUNT(*) AS reasons,
           COALESCE(MIN(samples), 0) AS polls,
           MIN(latest_timestamp_ms) AS oldest_latest_timestamp_ms,
           COALESCE(MAX(positive), 0) AS has_positive,
           COALESCE(MAX(CASE
               WHEN positive > 0
                AND {end_ms} - latest_timestamp_ms <= {VLLM_FRESHNESS_THRESHOLD_MS}
               THEN 1 ELSE 0
           END), 0) AS has_fresh_positive
    FROM publication_health_reasons
), collection_failure_increments AS (
    SELECT COALESCE(json_get(attributes_json, 'stage'), 'unknown') AS stage,
           CASE
               WHEN previous_value IS NULL THEN NULL
               WHEN value < previous_value THEN value
               ELSE value - previous_value
           END AS delta,
           CASE WHEN previous_value IS NULL AND value > 0 THEN 1 ELSE 0 END AS uncertain
    FROM cumulative_samples
    WHERE timestamp_ms >= {start_ms}
      AND name = 'prometheus_stage_failures'
      AND json_get(attributes_json, 'metric_source') = 'vllm'
), collection_failure_totals AS (
    SELECT stage, SUM(delta) AS value
    FROM collection_failure_increments
    WHERE delta > 0
    GROUP BY 1
), dropped_sample_totals AS (
    SELECT COALESCE(json_get(attributes_json, 'drop_reason'), 'unknown') AS drop_reason,
           SUM(value) AS value
    FROM base
    WHERE timestamp_ms >= {start_ms}
      AND name = 'prometheus_dropped_samples'
      AND json_get(attributes_json, 'metric_source') = 'vllm'
      AND value > 0
    GROUP BY 1

    UNION ALL

    SELECT drop_reason, current_value
    FROM publication_health_reasons
    WHERE current_value > 0
), telemetry_health AS (
    SELECT CASE WHEN collector.polls > 0 THEN collector.polls ELSE publication.polls END AS polls,
           COALESCE(unavailable_polls, 0) AS unavailable_polls,
           CASE
               WHEN collector.polls > 0
                AND {end_ms} - collector.latest_timestamp_ms > {VLLM_FRESHNESS_THRESHOLD_MS}
               THEN 'unknown'
               WHEN collector.polls = 0 AND publication.has_fresh_positive > 0
               THEN 'incomplete'
               WHEN collector.polls = 0
                AND (
                    publication.reasons < 2
                    OR {end_ms} - publication.oldest_latest_timestamp_ms > {VLLM_FRESHNESS_THRESHOLD_MS}
                )
               THEN 'unknown'
               WHEN COALESCE(unavailable_polls, 0) > 0
                 OR failure_stages > 0
                 OR uncertain_failures > 0
                 OR drop_reasons > 0
                 OR publication.has_positive > 0
               THEN 'incomplete'
               WHEN collector.polls > 0 OR publication.reasons = 2 THEN 'healthy'
               ELSE 'unknown'
           END AS status
    FROM collector_polls AS collector
    CROSS JOIN publication_health AS publication
    CROSS JOIN (SELECT COUNT(*) AS failure_stages FROM collection_failure_totals)
    CROSS JOIN (SELECT COUNT(*) AS uncertain_failures FROM collection_failure_increments WHERE uncertain > 0)
    CROSS JOIN (SELECT COUNT(*) AS drop_reasons FROM dropped_sample_totals)
), producer_samples AS (
    SELECT DISTINCT origin_cluster,
           service,
           resource_attributes_json,
           CASE
               WHEN service = 'marinskyrl' THEN COALESCE(json_get(attributes_json, 'engine'), resource_attributes_json)
               ELSE resource_attributes_json
           END AS producer_identity,
           timestamp_ms
    FROM base
    WHERE name IN ({serving_metric_names})
), producer_ordered AS (
    SELECT *,
           LAG(timestamp_ms) OVER (
               PARTITION BY origin_cluster, service, resource_attributes_json, producer_identity
               ORDER BY timestamp_ms
           ) AS previous_timestamp_ms
    FROM producer_samples
), producer_freshness_data AS (
    SELECT origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           MAX(timestamp_ms) AS latest_timestamp_ms,
           SUM(CASE WHEN timestamp_ms >= {start_ms} THEN 1 ELSE 0 END) AS samples,
           MAX(CASE
               WHEN timestamp_ms >= {start_ms} THEN timestamp_ms - previous_timestamp_ms
           END) / 1000.0 AS gap_seconds
    FROM producer_ordered
    GROUP BY 1, 2, 3, 4
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
                        resource_attributes_json,
                        producer_identity
           ) AS freshness_rank
    FROM producer_freshness
), freshness_summary AS (
    SELECT * FROM ranked_freshness WHERE freshness_rank = 1
), output AS (
    SELECT t AS t,
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
           value AS value,
           'tokens/s' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM token_rates

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'counter_total' AS section,
           CASE name
               WHEN 'prompt_tokens_total' THEN 'prompt_tokens'
               WHEN 'generation_tokens_total' THEN 'generated_tokens'
               ELSE 'preemptions'
           END AS metric,
           'total' AS stat,
           CASE name
               WHEN 'prompt_tokens_total' THEN 'prompt tokens'
               WHEN 'generation_tokens_total' THEN 'generated tokens'
               ELSE 'preemptions'
           END AS series,
           value AS value,
           CASE WHEN name IN ({token_counters}) THEN 'tokens' ELSE 'requests' END AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM counter_totals

    UNION ALL

    SELECT t AS t,
           'saturation' AS section,
           name AS metric,
           'value' AS stat,
           name AS series,
           value AS value,
           CASE WHEN name = 'kv_cache_usage' THEN 'ratio' ELSE 'requests' END AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM canonical_gauge_bins

    UNION ALL

    SELECT t AS t,
           'saturation' AS section,
           'num_requests_in_flight' AS metric,
           'value' AS stat,
           'num_requests_in_flight' AS series,
           value AS value,
           'requests' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM request_in_flight_bins

    UNION ALL

    SELECT t AS t,
           'saturation' AS section,
           'kv_cache_usage_peak' AS metric,
           'value' AS stat,
           'kv_cache_usage_peak' AS series,
           value AS value,
           'ratio' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM kv_peak_bins

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'engine_summary' AS section,
           metric AS metric,
           'observed' AS stat,
           series AS series,
           value AS value,
           unit AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(samples AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM ranked_engine_stats
    WHERE producer_rank <= {VLLM_MAX_FRESHNESS_DETAILS}

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'engine_summary' AS section,
           'inter_token_latency_mean' AS metric,
           'observed' AS stat,
           series AS series,
           value AS value,
           's' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(samples AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM ranked_engine_itl
    WHERE producer_rank <= {VLLM_MAX_FRESHNESS_DETAILS}

    UNION ALL

    SELECT t AS t,
           'saturation' AS section,
           'iteration_tokens' AS metric,
           'mean' AS stat,
           'iteration tokens per engine step' AS series,
           mean AS value,
           'tokens' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM histogram_time_means
    WHERE family = 'iteration_tokens'

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'saturation_summary' AS section,
           name AS metric,
           'average' AS stat,
           name AS series,
           average AS value,
           CASE WHEN name = 'kv_cache_usage' THEN 'ratio' ELSE 'requests' END AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM gauge_stats

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'saturation_summary' AS section,
           name AS metric,
           'peak' AS stat,
           name AS series,
           peak AS value,
           CASE WHEN name = 'kv_cache_usage' THEN 'ratio' ELSE 'requests' END AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM gauge_stats

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           CASE WHEN family = 'output_tokens' THEN 'workload' ELSE 'latency' END AS section,
           family AS metric,
           stat AS stat,
           family AS series,
           value AS value,
           CASE WHEN family = 'output_tokens' THEN 'tokens' ELSE 's' END AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(samples AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM histogram_evidence
    WHERE family <> 'iteration_tokens'

    UNION ALL

    SELECT t AS t,
           'latency' AS section,
           family AS metric,
           'mean_over_time' AS stat,
           CASE WHEN family = 'tpot' THEN 'time per output token' ELSE family END AS series,
           mean AS value,
           's' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(samples AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM histogram_time_means
    WHERE family NOT IN ('iteration_tokens', 'output_tokens')

    UNION ALL

    SELECT t AS t,
           'request_rate' AS section,
           'requests' AS metric,
           'rate' AS stat,
           outcome AS series,
           value AS value,
           'requests/s' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM outcome_rates

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'request_outcome' AS section,
           'requests' AS metric,
           'total' AS stat,
           outcome AS series,
           value AS value,
           'requests' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM outcome_totals

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'length_finish_fraction' AS section,
           'length_finish_fraction' AS metric,
           'fraction' AS stat,
           'length / all engine finishes' AS series,
           value AS value,
           'ratio' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(samples AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM length_finish_fraction

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'output_length_distribution' AS section,
           'output_tokens' AS metric,
           'interval_count' AS stat,
           upper_bound AS series,
           value AS value,
           'requests' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(total_count AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM output_length_distribution

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'telemetry_health' AS section,
           'collector' AS metric,
           'polls' AS stat,
           'all resources' AS series,
           CAST(polls AS DOUBLE) AS value,
           'polls' AS unit,
           status AS status,
           CAST(polls AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM telemetry_health

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'telemetry_health' AS section,
           'source availability' AS metric,
           'unavailable polls' AS stat,
           'all resources' AS series,
           CAST(unavailable_polls AS DOUBLE) AS value,
           'polls' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM telemetry_health
    WHERE unavailable_polls > 0

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'telemetry_health' AS section,
           'collection failures' AS metric,
           'delta' AS stat,
           stage AS series,
           value AS value,
           'failures' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM collection_failure_totals

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'telemetry_health' AS section,
           'dropped samples' AS metric,
           'total' AS stat,
           drop_reason AS series,
           value AS value,
           'samples' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(NULL AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM dropped_sample_totals

    UNION ALL

    SELECT latest_timestamp_ms AS t,
           'freshness' AS section,
           'telemetry' AS metric,
           'latest_sample_age' AS stat,
           origin_cluster || ':' || service || ':' || producer_identity AS series,
           ({end_ms} - latest_timestamp_ms) / 1000.0 AS value,
           's' AS unit,
           freshness_status AS status,
           CAST(samples AS BIGINT) AS samples,
           gap_seconds AS gap_seconds
    FROM freshness_summary

    UNION ALL

    SELECT latest_timestamp_ms AS t,
           'freshness_detail' AS section,
           'telemetry' AS metric,
           'latest_sample_age' AS stat,
           origin_cluster || ':' || service || ':' || producer_identity AS series,
           ({end_ms} - latest_timestamp_ms) / 1000.0 AS value,
           's' AS unit,
           freshness_status AS status,
           CAST(samples AS BIGINT) AS samples,
           gap_seconds AS gap_seconds
    FROM ranked_freshness
    WHERE freshness_rank <= {VLLM_MAX_FRESHNESS_DETAILS}

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'freshness' AS section,
           'telemetry' AS metric,
           'latest_sample_age' AS stat,
           'telemetry' AS series,
           CAST(NULL AS DOUBLE) AS value,
           's' AS unit,
           'no_data' AS status,
           CAST(0 AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    WHERE NOT EXISTS (SELECT 1 FROM producer_freshness)
)
SELECT t, section, metric, stat, series, value, unit, status, samples, gap_seconds
FROM output
ORDER BY section,
         CASE WHEN section = 'telemetry_health' AND metric = 'collector' THEN 0 ELSE 1 END,
         t,
         metric,
         stat,
         CASE WHEN section = 'output_length_distribution'
              THEN {_HISTOGRAM_BOUND_ORDER_SQL.replace('upper_bound', 'series')}
              ELSE 0 END,
         series
LIMIT {VLLM_MAX_RESULT_ROWS + 1}
""".strip()
    return VllmOverviewQuery(
        sql=sql,
        samples_sql=samples_sql,
        identity_field=identity_field,
        identity=identity,
        start_ms=start_ms,
        end_ms=end_ms,
        bucket_ms=bucket_ms,
    )


def _vllm_project_table(
    sql: str, series: pa.Table, projection_lock: AbstractContextManager[None], *, max_rows: int
) -> pa.Table:
    """Project one compact Finelog scan into the shared diagnostic result."""
    validate_table_budget("vLLM samples", series, max_rows=VLLM_MAX_SERIES, max_samples=VLLM_MAX_SAMPLES)
    with projection_lock, projection_database() as database:
        database.register("series", series)
        # Dictionary labels stay compact through repeated window and histogram
        # joins. Ordered dictionaries preserve the original string tie breaks.
        for column, label_type in (
            ("resource_attributes_json", "resource_labels"),
            ("attributes_json", "metric_labels"),
        ):
            database.execute(
                f"CREATE TYPE {label_type} AS ENUM (SELECT DISTINCT {column} FROM series UNION SELECT '' ORDER BY 1)"
            )
        database.execute("CREATE MACRO json_get(d, f) AS json_extract_string(CAST(d AS VARCHAR), concat('$.', f))")
        try:
            table = database.execute(sql).to_arrow_table()
        except duckdb.OutOfMemoryException as err:
            raise QueryResultTooLargeError("vLLM projection memory budget exceeded") from err
    if table.num_rows > max_rows:
        raise QueryResultTooLargeError(f"vLLM diagnostic returned more than {max_rows} rows")
    return table


def vllm_overview_table(
    overview: VllmOverviewQuery,
    series: pa.Table,
    projection_lock: AbstractContextManager[None],
    *,
    max_rows: int,
) -> pa.Table:
    """Project one compact Finelog scan into the shared diagnostic result."""
    return _vllm_project_table(overview.sql, series, projection_lock, max_rows=max_rows)


def vllm_run_summary_samples_query(overview: VllmOverviewQuery) -> str:
    """Keep the detail scan bounds and caps while selecting fewer metric names."""
    return _vllm_samples_query(
        overview.identity_field,
        overview.identity,
        max(0, overview.start_ms - VLLM_SNAPSHOT_LOOKBACK_MS),
        overview.end_ms,
        _SUMMARY_METRIC_NAMES,
        compact_histograms=True,
    )


def vllm_run_summary_query(overview: VllmOverviewQuery) -> str:
    """Render reset-aware local summary SQL over the compact Finelog result."""
    return f"""
WITH base AS MATERIALIZED (
    SELECT origin_cluster, service, name, kind,
           CAST(resource_attributes_json AS resource_labels) AS resource_attributes_json,
           CAST(attributes_json AS metric_labels) AS attributes_json,
           point.timestamp_ms AS timestamp_ms, point.seq AS seq, point.value AS value,
           point.publication_id AS publication_id,
           point.histogram_count AS histogram_count,
           point.histogram_sum AS histogram_sum,
           point.histogram_bounds AS histogram_bounds,
           point.producer_epoch AS producer_epoch,
           point.source_sequence AS source_sequence
    FROM (SELECT * EXCLUDE (points), unnest(points) AS point FROM series)
), structured_histograms AS MATERIALIZED (
    SELECT origin_cluster, service, name, resource_attributes_json, attributes_json,
           timestamp_ms, seq,
           histogram_count AS count, histogram_sum AS total,
           producer_epoch, source_sequence, histogram_bounds AS schema_key,
           producer_epoch || ':' || CAST(source_sequence AS VARCHAR) AS publication_id
    FROM base
    WHERE name IN ('time_to_first_token_seconds', 'inter_token_latency_seconds')
      AND kind = 'histogram' AND histogram_count IS NOT NULL
      AND (service = 'vllm' OR json_get(attributes_json, 'engine_index') IS NOT NULL)
    WINDOW publication AS (
        PARTITION BY origin_cluster, service, name, resource_attributes_json, attributes_json,
                     producer_epoch, source_sequence
    )
    QUALIFY MIN(count) OVER publication = MAX(count) OVER publication
        AND MIN(total) OVER publication = MAX(total) OVER publication
        AND MIN(schema_key) OVER publication = MAX(schema_key) OVER publication
        AND MIN(timestamp_ms) OVER publication = MAX(timestamp_ms) OVER publication
        AND ROW_NUMBER() OVER (publication ORDER BY seq DESC) = 1
), scalar_histogram_values AS (
    SELECT samples.origin_cluster, samples.service, samples.name,
           samples.resource_attributes_json, samples.attributes_json,
           samples.timestamp_ms, samples.seq, samples.value,
           CASE WHEN samples.name LIKE '%_count'
                     AND samples.value >= 0 AND samples.value < 9007199254740992
                     AND samples.value = FLOOR(samples.value)
                THEN CAST(samples.value AS BIGINT) ELSE NULL END AS integer_value,
           CAST(NULL AS VARCHAR) AS producer_epoch,
           CAST(NULL AS BIGINT) AS source_sequence,
           CAST(NULL AS VARCHAR) AS schema_key
    FROM base AS samples
    WHERE samples.name IN (
        'time_to_first_token_seconds_count', 'time_to_first_token_seconds_sum',
        'inter_token_latency_seconds_count', 'inter_token_latency_seconds_sum')
      AND json_get(samples.attributes_json, 'source_temporality') = 'cumulative_snapshot'
      AND (samples.service = 'vllm' OR json_get(samples.attributes_json, 'engine_index') IS NOT NULL)
      AND NOT EXISTS (
          SELECT 1 FROM structured_histograms AS structured
          WHERE samples.publication_id IS NOT NULL
            AND structured.publication_id = samples.publication_id
            AND samples.name IN (structured.name || '_count', structured.name || '_sum')
            AND structured.origin_cluster = samples.origin_cluster
            AND structured.service = samples.service
            AND structured.resource_attributes_json = samples.resource_attributes_json
            AND structured.attributes_json = samples.attributes_json
      )
), histogram_values AS (
    SELECT * FROM scalar_histogram_values
    UNION ALL
    SELECT origin_cluster, service, name || '_count', resource_attributes_json, attributes_json,
           timestamp_ms, seq, CAST(count AS DOUBLE), count, producer_epoch, source_sequence, schema_key
    FROM structured_histograms
    UNION ALL
    SELECT origin_cluster, service, name || '_sum', resource_attributes_json, attributes_json,
           timestamp_ms, seq, total, CAST(NULL AS BIGINT), producer_epoch, source_sequence, schema_key
    FROM structured_histograms
), histogram_ordered AS (
    SELECT *,
           LAG(value) OVER hist_order AS previous_value,
           LAG(integer_value) OVER hist_order AS previous_integer_value,
           LAG(producer_epoch) OVER hist_order AS previous_producer_epoch,
           LAG(schema_key) OVER hist_order AS previous_schema_key
    FROM histogram_values
    WINDOW hist_order AS (
        PARTITION BY origin_cluster, service, name, resource_attributes_json, attributes_json
        ORDER BY timestamp_ms, COALESCE(source_sequence, seq), seq
    )
), histogram_increments AS (
    SELECT origin_cluster, service, name, resource_attributes_json, attributes_json, timestamp_ms,
           CASE
               WHEN previous_value IS NULL
                 OR (producer_epoch IS NOT NULL AND previous_producer_epoch IS NOT NULL
                     AND producer_epoch <> previous_producer_epoch)
                 OR (schema_key IS NOT NULL AND previous_schema_key IS NOT NULL
                     AND schema_key <> previous_schema_key)
               THEN NULL
               WHEN integer_value IS NOT NULL AND previous_integer_value IS NOT NULL
               THEN CASE WHEN integer_value < previous_integer_value THEN NULL
                         ELSE CAST(integer_value - previous_integer_value AS DOUBLE) END
               WHEN value < previous_value THEN NULL
               ELSE value - previous_value
           END AS delta
    FROM histogram_ordered
    WHERE timestamp_ms >= {overview.start_ms}
), cumulative AS MATERIALIZED (
    SELECT *, LAG(value) OVER (
        PARTITION BY origin_cluster, service, name, resource_attributes_json, attributes_json
        ORDER BY timestamp_ms, seq
    ) AS previous_value
    FROM base
    WHERE json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
      AND name NOT IN ('time_to_first_token_seconds_count', 'time_to_first_token_seconds_sum',
                       'inter_token_latency_seconds_count', 'inter_token_latency_seconds_sum')
), increments AS MATERIALIZED (
    SELECT origin_cluster, service, name, resource_attributes_json, attributes_json, timestamp_ms,
           CASE WHEN previous_value IS NULL OR value < previous_value
                THEN NULL ELSE value - previous_value END AS delta
    FROM cumulative
    WHERE timestamp_ms >= {overview.start_ms}
    UNION ALL
    SELECT origin_cluster, service, name, resource_attributes_json, attributes_json, timestamp_ms,
           value AS delta
    FROM base
    WHERE timestamp_ms >= {overview.start_ms} AND kind = 'counter'
      AND COALESCE(json_get(attributes_json, 'source_temporality'), '') <> 'cumulative_snapshot'
), coherent_histograms AS (
    SELECT origin_cluster, service, resource_attributes_json, attributes_json,
           CASE WHEN name LIKE 'time_to_first_token_seconds_%' THEN 'ttft' ELSE 'itl' END AS family,
           timestamp_ms - timestamp_ms % {VLLM_HISTOGRAM_COHERENCE_MS} AS sample_t,
           SUM(CASE WHEN name LIKE '%_sum' THEN delta ELSE 0 END) AS seconds,
           SUM(CASE WHEN name LIKE '%_count' THEN delta ELSE 0 END) AS tokens,
           COUNT(*) AS components,
           COUNT(delta) AS valid_components,
           COUNT(*) FILTER (WHERE name LIKE '%_sum') AS sums,
           COUNT(*) FILTER (WHERE name LIKE '%_count') AS counts
    FROM histogram_increments
    GROUP BY 1, 2, 3, 4, 5, 6
    HAVING COUNT(*) = COUNT(delta)
       AND COUNT(*) FILTER (WHERE name LIKE '%_sum') = COUNT(*) FILTER (WHERE name LIKE '%_count')
       AND COUNT(*) FILTER (WHERE name LIKE '%_count') > 0
), output AS (
    SELECT CAST(NULL AS BIGINT) AS t, 'run_summary' AS section,
           CASE name
               WHEN 'generation_tokens_total' THEN 'generated_tokens'
               WHEN 'prompt_tokens_total' THEN 'prompt_tokens'
               WHEN 'num_preemptions_total' THEN 'preemptions'
               ELSE 'ttft_observations' END AS metric,
           'total' AS stat, name AS series, SUM(delta) AS value,
           CASE WHEN name = 'num_preemptions_total' THEN 'preemptions' ELSE 'tokens' END AS unit,
           CAST(NULL AS VARCHAR) AS status, CAST(COUNT(delta) AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM increments
    WHERE name IN ('generation_tokens_total', 'prompt_tokens_total', 'num_preemptions_total')
    GROUP BY name HAVING COUNT(delta) > 0

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t, 'run_summary' AS section,
           'ttft_observations' AS metric, 'total' AS stat,
           'time_to_first_token_seconds_count' AS series,
           SUM(delta) AS value, 'observations' AS unit,
           CAST(NULL AS VARCHAR) AS status, CAST(COUNT(delta) AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM histogram_increments
    WHERE name = 'time_to_first_token_seconds_count'
    HAVING COUNT(delta) > 0

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t, 'run_summary' AS section, name AS metric, 'total' AS stat,
           name || ':' || COALESCE(json_get(attributes_json, 'finished_reason'),
                                  json_get(attributes_json, 'finish_reason'), 'unknown') AS series,
           SUM(delta) AS value, 'engine finishes' AS unit, CAST(NULL AS VARCHAR) AS status,
           CAST(COUNT(delta) AS BIGINT) AS samples, CAST(NULL AS DOUBLE) AS gap_seconds
    FROM increments
    WHERE name IN ({sql_values(_OUTCOME_COUNTERS)})
    GROUP BY name, COALESCE(json_get(attributes_json, 'finished_reason'),
                            json_get(attributes_json, 'finish_reason'), 'unknown')
    HAVING COUNT(delta) > 0

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t, 'run_summary' AS section,
           'inter_token_latency' AS metric, 'token_weighted_mean' AS stat,
           'native inter-token latency' AS series, SUM(seconds) / NULLIF(SUM(tokens), 0) AS value, 's' AS unit,
           CAST(NULL AS VARCHAR) AS status, CAST(SUM(tokens) AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM coherent_histograms
    WHERE family = 'itl'
    HAVING SUM(tokens) > 0

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t, 'run_summary' AS section,
           name AS metric, 'observed_peak' AS stat, name AS series,
           MAX(value) AS value,
           CASE WHEN name = 'num_requests_waiting' THEN 'requests' ELSE 'ratio' END AS unit,
           CAST(NULL AS VARCHAR) AS status, CAST(COUNT(*) AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM base
    WHERE timestamp_ms >= {overview.start_ms}
      AND name IN ('num_requests_waiting', 'kv_cache_usage_perc', 'gpu_cache_usage_perc')
      AND json_get(attributes_json, 'source_temporality') = 'current_snapshot'
    GROUP BY name

    UNION ALL

    SELECT {overview.start_ms} + (timestamp_ms - {overview.start_ms})
               - (timestamp_ms - {overview.start_ms}) % 3600000 AS t,
           'run_timeline' AS section, 'generated_tokens' AS metric,
           'hourly_total' AS stat, 'generated tokens' AS series,
           SUM(delta) AS value, 'tokens' AS unit, CAST(NULL AS VARCHAR) AS status,
           CAST(COUNT(delta) AS BIGINT) AS samples, CAST(NULL AS DOUBLE) AS gap_seconds
    FROM increments
    WHERE name = 'generation_tokens_total' AND delta IS NOT NULL
    GROUP BY 1
)
SELECT t, section, metric, stat, series, value, unit, status, samples, gap_seconds
FROM output
ORDER BY section, t, metric, series
LIMIT {VLLM_MAX_SUMMARY_ROWS + 1}
""".strip()


def vllm_run_summary_table(
    overview: VllmOverviewQuery,
    series: pa.Table,
    projection_lock: AbstractContextManager[None],
    *,
    max_rows: int,
) -> pa.Table:
    """Return a bounded summary of the selected long-range signals."""
    return _vllm_project_table(
        vllm_run_summary_query(overview), series, projection_lock, max_rows=min(max_rows, VLLM_MAX_SUMMARY_ROWS)
    )
