# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded compatibility query for legacy and bundled vLLM histograms."""

from vllm_observability import (
    _HISTOGRAM_BOUND_ORDER_SQL,
    _HISTOGRAM_FAMILIES,
    _HISTOGRAM_NAMES,
    VLLM_HISTOGRAM_COHERENCE_MS,
    VLLM_MAX_FRESHNESS_DETAILS,
    VLLM_MAX_RESULT_ROWS,
    VLLM_SNAPSHOT_LOOKBACK_MS,
    VllmOverviewQuery,
    _case_for,
    _histogram_component_mapping,
    _histogram_name_mapping,
    _histogram_source_mapping,
    sql_string,
    sql_values,
)

VLLM_HISTOGRAM_BUNDLE_NAME = "vllm_histogram_bundle"
VLLM_HISTOGRAM_BUNDLE_ENCODING = "explicit_bounds_cumulative_bundle_v2"
VLLM_HISTOGRAM_QUERY_DELIMITER = "|"


def _time_output_sql(start_ms: int, bucket_ms: int) -> str:
    return f"""histogram_time_means AS (
    SELECT {start_ms} + (sample_t - {start_ms}) - (sample_t - {start_ms}) % {bucket_ms} AS t,
           family,
           SUM(CASE WHEN component = 'sum' THEN delta ELSE 0 END)
               / NULLIF(SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END), 0) AS mean,
           SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) AS samples
    FROM coherent_histogram_increments
    GROUP BY 1, 2
    HAVING SUM(CASE WHEN component = 'count' THEN delta ELSE 0 END) > 0
), output AS (
    SELECT t,
           CASE WHEN family = 'iteration_tokens' THEN 'saturation' ELSE 'latency' END AS section,
           family AS metric,
           CASE WHEN family = 'iteration_tokens' THEN 'mean' ELSE 'mean_over_time' END AS stat,
           CASE
               WHEN family = 'iteration_tokens' THEN 'iteration tokens per engine step'
               WHEN family = 'tpot' THEN 'time per output token'
               ELSE family
           END AS series,
           mean AS value,
           CASE WHEN family = 'iteration_tokens' THEN 'tokens' ELSE 's' END AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(samples AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM histogram_time_means
    WHERE family <> 'output_tokens'
)"""


def _summary_output_sql() -> str:
    return f"""histogram_components AS (
    SELECT family, component, upper_bound, SUM(delta) AS component_total
    FROM coherent_histogram_increments
    GROUP BY 1, 2, 3
), histogram_rollup AS (
    SELECT *,
           SUM(CASE WHEN component = 'sum' THEN component_total ELSE 0 END) OVER (PARTITION BY family)
               / NULLIF(
                   SUM(CASE WHEN component = 'count' THEN component_total ELSE 0 END)
                       OVER (PARTITION BY family),
                   0
               ) AS mean,
           SUM(CASE WHEN component = 'count' THEN component_total ELSE 0 END) OVER (PARTITION BY family)
               AS samples,
           MAX(CASE WHEN component = 'bucket' THEN component_total END) OVER (PARTITION BY family)
               AS total_bucket_count,
           ROW_NUMBER() OVER (PARTITION BY family ORDER BY component, upper_bound) AS family_rank
    FROM histogram_components
), histogram_stats AS (
    SELECT *,
           MIN(CASE
               WHEN component = 'bucket'
                AND upper_bound NOT IN ('+Inf', 'Inf')
                AND component_total >= total_bucket_count * 0.50
               THEN CAST(upper_bound AS DOUBLE)
           END) OVER (PARTITION BY family) AS p50,
           MIN(CASE
               WHEN component = 'bucket'
                AND upper_bound NOT IN ('+Inf', 'Inf')
                AND component_total >= total_bucket_count * 0.90
               THEN CAST(upper_bound AS DOUBLE)
           END) OVER (PARTITION BY family) AS p90,
           MIN(CASE
               WHEN component = 'bucket'
                AND upper_bound NOT IN ('+Inf', 'Inf')
                AND component_total >= total_bucket_count * 0.99
               THEN CAST(upper_bound AS DOUBLE)
           END) OVER (PARTITION BY family) AS p99
    FROM histogram_rollup
), output_length_distribution AS (
    SELECT upper_bound,
           component_total - COALESCE(
               LAG(component_total) OVER (ORDER BY {_HISTOGRAM_BOUND_ORDER_SQL}),
               0
           ) AS value,
           total_bucket_count
    FROM histogram_stats
    WHERE family = 'output_tokens'
      AND component = 'bucket'
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
), output AS (
    SELECT CAST(NULL AS BIGINT) AS t,
           CASE WHEN family = 'output_tokens' THEN 'workload' ELSE 'latency' END AS section,
           family AS metric,
           requested.stat,
           family AS series,
           CASE requested.stat
               WHEN 'mean' THEN mean
               WHEN 'p50' THEN p50
               WHEN 'p90' THEN p90
               ELSE p99
           END AS value,
           CASE WHEN family = 'output_tokens' THEN 'tokens' ELSE 's' END AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(samples AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM histogram_stats
    CROSS JOIN (VALUES ('mean'), ('p50'), ('p90'), ('p99')) AS requested(stat)
    WHERE family_rank = 1
      AND family <> 'iteration_tokens'
      AND samples > 0

    UNION ALL

    SELECT CAST(NULL AS BIGINT) AS t,
           'output_length_distribution' AS section,
           'output_tokens' AS metric,
           'interval_count' AS stat,
           upper_bound AS series,
           value AS value,
           'requests' AS unit,
           CAST(NULL AS VARCHAR) AS status,
           CAST(total_bucket_count AS BIGINT) AS samples,
           CAST(NULL AS DOUBLE) AS gap_seconds
    FROM output_length_distribution

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
)"""


def _vllm_histogram_overview_query(
    overview: VllmOverviewQuery,
    output_sql: str,
) -> VllmOverviewQuery:
    """Return one histogram query over an already validated overview window."""
    identity_field = overview.identity_field
    identity = overview.identity
    start_ms = overview.start_ms
    end_ms = overview.end_ms
    bucket_ms = overview.bucket_ms
    scan_start_ms = max(0, start_ms - VLLM_SNAPSHOT_LOOKBACK_MS)
    identity_literal = sql_string(identity)
    histogram_names = sql_values(_HISTOGRAM_NAMES)
    histogram_family = _case_for(_histogram_name_mapping(), "name")
    histogram_component = _case_for(_histogram_component_mapping(), "name")
    histogram_source_family = _case_for(_histogram_source_mapping(), "name")
    structured_family = _case_for(_HISTOGRAM_FAMILIES, "source_family")
    sql = f"""
WITH legacy_base AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
           service,
           resource_attributes_json,
           attributes_json,
           timestamp_ms,
           seq,
           name,
           value
    FROM "telemetry_v1.vllm"
    WHERE service = 'vllm'
      AND {identity_field.value} = {identity_literal}
      AND name IN ({histogram_names})
      AND timestamp_ms >= {scan_start_ms}
      AND timestamp_ms < {end_ms}

    UNION ALL

    SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
           service,
           resource_attributes_json,
           attributes_json,
           timestamp_ms,
           seq,
           name,
           value
    FROM "telemetry_v1.marinskyrl"
    WHERE service = 'marinskyrl'
      AND {identity_field.value} = {identity_literal}
      AND name IN ({histogram_names})
      AND json_get(attributes_json, 'metric_source') = 'vllm'
      AND timestamp_ms >= {scan_start_ms}
      AND timestamp_ms < {end_ms}
), bundle_base AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'local') AS origin_cluster,
           service,
           resource_attributes_json,
           COALESCE(json_get(attributes_json, 'engine'), resource_attributes_json) AS producer_identity,
           timestamp_ms,
           json_get(attributes_json, 'histogram_publication_id') AS publication_id,
           json_get(body_json, 'delta_family_names_pipe') AS delta_family_names_pipe,
           json_get(body_json, 'delta_family_indexes_pipe') AS delta_family_indexes_pipe,
           json_get(body_json, 'delta_component_kinds_pipe') AS delta_component_kinds_pipe,
           json_get(body_json, 'delta_component_bounds_pipe') AS delta_component_bounds_pipe,
           json_get(body_json, 'delta_component_values_pipe') AS delta_component_values_pipe
    FROM "telemetry_v1.marinskyrl"
    WHERE service = 'marinskyrl'
      AND {identity_field.value} = {identity_literal}
      AND name = {sql_string(VLLM_HISTOGRAM_BUNDLE_NAME)}
      AND kind = 'event'
      AND body_json IS NOT NULL
      AND json_get(attributes_json, 'metric_source') = 'vllm'
      AND json_get(attributes_json, 'source_kind') = 'histogram_bundle'
      AND json_get(attributes_json, 'source_temporality') = 'cumulative_snapshot'
      AND json_get(attributes_json, 'histogram_encoding') = {sql_string(VLLM_HISTOGRAM_BUNDLE_ENCODING)}
      AND json_get(body_json, 'encoding') = {sql_string(VLLM_HISTOGRAM_BUNDLE_ENCODING)}
      AND timestamp_ms >= {scan_start_ms}
      AND timestamp_ms < {end_ms}
), structured_publications AS (
    SELECT DISTINCT origin_cluster, service, resource_attributes_json, publication_id
    FROM bundle_base
    WHERE publication_id IS NOT NULL
), structured_bundle_base AS (
    SELECT *
    FROM bundle_base
    WHERE delta_component_values_pipe IS NOT NULL
), structured_components AS (
    SELECT timestamp_ms,
           origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           delta_family_names_pipe,
           UNNEST(string_to_array(delta_family_indexes_pipe, {sql_string(VLLM_HISTOGRAM_QUERY_DELIMITER)}))
               AS family_index,
           UNNEST(string_to_array(delta_component_kinds_pipe, {sql_string(VLLM_HISTOGRAM_QUERY_DELIMITER)}))
               AS component,
           UNNEST(string_to_array(delta_component_bounds_pipe, {sql_string(VLLM_HISTOGRAM_QUERY_DELIMITER)}))
               AS component_bound,
           UNNEST(string_to_array(delta_component_values_pipe, {sql_string(VLLM_HISTOGRAM_QUERY_DELIMITER)}))
               AS component_value
    FROM structured_bundle_base
), structured_named AS (
    SELECT *,
           split_part(
               delta_family_names_pipe,
               {sql_string(VLLM_HISTOGRAM_QUERY_DELIMITER)},
               TRY_CAST(family_index AS BIGINT) + 1
           ) AS source_family
    FROM structured_components
), structured_coherent AS (
    SELECT timestamp_ms - timestamp_ms % {VLLM_HISTOGRAM_COHERENCE_MS} AS sample_t,
           origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           {structured_family} AS family,
           component,
           CASE WHEN component = 'bucket' THEN component_bound END AS upper_bound,
           CASE
               WHEN component = 'sum' THEN TRY_CAST(component_value AS DOUBLE)
               ELSE CAST(TRY_CAST(component_value AS BIGINT) AS DOUBLE)
           END AS delta
    FROM structured_named
    WHERE timestamp_ms >= {start_ms}
      AND component IN ('bucket', 'count', 'sum')
      AND component_value IS NOT NULL
), legacy_samples AS (
    SELECT legacy.origin_cluster,
           legacy.service,
           legacy.resource_attributes_json,
           legacy.attributes_json,
           COALESCE(json_get(legacy.attributes_json, 'engine'), legacy.resource_attributes_json)
               AS producer_identity,
           COALESCE(
               TRY_CAST(json_get(legacy.attributes_json, 'histogram_collection_timestamp_ms') AS BIGINT),
               legacy.timestamp_ms
           ) AS timestamp_ms,
           legacy.seq,
           {histogram_source_family} AS source_family,
           {histogram_family} AS family,
           {histogram_component} AS component,
           json_get(legacy.attributes_json, 'le') AS upper_bound,
           COALESCE(
               json_get(legacy.attributes_json, 'histogram_series'),
               json_get(legacy.attributes_json, 'model_name'),
               ''
           ) AS series_identity,
           COALESCE(json_get(legacy.attributes_json, 'histogram_schema'), 'legacy') AS histogram_schema,
           TRY_CAST(json_get(legacy.attributes_json, 'histogram_sample_sequence') AS BIGINT) AS sample_sequence,
           legacy.value AS cumulative_value
    FROM legacy_base AS legacy
    LEFT JOIN structured_publications AS structured
      ON legacy.origin_cluster = structured.origin_cluster
     AND legacy.service = structured.service
     AND COALESCE(legacy.resource_attributes_json, '') = COALESCE(structured.resource_attributes_json, '')
     AND json_get(legacy.attributes_json, 'histogram_publication_id') = structured.publication_id
    WHERE legacy.value IS NOT NULL
      AND json_get(legacy.attributes_json, 'source_temporality') = 'cumulative_snapshot'
      AND structured.publication_id IS NULL
      AND (
          legacy.service = 'vllm'
          OR json_get(legacy.attributes_json, 'engine_index') IS NOT NULL
      )
), legacy_ordered AS (
    SELECT *,
           timestamp_ms - timestamp_ms % {VLLM_HISTOGRAM_COHERENCE_MS} AS sample_t,
           LAG(cumulative_value) OVER (
               PARTITION BY origin_cluster,
                            service,
                            resource_attributes_json,
                            producer_identity,
                            source_family,
                            series_identity,
                            histogram_schema,
                            component,
                            upper_bound
               ORDER BY timestamp_ms, sample_sequence, seq
           ) AS previous_value
    FROM legacy_samples
), legacy_component_increments AS (
    SELECT *,
           cumulative_value - previous_value AS delta,
           CASE
               WHEN previous_value IS NULL OR cumulative_value < previous_value THEN 1
               ELSE 0
           END AS invalid_component
    FROM legacy_ordered
), legacy_series AS (
    SELECT DISTINCT origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           source_family,
           series_identity,
           histogram_schema,
           component,
           upper_bound
    FROM legacy_component_increments
), legacy_expected_series AS (
    SELECT origin_cluster,
           service,
           resource_attributes_json,
           producer_identity,
           source_family,
           series_identity,
           histogram_schema,
           COUNT(*) AS expected_series
    FROM legacy_series
    GROUP BY 1, 2, 3, 4, 5, 6, 7
), legacy_sample_validity AS (
    SELECT samples.origin_cluster,
           samples.service,
           samples.resource_attributes_json,
           samples.producer_identity,
           samples.source_family,
           samples.series_identity,
           samples.histogram_schema,
           samples.sample_t,
           CASE
               WHEN MAX(samples.invalid_component) = 1 OR COUNT(*) < MAX(expected.expected_series) THEN 0
               ELSE 1
           END AS valid_sample
    FROM legacy_component_increments AS samples
    JOIN legacy_expected_series AS expected
      ON samples.origin_cluster = expected.origin_cluster
     AND samples.service = expected.service
     AND samples.resource_attributes_json = expected.resource_attributes_json
     AND samples.producer_identity = expected.producer_identity
     AND samples.source_family = expected.source_family
     AND samples.series_identity = expected.series_identity
     AND samples.histogram_schema = expected.histogram_schema
    WHERE samples.timestamp_ms >= {start_ms}
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
), legacy_coherent AS (
    SELECT samples.sample_t,
           samples.origin_cluster,
           samples.service,
           samples.resource_attributes_json,
           samples.producer_identity,
           samples.family,
           samples.component,
           samples.upper_bound,
           samples.delta
    FROM legacy_component_increments AS samples
    JOIN legacy_sample_validity AS validity
      ON samples.origin_cluster = validity.origin_cluster
     AND samples.service = validity.service
     AND samples.resource_attributes_json = validity.resource_attributes_json
     AND samples.producer_identity = validity.producer_identity
     AND samples.source_family = validity.source_family
     AND samples.series_identity = validity.series_identity
     AND samples.histogram_schema = validity.histogram_schema
     AND samples.sample_t = validity.sample_t
    WHERE validity.valid_sample = 1
), coherent_histogram_increments AS (
    SELECT * FROM legacy_coherent

    UNION ALL

    SELECT * FROM structured_coherent
), {output_sql}
SELECT t, section, metric, stat, series, value, unit, status, samples, gap_seconds
FROM output
ORDER BY section,
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
        samples_sql="",
        identity_field=identity_field,
        identity=identity,
        start_ms=start_ms,
        end_ms=end_ms,
        bucket_ms=bucket_ms,
    )


def vllm_histogram_overview_queries(
    overview: VllmOverviewQuery,
) -> tuple[VllmOverviewQuery, VllmOverviewQuery]:
    """Return independently bounded time-series and summary histogram queries."""
    return (
        _vllm_histogram_overview_query(overview, _time_output_sql(overview.start_ms, overview.bucket_ms)),
        _vllm_histogram_overview_query(overview, _summary_output_sql()),
    )
