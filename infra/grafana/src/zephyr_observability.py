# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One ranked shuffle snapshot shared by the Zephyr dashboard panels."""

from dashboard_dataset import DashboardDataset, SourceQuery, validate_time_window, validate_value
from vllm_observability import sql_string

ZEPHYR_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
ZEPHYR_MAX_TARGETS = 100_000
ZEPHYR_MAX_RESULT_ROWS = 200_000
ZEPHYR_MAX_IDENTITY_LENGTH = 512


def zephyr_overview_dataset(
    execution_id: str,
    stage_name: str,
    start_ms: int,
    end_ms: int,
) -> DashboardDataset:
    """Build the bounded ranked snapshot and its four fixed projections."""
    validate_value("execution_id", execution_id, max_length=ZEPHYR_MAX_IDENTITY_LENGTH)
    validate_value("stage_name", stage_name, max_length=ZEPHYR_MAX_IDENTITY_LENGTH)
    validate_time_window(
        start_ms,
        end_ms,
        max_window_ms=ZEPHYR_MAX_WINDOW_MS,
        max_window_error="Zephyr overview range must not exceed 7 days",
    )

    source_sql = f"""
WITH snapshots AS (
    SELECT target_shard,
           num_targets,
           input_rows,
           payload_bytes,
           num_sources,
           attempt,
           ROW_NUMBER() OVER (
               PARTITION BY target_shard
               ORDER BY attempt DESC, (input_rows IS NOT NULL) DESC, ts DESC, seq DESC
           ) AS sample_rank
    FROM "zephyr.shuffle"
    WHERE ts >= to_timestamp_millis({start_ms})
      AND ts < to_timestamp_millis({end_ms})
      AND execution_id = {sql_string(execution_id)}
      AND stage_name = {sql_string(stage_name)}
)
SELECT target_shard, num_targets, input_rows, payload_bytes, num_sources, attempt
FROM snapshots
WHERE sample_rank = 1
ORDER BY target_shard
LIMIT {ZEPHYR_MAX_TARGETS + 1}
""".strip()
    views = {
        "rows": (
            """
SELECT CAST(target_shard AS VARCHAR) AS target_reducer, input_rows
FROM snapshots WHERE input_rows IS NOT NULL
ORDER BY input_rows DESC, target_shard
""".strip()
        ),
        "payload_bytes": (
            """
SELECT CAST(target_shard AS VARCHAR) AS target_reducer, payload_bytes
FROM snapshots WHERE input_rows IS NOT NULL
ORDER BY payload_bytes DESC, target_shard
""".strip()
        ),
        "coverage": (
            """
SELECT CAST(COUNT(input_rows) AS BIGINT) AS observed_targets,
       CAST(MAX(num_targets) AS BIGINT) AS expected_targets,
       CAST(MAX(num_targets) - COUNT(input_rows) AS BIGINT) AS unreported_targets,
       CAST(SUM(CASE WHEN input_rows = 0 THEN 1 ELSE 0 END) AS BIGINT) AS reported_empty_targets,
       CAST(SUM(input_rows) AS BIGINT) AS observed_input_rows
FROM snapshots
""".strip()
        ),
        "reducers": (
            """
SELECT target_shard,
       CASE WHEN input_rows IS NULL THEN 'UNREPORTED'
            WHEN input_rows = 0 THEN 'EMPTY' ELSE 'REPORTED' END AS status,
       CASE WHEN input_rows IS NOT NULL THEN attempt END AS attempt,
       input_rows,
       payload_bytes,
       num_sources
FROM snapshots ORDER BY target_shard
""".strip()
        ),
    }
    return DashboardDataset(
        name="Zephyr overview",
        cache_key=(execution_id, stage_name, start_ms, end_ms),
        sources=(SourceQuery("snapshots", source_sql, ZEPHYR_MAX_TARGETS),),
        setup_sql=(),
        views=views,
        max_result_rows=ZEPHYR_MAX_RESULT_ROWS,
    )
