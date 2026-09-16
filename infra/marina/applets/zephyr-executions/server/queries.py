# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQL for the Zephyr execution page, run against Finelog's stats service.

The statements match ``infra/grafana/dashboards/zephyr.json``. Every reader
applies one rule per target: keep the highest attempt, then a measurement
over a placeholder, then the latest ``ts`` and ingest ``seq``.
"""

from datetime import datetime

REDUCER_PAGE_SIZE = 20
EXECUTION_LIMIT = 100
STAGE_LOOKBACK_SECONDS = 60


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def time_predicate(start: datetime) -> str:
    return f"ts >= TIMESTAMP {sql_string(start.isoformat())} AND ts <= now()"


def executions_sql(*, since: datetime, root_job: str | None, limit: int) -> str:
    where = [time_predicate(since)]
    if root_job:
        where.append(f"root_job_id = {sql_string(root_job)}")
    return f"""SELECT execution_id, root_job_id, coordinator_job_id, ts, input_shards, stages_json
FROM "zephyr.execution"
WHERE {" AND ".join(where)}
QUALIFY ROW_NUMBER() OVER (PARTITION BY execution_id ORDER BY ts DESC, seq DESC) = 1
ORDER BY ts DESC LIMIT {int(limit)}"""


def execution_sql(execution_id: str) -> str:
    return f"""SELECT execution_id, root_job_id, coordinator_job_id, ts, input_shards, stages_json
FROM "zephyr.execution"
WHERE execution_id = {sql_string(execution_id)}
QUALIFY ROW_NUMBER() OVER (PARTITION BY execution_id ORDER BY ts DESC, seq DESC) = 1"""


def stage_stats_sql(execution_id: str, start: datetime) -> str:
    return f"""SELECT stage_name, status, elapsed, items, total_shards, mem_peak_bytes_max
FROM "zephyr.stage" WHERE execution_id = {sql_string(execution_id)} AND {time_predicate(start)}
QUALIFY ROW_NUMBER() OVER (PARTITION BY stage_name ORDER BY ts DESC, seq DESC) = 1"""


def shuffle_snapshots_sql(execution_id: str, stage: str, start: datetime) -> str:
    return f"""WITH snapshots AS (
SELECT *, ROW_NUMBER() OVER (
PARTITION BY target_shard ORDER BY attempt DESC, (input_rows IS NOT NULL) DESC, ts DESC, seq DESC
) AS sample_rank
FROM "zephyr.shuffle" WHERE execution_id = {sql_string(execution_id)}
AND stage_name = {sql_string(stage)} AND {time_predicate(start)}
)"""


def reducer_stats_sql(execution_id: str, stage: str, start: datetime, page: int) -> str:
    return f"""{shuffle_snapshots_sql(execution_id, stage, start)}
SELECT target_shard, input_rows, payload_bytes, num_sources, attempt FROM snapshots
WHERE sample_rank = 1 ORDER BY payload_bytes DESC NULLS LAST, target_shard
LIMIT {REDUCER_PAGE_SIZE} OFFSET {int(page) * REDUCER_PAGE_SIZE}"""


def reducer_task_stats_sql(execution_id: str, stage: str, start: datetime, page: int) -> str:
    return f"""WITH targets AS ({reducer_stats_sql(execution_id, stage, start, page)}),
task_states AS (
SELECT shard_idx, status FROM "zephyr.worker"
WHERE execution_id = {sql_string(execution_id)} AND stage_name = {sql_string(stage)} AND {time_predicate(start)}
AND shard_idx IN (SELECT target_shard FROM targets)
QUALIFY ROW_NUMBER() OVER (PARTITION BY shard_idx ORDER BY ts DESC, seq DESC) = 1
)
SELECT targets.*, task_states.status AS task_status FROM targets
LEFT JOIN task_states ON targets.target_shard = task_states.shard_idx
ORDER BY payload_bytes DESC NULLS LAST, target_shard"""


def shuffle_summary_sql(execution_id: str, stage: str, start: datetime) -> str:
    return f"""{shuffle_snapshots_sql(execution_id, stage, start)}
SELECT COUNT(*) AS persisted_targets, COUNT(input_rows) AS observed_targets,
MAX(num_targets) AS expected_targets, MEDIAN(CAST(input_rows AS DOUBLE)) AS median_rows,
MAX(input_rows) AS max_rows, MEDIAN(CAST(payload_bytes AS DOUBLE)) AS median_bytes,
MAX(payload_bytes) AS max_bytes FROM snapshots WHERE sample_rank = 1"""
