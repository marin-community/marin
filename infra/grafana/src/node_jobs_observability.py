# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One bounded fetch of a node's task attempts and their usage, for the Node Details job row.

A separate dataset from the node overview: a dataset runs all its sources on every request, so
the controller-row queries here cost nothing until someone opens the job row.
"""

import re

from config import CLUSTERS, K8S_CLUSTERS
from dashboard_dataset import DashboardDataset, SourceQuery, bounded_bucket_ms, validate_values
from node_observability import NODE_IDENTITY_MAX_LENGTH, NODE_MAX_POINTS, NODE_MAX_WINDOW_MS, NODE_MIN_BUCKET_MS
from vllm_observability import sql_string, sql_values

# An attempt still running when the range starts is found only if its newest controller row is
# at most this old; a longer one needs a precomputed table.
NODE_JOBS_LOOKBACK_MS = 24 * 60 * 60 * 1000
# Attempts that overlap the range on one node. A busy GPU node had about 10 over 3 h
# (2026-11-26); a CPU node running many short tasks for 7 days can have thousands.
NODE_JOBS_MAX_ATTEMPTS = 5_000
# Usage rows are per bucket per attempt; the bucket caps a range at NODE_MAX_POINTS buckets.
# 545 rows over 3 h for 3 attempts on the busy node (2026-11-26).
NODE_JOBS_MAX_USAGE_ROWS = 200_000
NODE_JOBS_MAX_RESULT_ROWS = 250_000

# A Kubernetes node name or an Iris worker id.
_NODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_JOB_KEY = '"cluster", root_job_id, root_submitted_at_ms, job_id, submitted_at_ms'


def _validate_node_scope(clusters: tuple[str, ...], node: str) -> None:
    validate_values(
        "clusters", clusters, max_values=len(CLUSTERS) + len(K8S_CLUSTERS), max_length=NODE_IDENTITY_MAX_LENGTH
    )
    known = {cluster.name for cluster in CLUSTERS} | {cluster.name for cluster in K8S_CLUSTERS}
    unknown = sorted(set(clusters) - known)
    if unknown:
        raise ValueError(f"unknown clusters {unknown}; configured: {sorted(known)}")
    validate_values("node", (node,), max_values=1, max_length=NODE_IDENTITY_MAX_LENGTH)
    if not _NODE_PATTERN.match(node):
        raise ValueError("node must be a node name or worker id")


def _attempts_on_node(clusters: tuple[str, ...], node: str, start_ms: int, end_ms: int) -> str:
    """The newest row of each attempt on the node that overlaps the range."""
    return f"""
SELECT {_JOB_KEY}, task_id, attempt_id, attempt_uid, state, started_at, finished_at
FROM (
    SELECT {_JOB_KEY}, task_id, attempt_id, attempt_uid, state, started_at, finished_at,
           row_number() OVER (PARTITION BY attempt_uid ORDER BY ts DESC, seq DESC) AS rn
    FROM "iris.controller_db__task_attempts"
    WHERE (node_name = {sql_string(node)} OR worker_id = {sql_string(node)})
      AND "cluster" IN ({sql_values(clusters)})
      AND ts >= to_timestamp_millis({start_ms - NODE_JOBS_LOOKBACK_MS})
      AND ts < to_timestamp_millis({end_ms})
)
WHERE rn = 1 AND started_at IS NOT NULL
  AND started_at < to_timestamp_millis({end_ms})
  AND COALESCE(finished_at, to_timestamp_millis({end_ms})) > to_timestamp_millis({start_ms})
"""


def node_jobs_dataset(
    clusters: tuple[str, ...],
    node: str,
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
) -> DashboardDataset:
    """Build the node's attempt and usage sources and the job row's local views."""
    _validate_node_scope(clusters, node)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=NODE_MAX_WINDOW_MS,
        max_window_error="node jobs range must not exceed 7 days",
        min_bucket_ms=max(NODE_MIN_BUCKET_MS, 60_000),
        max_points=NODE_MAX_POINTS,
    )
    attempts = _attempts_on_node(clusters, node, start_ms, end_ms)
    lookback = f"ts >= to_timestamp_millis({start_ms - NODE_JOBS_LOOKBACK_MS}) AND ts < to_timestamp_millis({end_ms})"
    attempts_sql = f"""
WITH attempt AS ({attempts}),
job AS (
    SELECT {_JOB_KEY}, name FROM (
        SELECT {_JOB_KEY}, name,
               row_number() OVER (PARTITION BY {_JOB_KEY} ORDER BY ts DESC, seq DESC) AS rn
        FROM "iris.controller_db__jobs"
        WHERE {lookback} AND root_job_id IN (SELECT root_job_id FROM attempt)
    ) WHERE rn = 1
),
request AS (
    SELECT {_JOB_KEY}, res_cpu_millicores, res_memory_bytes, accelerator_kind, accelerator_type, accelerators_per_task
    FROM (
        SELECT {_JOB_KEY}, res_cpu_millicores, res_memory_bytes, accelerator_kind, accelerator_type,
               accelerators_per_task,
               row_number() OVER (PARTITION BY {_JOB_KEY} ORDER BY ts DESC, seq DESC) AS rn
        FROM "iris.controller_db__job_config"
        WHERE {lookback} AND root_job_id IN (SELECT root_job_id FROM attempt)
    ) WHERE rn = 1
)
SELECT a.attempt_uid, a.task_id, a.attempt_id, a.job_id, j.name AS job_name, a.state,
       a.started_at, a.finished_at, r.res_cpu_millicores / 1000.0 AS cpu_request,
       r.res_memory_bytes AS memory_request_bytes, r.accelerator_kind, r.accelerator_type,
       r.accelerators_per_task, CAST(NULL AS VARCHAR) AS category
FROM attempt a LEFT JOIN job j USING ({_JOB_KEY}) LEFT JOIN request r USING ({_JOB_KEY})
"""
    usage_sql = f"""
WITH attempt AS ({attempts})
SELECT date_bin(INTERVAL '{bucket_ms} milliseconds', t.ts) AS bucket,
       t.task_id || ':' || CAST(t.attempt_id AS VARCHAR) AS attempt,
       avg(t.cpu_millicores) / 1000.0 AS cpu_cores, avg(t.memory_mb) * 1048576 AS memory_bytes
FROM "iris.task" t JOIN attempt a
  ON COALESCE(NULLIF(t."cluster", ''), 'marin') = a."cluster"
 AND t.task_id = a.task_id AND t.attempt_id = a.attempt_id
 AND t.ts >= a.started_at AND t.ts <= COALESCE(a.finished_at, to_timestamp_millis({end_ms}))
WHERE t.ts >= to_timestamp_millis({start_ms}) AND t.ts < to_timestamp_millis({end_ms})
  AND t.task_id IN (SELECT task_id FROM attempt)
GROUP BY 1, 2
"""
    job_activity_view = f"""
WITH grid AS (
    SELECT range AS bucket
    FROM range(epoch_ms({start_ms - start_ms % bucket_ms}), epoch_ms({end_ms}), INTERVAL '{bucket_ms} milliseconds')
)
SELECT g.bucket, a.job_id,
       max(CASE WHEN a.started_at < g.bucket + INTERVAL '{bucket_ms} milliseconds'
                 AND COALESCE(a.finished_at, epoch_ms({end_ms})) > g.bucket THEN 1 ELSE 0 END) AS active
FROM grid g CROSS JOIN attempts a
GROUP BY 1, 2
ORDER BY 1, 2
"""
    return DashboardDataset(
        name="node jobs",
        cache_key=(tuple(sorted(clusters)), node, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("attempts", attempts_sql, NODE_JOBS_MAX_ATTEMPTS),
            SourceQuery("attempt_usage", usage_sql, NODE_JOBS_MAX_USAGE_ROWS),
        ),
        setup_sql=(),
        views={
            "job_activity": job_activity_view,
            "attempts": (
                "SELECT job_id, job_name, task_id, attempt_id, attempt_uid, state, started_at, finished_at, cpu_request,"
                " memory_request_bytes, accelerator_kind, accelerator_type, accelerators_per_task, category"
                " FROM attempts ORDER BY started_at, attempt_uid"
            ),
            "attempt_usage": "SELECT bucket, attempt, cpu_cores, memory_bytes FROM attempt_usage ORDER BY 1, 2",
        },
        max_result_rows=NODE_JOBS_MAX_RESULT_ROWS,
    )
