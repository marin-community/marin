# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Five namespace-bounded sources shared by the Jobs dashboard."""

from dashboard_dataset import DashboardDataset, SourceQuery, bounded_bucket_ms, validate_optional_values, validate_values
from vllm_observability import sql_values

JOBS_MAX_WINDOW_MS = 24 * 60 * 60 * 1000
JOBS_MAX_POINTS = 360
JOBS_MIN_BUCKET_MS = 15_000
JOBS_MAX_CLUSTERS = 16
JOBS_MAX_JOBS = 256
JOBS_MAX_STATE_ROWS = 150_000
JOBS_MAX_EVENT_ROWS = 50_000
JOBS_MAX_PROVISIONING_ROWS = 50_000
JOBS_MAX_RESOURCE_ROWS = 20_000
JOBS_MAX_RESULT_ROWS = 200_000
JOBS_STUCK_AGE_MS = 15 * 60 * 1000
JOBS_IDENTITY_MAX_LENGTH = 512
JOBS_RESOURCE_SERIES_LIMIT = 20


def jobs_overview_dataset(
    clusters: tuple[str, ...],
    jobs: tuple[str, ...],
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
) -> DashboardDataset:
    """Build one bounded source per Iris telemetry namespace."""
    validate_values("clusters", clusters, max_values=JOBS_MAX_CLUSTERS, max_length=JOBS_IDENTITY_MAX_LENGTH)
    validate_optional_values("jobs", jobs, max_values=JOBS_MAX_JOBS, max_length=JOBS_IDENTITY_MAX_LENGTH)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=JOBS_MAX_WINDOW_MS,
        max_window_error="Jobs overview range must not exceed 24 hours",
        min_bucket_ms=JOBS_MIN_BUCKET_MS,
        max_points=JOBS_MAX_POINTS,
    )
    cluster_values = sql_values(clusters)
    selected_jobs = f"root_job_id IN ({sql_values(jobs)})" if jobs else "FALSE"
    start = f"to_timestamp_millis({start_ms})"
    end = f"to_timestamp_millis({end_ms})"
    bucket = f"date_bin(INTERVAL '{bucket_ms} milliseconds', ts)"
    state_sql = f"""
WITH filtered AS (
    SELECT ts, COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster, root_job_id,
           pending, assigned, building, running, oldest_pending_age_ms, oldest_building_age_ms
    FROM "iris.task_state"
    WHERE COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND ts >= {start} AND ts < {end}
), historical AS (
    SELECT 'historical' AS kind, {bucket} AS t, origin_cluster, root_job_id,
           MAX(pending) AS pending, MAX(assigned) AS assigned,
           MAX(building) AS building, MAX(running) AS running,
           MAX(oldest_pending_age_ms) AS oldest_pending_age_ms,
           MAX(oldest_building_age_ms) AS oldest_building_age_ms
    FROM filtered
    WHERE root_job_id = '' OR {selected_jobs}
    GROUP BY 2, 3, 4
), latest_ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY origin_cluster, root_job_id ORDER BY ts DESC) AS rn
    FROM filtered
), latest AS (
    SELECT 'latest' AS kind, ts AS t, origin_cluster, root_job_id,
           pending, assigned, building, running, oldest_pending_age_ms, oldest_building_age_ms
    FROM latest_ranked
    WHERE rn = 1
      AND (root_job_id = '' OR {selected_jobs}
           OR GREATEST(oldest_pending_age_ms, oldest_building_age_ms) > {JOBS_STUCK_AGE_MS})
)
SELECT * FROM historical
UNION ALL SELECT * FROM latest
    ORDER BY kind, t, origin_cluster, root_job_id
LIMIT {JOBS_MAX_STATE_ROWS + 1}
""".strip()
    event_sql = f"""
WITH warnings AS (
    SELECT ts, COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           task_id, attempt_id, reason, source, message, count
    FROM "iris.task_event"
    WHERE type = 'Warning'
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND ts >= {start} AND ts < {end}
), recent AS (
    SELECT 'recent' AS kind, ts, origin_cluster, CAST(task_id AS VARCHAR) AS task_id,
           CAST(attempt_id AS VARCHAR) AS attempt_id, reason, source, message, CAST(count AS BIGINT) AS count
    FROM warnings ORDER BY ts DESC LIMIT 200
), bucketed AS (
    SELECT 'bucketed' AS kind, {bucket} AS ts,
           CAST(NULL AS VARCHAR) AS origin_cluster,
           CAST(NULL AS VARCHAR) AS task_id,
           CAST(NULL AS VARCHAR) AS attempt_id,
           reason,
           CAST(NULL AS VARCHAR) AS source,
           CAST(NULL AS VARCHAR) AS message,
           SUM(count) AS count
    FROM warnings GROUP BY 2, 6
)
SELECT * FROM recent
UNION ALL SELECT * FROM bucketed
ORDER BY kind, ts DESC
LIMIT {JOBS_MAX_EVENT_ROWS + 1}
""".strip()
    provisioning_sql = f"""
SELECT {bucket} AS t,
       outcome,
       CASE WHEN accelerator_variant = '' THEN resource_type ELSE accelerator_variant END AS accelerator,
       CAST(COUNT(*) AS BIGINT) AS attempts,
       AVG(CASE WHEN outcome = 'ready' AND provision_latency_ms > 0 THEN provision_latency_ms END) AS latency_ms
FROM "iris.provisioning"
WHERE COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
  AND ts >= {start} AND ts < {end}
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
LIMIT {JOBS_MAX_PROVISIONING_ROWS + 1}
""".strip()
    task_sql = f"""
WITH averages AS (
    SELECT task_id, AVG(memory_mb) AS memory_mb, AVG(cpu_millicores) AS cpu_millicores
    FROM "iris.task"
    WHERE COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND ts >= {start} AND ts < {end}
    GROUP BY 1
), selected_ids AS (
    SELECT task_id, memory_rank, cpu_rank
    FROM (SELECT task_id, ROW_NUMBER() OVER (ORDER BY memory_mb DESC, task_id) AS memory_rank,
                         ROW_NUMBER() OVER (ORDER BY cpu_millicores DESC, task_id) AS cpu_rank
          FROM averages)
    WHERE memory_rank <= {JOBS_RESOURCE_SERIES_LIMIT} OR cpu_rank <= {JOBS_RESOURCE_SERIES_LIMIT}
)
SELECT {bucket} AS t, task.task_id,
       AVG(task.memory_mb) AS memory_mb, AVG(task.cpu_millicores) AS cpu_millicores,
       selected_ids.memory_rank, selected_ids.cpu_rank
FROM "iris.task" AS task JOIN selected_ids USING (task_id)
WHERE COALESCE(NULLIF(task.cluster, ''), 'marin') IN ({cluster_values})
  AND task.ts >= {start} AND task.ts < {end}
GROUP BY 1, 2, 5, 6 ORDER BY 1, 2
LIMIT {JOBS_MAX_RESOURCE_ROWS + 1}
""".strip()
    worker_sql = f"""
WITH averages AS (
    SELECT worker_id, AVG(cpu_pct) AS cpu_pct, AVG(mem_bytes) AS mem_bytes
    FROM "iris.worker"
    WHERE COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND ts >= {start} AND ts < {end}
    GROUP BY 1
), selected_ids AS (
    SELECT worker_id, cpu_rank, memory_rank
    FROM (SELECT worker_id, ROW_NUMBER() OVER (ORDER BY cpu_pct DESC, worker_id) AS cpu_rank,
                           ROW_NUMBER() OVER (ORDER BY mem_bytes DESC, worker_id) AS memory_rank
          FROM averages)
    WHERE cpu_rank <= {JOBS_RESOURCE_SERIES_LIMIT} OR memory_rank <= {JOBS_RESOURCE_SERIES_LIMIT}
)
SELECT {bucket} AS t, worker.worker_id,
       AVG(worker.cpu_pct) AS cpu_pct, AVG(worker.mem_bytes) AS mem_bytes,
       selected_ids.cpu_rank, selected_ids.memory_rank
FROM "iris.worker" AS worker JOIN selected_ids USING (worker_id)
WHERE COALESCE(NULLIF(worker.cluster, ''), 'marin') IN ({cluster_values})
  AND worker.ts >= {start} AND worker.ts < {end}
GROUP BY 1, 2, 5, 6 ORDER BY 1, 2
LIMIT {JOBS_MAX_RESOURCE_ROWS + 1}
""".strip()
    views = {
        "provisioning_failures": (
            "SELECT SUM(attempts) AS value FROM provisioning WHERE outcome IN ('stockout', 'error')"
        ),
        "warning_count": "SELECT SUM(count) AS value FROM events WHERE kind = 'bucketed'",
        "fleet_state": (
            """
SELECT t, SUM(pending) AS pending, SUM(assigned) AS assigned,
       SUM(building) AS building, SUM(running) AS running
FROM task_state WHERE kind = 'historical' AND root_job_id = '' GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "oldest_waiting": (
            """
SELECT t, origin_cluster AS series, MAX(GREATEST(oldest_pending_age_ms, oldest_building_age_ms)) AS value
FROM task_state WHERE kind = 'historical' AND root_job_id = '' GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "provisioning_outcomes": (
            "SELECT t, outcome AS series, SUM(attempts) AS value FROM provisioning GROUP BY 1, 2 ORDER BY 1"
        ),
        "provisioning_latency": (
            "SELECT t, accelerator AS series, AVG(latency_ms) AS value "
            "FROM provisioning WHERE latency_ms IS NOT NULL GROUP BY 1, 2 ORDER BY 1"
        ),
        "waiting_jobs": (
            """
SELECT t, root_job_id AS series, MAX(pending + assigned + building) AS value
FROM task_state WHERE kind = 'historical' AND root_job_id <> '' GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "active_jobs": (
            f"""
SELECT origin_cluster AS cluster, root_job_id AS job, pending, assigned, building, running,
       oldest_pending_age_ms, oldest_building_age_ms
FROM task_state WHERE kind = 'latest' AND root_job_id <> '' AND {selected_jobs}
ORDER BY GREATEST(oldest_building_age_ms, oldest_pending_age_ms) DESC
""".strip()
        ),
        "recent_warnings": (
            "SELECT ts, origin_cluster AS cluster, task_id, attempt_id, reason, source, message, count "
            "FROM events WHERE kind = 'recent' ORDER BY ts DESC"
        ),
        "task_memory": (
            f"SELECT t, task_id AS series, memory_mb AS value FROM tasks "
            f"WHERE memory_rank <= {JOBS_RESOURCE_SERIES_LIMIT} ORDER BY 1"
        ),
        "task_cpu": (
            f"SELECT t, task_id AS series, cpu_millicores AS value FROM tasks "
            f"WHERE cpu_rank <= {JOBS_RESOURCE_SERIES_LIMIT} ORDER BY 1"
        ),
        "worker_cpu": (
            f"SELECT t, worker_id AS series, cpu_pct AS value FROM workers "
            f"WHERE cpu_rank <= {JOBS_RESOURCE_SERIES_LIMIT} ORDER BY 1"
        ),
        "worker_memory": (
            f"SELECT t, worker_id AS series, mem_bytes AS value FROM workers "
            f"WHERE memory_rank <= {JOBS_RESOURCE_SERIES_LIMIT} ORDER BY 1"
        ),
        "fleet_tasks": (
            "SELECT SUM(running) AS running, SUM(building) AS building, SUM(pending) AS pending "
            "FROM task_state WHERE kind = 'latest' AND root_job_id = ''"
        ),
        "stuck_jobs": (
            "SELECT COUNT(*) AS stuck FROM task_state WHERE kind = 'latest' AND root_job_id <> '' "
            f"AND GREATEST(oldest_pending_age_ms, oldest_building_age_ms) > {JOBS_STUCK_AGE_MS}"
        ),
        "queue_depth": (
            "SELECT t, origin_cluster AS series, MAX(pending) AS value FROM task_state "
            "WHERE kind = 'historical' AND root_job_id = '' GROUP BY 1, 2 ORDER BY 1"
        ),
        "warning_reasons": (
            "SELECT ts AS t, reason AS series, SUM(count) AS value "
            "FROM events WHERE kind = 'bucketed' GROUP BY 1, 2 ORDER BY 1"
        ),
    }
    return DashboardDataset(
        name="Jobs overview",
        cache_key=(clusters, jobs, start_ms, end_ms, bucket_ms),
        sources=(
            SourceQuery("task_state", state_sql, JOBS_MAX_STATE_ROWS),
            SourceQuery("events", event_sql, JOBS_MAX_EVENT_ROWS),
            SourceQuery("provisioning", provisioning_sql, JOBS_MAX_PROVISIONING_ROWS),
            SourceQuery("tasks", task_sql, JOBS_MAX_RESOURCE_ROWS),
            SourceQuery("workers", worker_sql, JOBS_MAX_RESOURCE_ROWS),
        ),
        setup_sql=(),
        views=views,
        max_result_rows=JOBS_MAX_RESULT_ROWS,
    )
