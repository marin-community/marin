# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Force-requeue any coscheduled job whose tasks are split across slices.

Pre-fix versions of ``ControllerTransitions.apply_state_updates`` /
``apply_direct_provider_updates`` / ``_remove_failed_worker`` did not
cascade transient task failures to the rest of a coscheduled job. When
one task hit a TPU-init failure (or its host died) the failed task
returned to PENDING alone; the scheduler's
``_find_coscheduled_assignments`` then placed the retry on **any**
worker with free capacity, including a worker on a different ``tpu-name``
group. The job ended up split across two physical TPU slices, the SPMD
collective could not form, and the run hung or crashed.

The code-side fix in ``transitions.py`` prevents new splits. This
migration heals any coscheduled job that's *currently* split:

  - find every coscheduled job whose active-task workers span more than
    one ``md_tpu_name`` value;
  - decommit each such task's resources from its worker
    (``committed_cpu_millicores`` / ``committed_mem_bytes`` /
    ``committed_gpu`` / ``committed_tpu``);
  - mark each in-flight attempt PREEMPTED with ``finished_at_ms`` set;
  - reset each task to PENDING with ``current_worker_id`` cleared.

After the controller starts, the next scheduling cycle re-coschedules
the entire job atomically onto a single ``tpu-name`` group via the
fixed ``_find_coscheduled_assignments`` path.

Reservation-holder jobs are skipped: they don't commit worker resources
and never participate in a slice.

Idempotent — once split jobs are healed, the WHERE clause matches
nothing on subsequent runs.
"""

import sqlite3

# Mirror of iris.cluster.controller.db.ACTIVE_TASK_STATES.
_ACTIVE = (2, 3, 9)  # BUILDING, RUNNING, ASSIGNED
_TASK_STATE_PENDING = 1
_TASK_STATE_PREEMPTED = 10
_RECONCILE_REASON = "Reconciled: requeued to heal split-slice coscheduled job"


def migrate(conn: sqlite3.Connection) -> None:
    active_placeholders = ",".join("?" * len(_ACTIVE))

    split_job_ids = [
        row[0]
        for row in conn.execute(
            f"""
            SELECT j.job_id
            FROM jobs j
            JOIN job_config jc ON jc.job_id = j.job_id
            JOIN tasks t ON t.job_id = j.job_id
            JOIN workers w ON w.worker_id = t.current_worker_id
            WHERE jc.has_coscheduling = 1
              AND j.is_reservation_holder = 0
              AND t.state IN ({active_placeholders})
              AND w.md_tpu_name != ''
            GROUP BY j.job_id
            HAVING COUNT(DISTINCT w.md_tpu_name) > 1
            """,
            _ACTIVE,
        ).fetchall()
    ]

    if not split_job_ids:
        return

    job_placeholders = ",".join("?" * len(split_job_ids))
    now_ms = int(conn.execute("SELECT CAST(strftime('%s','now') AS INTEGER) * 1000").fetchone()[0])

    # Per-task resource decommit. We must subtract from each worker exactly
    # what was committed at dispatch time, sourced from job_config (the same
    # spec workers.commit_resources used originally). Using SUM over an
    # aggregated UPDATE keeps it one statement per resource type even when
    # multiple tasks of a split job land on the same worker.
    conn.execute(
        f"""
        UPDATE workers
        SET committed_cpu_millicores = MAX(
              0,
              committed_cpu_millicores - COALESCE((
                SELECT SUM(jc.res_cpu_millicores)
                FROM tasks t
                JOIN job_config jc ON jc.job_id = t.job_id
                WHERE t.current_worker_id = workers.worker_id
                  AND t.job_id IN ({job_placeholders})
                  AND t.state IN ({active_placeholders})
              ), 0)
            ),
            committed_mem_bytes = MAX(
              0,
              committed_mem_bytes - COALESCE((
                SELECT SUM(jc.res_memory_bytes)
                FROM tasks t
                JOIN job_config jc ON jc.job_id = t.job_id
                WHERE t.current_worker_id = workers.worker_id
                  AND t.job_id IN ({job_placeholders})
                  AND t.state IN ({active_placeholders})
              ), 0)
            ),
            committed_tpu = MAX(
              0,
              committed_tpu - COALESCE((
                SELECT SUM(CAST(json_extract(jc.res_device_json, '$.tpu.count') AS INTEGER))
                FROM tasks t
                JOIN job_config jc ON jc.job_id = t.job_id
                WHERE t.current_worker_id = workers.worker_id
                  AND t.job_id IN ({job_placeholders})
                  AND t.state IN ({active_placeholders})
                  AND json_extract(jc.res_device_json, '$.tpu.count') IS NOT NULL
              ), 0)
            ),
            committed_gpu = MAX(
              0,
              committed_gpu - COALESCE((
                SELECT SUM(CAST(json_extract(jc.res_device_json, '$.gpu.count') AS INTEGER))
                FROM tasks t
                JOIN job_config jc ON jc.job_id = t.job_id
                WHERE t.current_worker_id = workers.worker_id
                  AND t.job_id IN ({job_placeholders})
                  AND t.state IN ({active_placeholders})
                  AND json_extract(jc.res_device_json, '$.gpu.count') IS NOT NULL
              ), 0)
            )
        WHERE worker_id IN (
          SELECT DISTINCT current_worker_id FROM tasks
          WHERE job_id IN ({job_placeholders})
            AND state IN ({active_placeholders})
            AND current_worker_id IS NOT NULL
        )
        """,
        (
            *split_job_ids,
            *_ACTIVE,
            *split_job_ids,
            *_ACTIVE,
            *split_job_ids,
            *_ACTIVE,
            *split_job_ids,
            *_ACTIVE,
            *split_job_ids,
            *_ACTIVE,
        ),
    )

    # Mark in-flight attempts terminal.
    conn.execute(
        f"""
        UPDATE task_attempts
        SET state = ?,
            finished_at_ms = COALESCE(finished_at_ms, ?),
            error = COALESCE(error, ?)
        WHERE state IN ({active_placeholders})
          AND task_id IN (
            SELECT task_id FROM tasks
            WHERE job_id IN ({job_placeholders}) AND state IN ({active_placeholders})
          )
        """,
        (_TASK_STATE_PREEMPTED, now_ms, _RECONCILE_REASON, *_ACTIVE, *split_job_ids, *_ACTIVE),
    )

    # Reset tasks so the scheduler picks them up; clear worker so committed_*
    # accounting can't drift back.
    conn.execute(
        f"""
        UPDATE tasks
        SET state = ?,
            current_worker_id = NULL,
            current_worker_address = NULL,
            error = NULL,
            finished_at_ms = NULL
        WHERE job_id IN ({job_placeholders})
          AND state IN ({active_placeholders})
        """,
        (_TASK_STATE_PENDING, *split_job_ids, *_ACTIVE),
    )
