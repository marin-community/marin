# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # _task_summaries_for_jobs (service.py) runs
    #     SELECT job_id, state, COUNT(*), SUM(failure_count), SUM(preemption_count)
    #     FROM tasks WHERE job_id IN (...) GROUP BY job_id, state
    # on every ListJobs and GetJobStatus call. The existing
    # idx_tasks_job_failures (job_id, failure_count, preemption_count) lacks
    # `state` and so SQLite has to read the base row for every matched task
    # to get the state column. idx_tasks_job_state (job_id, state) covers the
    # filter+GROUP BY keys but not the SUM targets.
    #
    # This index covers the whole query: leading (job_id, state) serves
    # WHERE + GROUP BY, and the trailing (failure_count, preemption_count)
    # columns let SQLite satisfy the SUMs directly from the index without
    # touching the tasks heap.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_job_state_counts "
        "ON tasks(job_id, state, failure_count, preemption_count)"
    )
