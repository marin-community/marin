# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store the band a job resolves to, replacing ``PRIORITY_BAND_INHERIT`` (0).

Submit used to persist the request's band verbatim, so a job that inherited its band
stored 0 and every reader re-derived a band from it under its own rule. The band is now
resolved once at submit — an explicit request, else the parent's band, else INTERACTIVE —
which leaves the rows written before that change as the only source of 0.

Two repairs, both applying that same rule:

- ``job_config.priority_band = 0`` takes the nearest non-zero band up its
  ``parent_job_id`` chain, or INTERACTIVE when the whole chain is 0.
- A PENDING task takes its job's band. Those rows were stamped with a flat INTERACTIVE
  default that ignored inheritance, so a child of a BATCH or PRODUCTION job carries the
  wrong sort key in ``idx_tasks_pending``. Tasks past PENDING keep the band the
  scheduler stamped at assignment (which may be a budget downgrade).
"""

_INTERACTIVE = 2
_TASK_STATE_PENDING = 1


def migrate(raw_conn) -> None:
    # Materialize the resolution before writing: the recursive walk reads the same
    # column the UPDATE rewrites.
    raw_conn.execute("DROP TABLE IF EXISTS _resolved_bands")
    raw_conn.execute(
        f"""
        CREATE TEMP TABLE _resolved_bands AS
        WITH RECURSIVE chain(input_id, current_band, parent_id) AS (
            SELECT jobs.job_id, job_config.priority_band, jobs.parent_job_id
            FROM jobs JOIN job_config ON job_config.job_id = jobs.job_id
            WHERE job_config.priority_band = 0
            UNION ALL
            SELECT chain.input_id, job_config.priority_band, jobs.parent_job_id
            FROM chain
            JOIN jobs ON jobs.job_id = chain.parent_id
            JOIN job_config ON job_config.job_id = jobs.job_id
            WHERE chain.current_band = 0
        )
        SELECT jobs.job_id AS job_id,
               COALESCE(
                   (SELECT chain.current_band FROM chain
                    WHERE chain.input_id = jobs.job_id AND chain.current_band != 0),
                   {_INTERACTIVE}
               ) AS band
        FROM jobs JOIN job_config ON job_config.job_id = jobs.job_id
        WHERE job_config.priority_band = 0
        """
    )
    raw_conn.execute(
        """
        UPDATE job_config
        SET priority_band = (SELECT band FROM _resolved_bands WHERE _resolved_bands.job_id = job_config.job_id)
        WHERE job_id IN (SELECT job_id FROM _resolved_bands)
        """
    )
    raw_conn.execute("DROP TABLE _resolved_bands")
    raw_conn.execute(
        f"""
        UPDATE tasks
        SET priority_band = (SELECT job_config.priority_band FROM job_config
                             WHERE job_config.job_id = tasks.job_id)
        WHERE state = {_TASK_STATE_PENDING}
          AND EXISTS (
              SELECT 1 FROM job_config
              WHERE job_config.job_id = tasks.job_id
                AND job_config.priority_band != tasks.priority_band
          )
        """
    )
