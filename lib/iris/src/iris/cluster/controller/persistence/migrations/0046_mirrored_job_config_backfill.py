# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill the ``job_config`` companion for jobs mirrored from a federation peer.

``job_config`` is a 1:1 extension of ``jobs``, and every dashboard read joins the two.
The federation sync used to mirror a child a peer spawned under a received root as a
``jobs`` row alone, so those children were dropped by the join: absent from ListJobs
and not addressable by id, on the one cluster a user watches the whole federation from.

The mirror now writes both rows. This repairs the rows it already created: the peer
reports resources only for jobs it mirrors from here on, so a backfilled row keeps the
column defaults and renders with no resource figures — enough to make a job that ran
visible again, which is the whole of what was lost.
"""


def migrate(raw_conn) -> None:
    raw_conn.execute(
        """
        INSERT INTO job_config (job_id, name)
        SELECT jobs.job_id, jobs.name FROM jobs
        WHERE NOT EXISTS (SELECT 1 FROM job_config WHERE job_config.job_id = jobs.job_id)
        """
    )
