# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clamp legacy ``preemption_count`` / ``max_retries_preemption`` rows to the cap.

Historical bug: clients passed ``max_retries_preemption = INT32_MAX`` to mean
"retry forever". The sibling-termination path in ``_terminate_coscheduled_siblings``
writes ``preemption_count = max_retries_preemption + 1`` as a tombstone, which
overflowed int32 (2^31 = 2_147_483_648) and tripped ``JobStatus`` serialization
in ``ListJobs`` (``Value out of range: 6_442_450_944`` — 3 sibling tasks summed).

The controller now caps ``max_retries_preemption`` at 1000 on submission. This
migration retroactively clamps any existing rows so the ``ListJobs`` RPC stops
failing on the back catalog.

Clamps four columns to ``MAX_RETRIES_PREEMPTION_CAP`` (1000):
  - ``tasks.preemption_count``
  - ``tasks.max_retries_preemption``
  - ``tasks.failure_count`` (same overflow class — defensive, no known offenders)
  - ``job_config.max_retries_preemption``
"""

import sqlite3

CAP = 1000


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE tasks SET preemption_count = ? WHERE preemption_count > ?", (CAP, CAP))
    conn.execute("UPDATE tasks SET max_retries_preemption = ? WHERE max_retries_preemption > ?", (CAP, CAP))
    conn.execute("UPDATE tasks SET failure_count = ? WHERE failure_count > ?", (CAP, CAP))
    conn.execute(
        "UPDATE job_config SET max_retries_preemption = ? WHERE max_retries_preemption > ?",
        (CAP, CAP),
    )
