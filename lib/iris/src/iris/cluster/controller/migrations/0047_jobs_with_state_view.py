# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the ``jobs_with_state`` view.

The view exposes every ``jobs`` column plus a derived ``state`` computed
from the per-job ``tasks`` aggregate, using the same formula as
``iris.cluster.controller.job_state.compute_job_state``. The column
``jobs.state`` itself is left in place — it remains the indexed fast path
for filtering, and is maintained by ``ControllerTransitions._recompute_job_state``.

The view is the canonical read path when a caller wants a derivation that
cannot lag behind a task transition (e.g. dashboards / CLI tooling that
read across the recompute boundary). For the per-tick controller hot path,
direct ``jobs.state`` reads remain fastest.
"""

import sqlite3

from iris.cluster.controller.job_state import (
    DROP_JOBS_WITH_STATE_VIEW_SQL,
    JOBS_WITH_STATE_VIEW_SQL,
)


def migrate(conn: sqlite3.Connection) -> None:
    # Drop first in case a prior partial install left a stale definition.
    conn.execute(DROP_JOBS_WITH_STATE_VIEW_SQL)
    conn.execute(JOBS_WITH_STATE_VIEW_SQL)
