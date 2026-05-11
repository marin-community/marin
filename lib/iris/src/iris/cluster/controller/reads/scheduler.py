# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-tick read helpers.

Named SA Core ``text`` constants and helper functions for the hot
per-tick queries driven by the scheduler loop. Stage 5 of the
SQLAlchemy Core migration introduces this module with the
``_jobs_with_reservations`` port (the perf canary); later stages add
the heavier reads (``resource_usage_by_worker``,
``reconcile_rows_for_workers``).

The hot-path queries use ``text(...)`` with bound parameters rather
than ``select(...)`` Core ORM expressions. The two forms produce
identical SQL but ``text()`` avoids ~370 µs/call of per-call statement
compilation overhead that ``select()`` reintroduces even after caching.
``bindparams(expanding=True)`` lets one compiled statement service
``IN (?, ?, ...)`` calls with varying list lengths.
"""

from sqlalchemy import bindparam, text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema import JobReservationRow
from iris.cluster.types import JobName

# Slim 2-column projection for the per-tick reservation-claim recomputation.
# Filters on ``jobs.has_reservation = 1`` (partial index
# ``idx_jobs_has_reservation``) and joins ``job_config`` solely to pull
# ``reservation_json``. The expanding ``:states`` bind lets SA reuse the
# compiled statement across different state-tuple lengths.
_JOBS_WITH_RESERVATIONS_SQL = text(
    "SELECT j.job_id AS job_id, jc.reservation_json AS reservation_json "
    "FROM jobs j JOIN job_config jc ON j.job_id = jc.job_id "
    "WHERE j.state IN :states AND j.has_reservation = 1"
).bindparams(bindparam("states", expanding=True))


def jobs_with_reservations(tx: Tx, states: tuple[int, ...]) -> list[JobReservationRow]:
    """Fetch ``(job_id, reservation_json)`` for reservation-holding jobs in ``states``."""
    rows = tx.execute(_JOBS_WITH_RESERVATIONS_SQL, {"states": list(states)}).all()
    return [
        JobReservationRow(job_id=JobName.from_wire(row.job_id), reservation_json=row.reservation_json) for row in rows
    ]
