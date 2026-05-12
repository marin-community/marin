# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write helpers for ``workers`` and ``worker_attributes``.

``remove_worker`` declares ``cascades_into=(task_attempts_table,)`` only.
The FK ``ON DELETE CASCADE`` from ``workers.worker_id`` also deletes from
``worker_attributes`` (Projection-owned), but the startup invariant check
forbids declaring that cascade because it would route a mutation around
:class:`WorkerAttrsProjection`. Instead, ``remove_worker`` calls
:meth:`WorkerAttrsProjection.invalidate_for_worker` inline so the cache
update commits atomically with the SQL delete.
"""

from sqlalchemy import delete, update

from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.schema import task_attempts_table, tasks_table, workers_table
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import WorkerId


@writes_to(workers_table, cascades_into=(task_attempts_table,))
def remove_worker(
    tx: Tx,
    worker_id: WorkerId,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
) -> None:
    """Delete a worker row and clear back-references on attempts / tasks.

    ``cascades_into`` records the FK fanout to ``task_attempts``.
    The cascade into ``worker_attributes`` is Projection-owned and therefore
    not declared on the decorator; instead this function calls
    :meth:`WorkerAttrsProjection.invalidate_for_worker` inline so the
    cache update commits under the same write lock as the SQL.

    The pre-emptive ``UPDATE`` on ``task_attempts`` / ``tasks`` sets
    ``current_worker_*`` to NULL before the delete so the row history
    is observable to readers in the same write transaction.
    """
    tx.execute(update(task_attempts_table).where(task_attempts_table.c.worker_id == worker_id).values(worker_id=None))
    tx.execute(update(tasks_table).where(tasks_table.c.current_worker_id == worker_id).values(current_worker_id=None))
    tx.execute(delete(workers_table).where(workers_table.c.worker_id == worker_id))
    worker_attrs.invalidate_for_worker(tx, worker_id)
    tx.register(lambda: health.forget(worker_id))
