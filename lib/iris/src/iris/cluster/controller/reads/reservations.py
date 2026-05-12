# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reservation read helpers (SA Core expression language).

All queries use ``select(table.c.col, ...)`` rather than ``text("SELECT
...")``. TypeDecorators on schema_v2 columns decode worker_id to WorkerId
on read.

Return shapes:

* ``list_claims`` — ``dict[WorkerId, ReservationClaim]``
* ``get_claim_for_worker`` — ``tuple[str, int] | None``
* ``list_claims_for_job`` — ``list[tuple[WorkerId, int]]``
* ``count_claims_for_job`` — ``int``
* ``get_last_submission_ms`` — ``int``
"""

from sqlalchemy import bindparam, func, select

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import meta_table, reservation_claims_table
from iris.cluster.types import WorkerId

# ---------------------------------------------------------------------------
# Reservation claims
# ---------------------------------------------------------------------------

LIST_CLAIMS_QUERY = select(
    reservation_claims_table.c.worker_id,
    reservation_claims_table.c.job_id,
    reservation_claims_table.c.entry_idx,
)

GET_CLAIM_FOR_WORKER_QUERY = select(reservation_claims_table.c.job_id, reservation_claims_table.c.entry_idx).where(
    reservation_claims_table.c.worker_id == bindparam("worker_id")
)

LIST_CLAIMS_FOR_JOB_QUERY = select(reservation_claims_table.c.worker_id, reservation_claims_table.c.entry_idx).where(
    reservation_claims_table.c.job_id == bindparam("job_id")
)

COUNT_CLAIMS_FOR_JOB_QUERY = (
    select(func.count().label("n"))
    .select_from(reservation_claims_table)
    .where(reservation_claims_table.c.job_id == bindparam("job_id"))
)


def list_claims(tx: Tx) -> dict:
    """Return ``{WorkerId: ReservationClaim}`` for every reservation claim.

    :class:`ReservationClaim` is imported lazily to avoid an import cycle
    with ``controller/transitions.py``.
    """
    from iris.cluster.controller.transitions import ReservationClaim

    rows = tx.execute(LIST_CLAIMS_QUERY).all()
    return {
        row.worker_id: ReservationClaim(
            job_id=str(row.job_id),
            entry_idx=int(row.entry_idx),
        )
        for row in rows
    }


def get_claim_for_worker(tx: Tx, worker_id: WorkerId) -> tuple[str, int] | None:
    """Return ``(job_id, entry_idx)`` for ``worker_id``'s claim, or None."""
    row = tx.execute(GET_CLAIM_FOR_WORKER_QUERY, {"worker_id": worker_id}).first()
    if row is None:
        return None
    return str(row.job_id), int(row.entry_idx)


def list_claims_for_job(tx: Tx, job_id_wire: str) -> list[tuple[WorkerId, int]]:
    """Return ``[(worker_id, entry_idx), ...]`` for every claim against ``job_id_wire``."""
    rows = tx.execute(LIST_CLAIMS_FOR_JOB_QUERY, {"job_id": job_id_wire}).all()
    return [(row.worker_id, int(row.entry_idx)) for row in rows]


def count_claims_for_job(tx: Tx, job_id_wire: str) -> int:
    """Return the number of reservation_claims rows for ``job_id_wire``."""
    result = tx.execute(COUNT_CLAIMS_FOR_JOB_QUERY, {"job_id": job_id_wire}).scalar()
    return int(result or 0)


# ---------------------------------------------------------------------------
# Last-submission counter
# ---------------------------------------------------------------------------

GET_LAST_SUBMISSION_QUERY = select(meta_table.c.value).where(meta_table.c.key == "last_submission_ms")


def get_last_submission_ms(tx: Tx) -> int:
    """Return ``meta.last_submission_ms``, or 0 if absent."""
    row = tx.execute(GET_LAST_SUBMISSION_QUERY).first()
    return int(row.value) if row is not None else 0
