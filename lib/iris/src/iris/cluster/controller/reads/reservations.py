# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reservation read helpers (SA Core port).

Named ``text(...)`` SQL constants and small helpers for reservation reads.
Today these live as ad-hoc SQL in
:func:`iris.cluster.controller.controller._read_reservation_claims` and a
small read on the ``meta`` table inside
:meth:`iris.cluster.controller.stores.ReservationStore.next_submission_ms`.
Stage 10 of the SA Core migration introduces this module alongside the
legacy paths; parity tests in ``tests/cluster/controller/test_reads_reservations.py``
assert the two paths return equal results.
"""

from sqlalchemy import text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.types import WorkerId

# ---------------------------------------------------------------------------
# Reservation claims
# ---------------------------------------------------------------------------

_LIST_CLAIMS_SQL = text("SELECT worker_id, job_id, entry_idx FROM reservation_claims")

_GET_CLAIM_FOR_WORKER_SQL = text("SELECT job_id, entry_idx FROM reservation_claims WHERE worker_id = :wid")

_LIST_CLAIMS_FOR_JOB_SQL = text("SELECT worker_id, entry_idx FROM reservation_claims WHERE job_id = :jid")


def list_claims(tx: Tx) -> dict:
    """Return ``{worker_id: ReservationClaim}`` for every reservation claim.

    Mirrors :func:`iris.cluster.controller.controller._read_reservation_claims`.
    :class:`ReservationClaim` is imported lazily to avoid the
    ``controller/transitions.py`` import cycle.
    """
    from iris.cluster.controller.transitions import ReservationClaim

    rows = tx.execute(_LIST_CLAIMS_SQL).all()
    return {
        WorkerId(str(row.worker_id)): ReservationClaim(
            job_id=str(row.job_id),
            entry_idx=int(row.entry_idx),
        )
        for row in rows
    }


def get_claim_for_worker(tx: Tx, worker_id: WorkerId) -> tuple[str, int] | None:
    """Return ``(job_id, entry_idx)`` for ``worker_id``'s claim, or None."""
    row = tx.execute(_GET_CLAIM_FOR_WORKER_SQL, {"wid": str(worker_id)}).first()
    if row is None:
        return None
    return str(row.job_id), int(row.entry_idx)


def list_claims_for_job(tx: Tx, job_id_wire: str) -> list[tuple[WorkerId, int]]:
    """Return ``[(worker_id, entry_idx), ...]`` for every claim against ``job_id_wire``."""
    rows = tx.execute(_LIST_CLAIMS_FOR_JOB_SQL, {"jid": job_id_wire}).all()
    return [(WorkerId(str(row.worker_id)), int(row.entry_idx)) for row in rows]


def count_claims_for_job(tx: Tx, job_id_wire: str) -> int:
    """Return the number of reservation_claims rows for ``job_id_wire``."""
    return len(list_claims_for_job(tx, job_id_wire))


# ---------------------------------------------------------------------------
# Last-submission counter
# ---------------------------------------------------------------------------

_GET_LAST_SUBMISSION_SQL = text("SELECT value FROM meta WHERE key = 'last_submission_ms'")


def get_last_submission_ms(tx: Tx) -> int:
    """Return ``meta.last_submission_ms``, or 0 if absent.

    Mirrors the read half of
    :meth:`iris.cluster.controller.stores.ReservationStore.next_submission_ms`.
    The write half (the upsert) stays on the legacy ``ReservationStore`` until
    stage 11.
    """
    row = tx.execute(_GET_LAST_SUBMISSION_SQL).first()
    return int(row.value) if row is not None else 0
