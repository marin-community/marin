# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for ``reservation_claims`` and the ``meta`` last-submission counter.

Stage 11 of the SA Core migration. Ports :meth:`ReservationStore.replace_claims`
and :meth:`ReservationStore.next_submission_ms` into module-level functions
taking a :class:`iris.cluster.controller.db_v2.Tx`.
"""

from sqlalchemy import text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import meta_table, reservation_claims_table
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import WorkerId

_DELETE_CLAIMS_SQL = text("DELETE FROM reservation_claims")
_INSERT_CLAIM_SQL = text(
    "INSERT INTO reservation_claims(worker_id, job_id, entry_idx) " "VALUES (:worker_id, :job_id, :entry_idx)"
)

_LAST_SUBMISSION_KEY = "last_submission_ms"

_GET_LAST_SUBMISSION_SQL = text("SELECT value FROM meta WHERE key = :key")
_INSERT_LAST_SUBMISSION_SQL = text("INSERT INTO meta(key, value) VALUES (:key, :value)")
_UPDATE_LAST_SUBMISSION_SQL = text("UPDATE meta SET value = :value WHERE key = :key")


@writes_to(reservation_claims_table)
def replace_claims(tx: Tx, claims: dict[WorkerId, tuple[str, int]]) -> None:
    """Replace the entire ``reservation_claims`` table with ``claims``."""
    tx.execute(_DELETE_CLAIMS_SQL)
    for worker_id, (job_id, entry_idx) in claims.items():
        tx.execute(
            _INSERT_CLAIM_SQL,
            {"worker_id": str(worker_id), "job_id": job_id, "entry_idx": entry_idx},
        )


@writes_to(meta_table)
def next_submission_ms(tx: Tx, submitted_ms: int) -> int:
    """Bump ``meta.last_submission_ms`` to ``max(submitted_ms, last + 1)``.

    Returns the new ``last_submission_ms``. Used by the reservation
    holder path to guarantee strictly-monotone submission timestamps.
    """
    row = tx.execute(_GET_LAST_SUBMISSION_SQL, {"key": _LAST_SUBMISSION_KEY}).fetchone()
    last_submission_ms = int(row[0]) if row is not None else 0
    effective_submission_ms = max(submitted_ms, last_submission_ms + 1)
    if row is None:
        tx.execute(_INSERT_LAST_SUBMISSION_SQL, {"key": _LAST_SUBMISSION_KEY, "value": effective_submission_ms})
    else:
        tx.execute(_UPDATE_LAST_SUBMISSION_SQL, {"key": _LAST_SUBMISSION_KEY, "value": effective_submission_ms})
    return effective_submission_ms
