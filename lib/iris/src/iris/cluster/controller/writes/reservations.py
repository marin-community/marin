# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for ``reservation_claims`` and the ``meta`` last-submission counter.

Stage M2 of the SA Core migration: replaces raw ``text("INSERT/UPDATE/DELETE ...")``
strings with SA Core expression-language constructs. TypeDecorators handle
all bind-side conversions automatically.
"""

from sqlalchemy import delete, insert, select, update

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import meta_table, reservation_claims_table
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import WorkerId

_LAST_SUBMISSION_KEY = "last_submission_ms"


@writes_to(reservation_claims_table)
def replace_claims(tx: Tx, claims: dict[WorkerId, tuple[str, int]]) -> None:
    """Replace the entire ``reservation_claims`` table with ``claims``."""
    tx.execute(delete(reservation_claims_table))
    if not claims:
        return
    tx.execute(
        insert(reservation_claims_table),
        [
            {"worker_id": worker_id, "job_id": job_id, "entry_idx": entry_idx}
            for worker_id, (job_id, entry_idx) in claims.items()
        ],
    )


@writes_to(meta_table)
def next_submission_ms(tx: Tx, submitted_ms: int) -> int:
    """Bump ``meta.last_submission_ms`` to ``max(submitted_ms, last + 1)``.

    Returns the new ``last_submission_ms``. Used by the reservation
    holder path to guarantee strictly-monotone submission timestamps.
    """
    row = tx.execute(select(meta_table.c.value).where(meta_table.c.key == _LAST_SUBMISSION_KEY)).fetchone()
    last_submission_ms = int(row[0]) if row is not None else 0
    effective_submission_ms = max(submitted_ms, last_submission_ms + 1)
    if row is None:
        tx.execute(insert(meta_table).values(key=_LAST_SUBMISSION_KEY, value=effective_submission_ms))
    else:
        tx.execute(
            update(meta_table).where(meta_table.c.key == _LAST_SUBMISSION_KEY).values(value=effective_submission_ms)
        )
    return effective_submission_ms
