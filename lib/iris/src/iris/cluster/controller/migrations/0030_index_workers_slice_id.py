# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``idx_workers_slice_id`` on ``workers(slice_id)``.

The orphan-slice prune (``pruner.find_prunable_slice``) asks "does any worker row
still reference this slice?" via ``workers.slice_id`` for each candidate slice, so
that column needs an index. Fresh DBs get it from ``schema.py``'s baseline
``create_all``; this delta backfills it onto already-baselined prod DBs.

Idempotent: ``CREATE INDEX IF NOT EXISTS`` is a no-op if the index is already
present, so a crash mid-run is safe to retry.
"""


def migrate(raw_conn) -> None:
    raw_conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_slice_id ON workers (slice_id)")
    raw_conn.commit()
