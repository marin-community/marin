# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the reservation apparatus, gracefully migrating any in-flight reservations.

Reservations are gone: ``--reserve`` now becomes a soft ``availability:<variant>``
scheduling hint, holding no capacity and spawning no holder job. A DB upgraded
mid-flight may still carry reservation state, so before stripping the schema this
migration handles the two cases the removed code would otherwise mishandle:

1. **Holder children are deleted** (``DELETE FROM jobs WHERE is_reservation_holder``).
   A holder is a synthetic child that only ever existed to pin capacity. Once the
   ``is_reservation_holder`` filters are gone it would otherwise read as an ordinary
   job — a RUNNING holder would keep consuming its accelerator, a PENDING one could
   be dispatched. Deleting it (``foreign_keys`` is pinned ON, so the cascade — incl.
   the self-referential ``parent_job_id`` FK — removes its tasks/attempts/config and
   any descendants) drops it from the controller's reconcile ``desired`` set, so the
   worker zombie-reaps the still-running process and frees the accelerator.

2. **Real reservations are converted to availability hints.** Each non-holder job
   with a ``reservation_json`` has its reserved accelerator variants folded into the
   job's ``constraints_json`` as soft ``availability:<variant>`` constraints (the same
   conversion the ingestion shim applies to new submissions), preserving the
   zone-steering intent of a job that was mid-flight at upgrade. This is an
   intentional, transitional data migration: it reuses live constraint helpers rather
   than re-encoding the wire format by hand, and is scheduled for removal with the
   rest of the back-compat shim.

Then it strips the schema the controller no longer writes or reads:

* ``jobs.is_reservation_holder`` / ``jobs.has_reservation`` (+ their indexes and the
  ``jobs_is_reservation_holder_check`` CHECK). ``jobs`` is FK-referenced by
  ``job_config``, ``tasks``, ``job_workdir_files`` and self-references via
  ``parent_job_id``; ``is_reservation_holder`` is bound by a CHECK constraint, so an
  in-place ``DROP COLUMN`` is impossible — the table is rebuilt with foreign keys
  off (mirroring ``0027``).
* ``job_config.has_reservation`` / ``job_config.reservation_json`` (+ index). No
  CHECK or inbound FK references these, so they drop in place once the index is gone.
* the ``reservation_claims`` table.

Every step is idempotent: holder deletion re-runs as a no-op once none remain;
conversion deduplicates by constraint key, so a re-run adds nothing; the schema
steps guard on column/table presence. A crash mid-migration is safe to retry.
"""

import json
import logging

from google.protobuf import json_format
from iris.cluster.constraints import availability_constraints_from_reservation
from iris.cluster.controller.codec import constraints_from_json, constraints_to_json
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# Rebuilt ``jobs`` schema, copied verbatim from schema.py's metadata as of this
# migration (sans the two reservation columns and the CHECK). Hardcoded so the
# migration stays a fixed historical snapshot even as schema.py evolves.
_JOBS_DDL = """
CREATE TABLE jobs_new (
    job_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    parent_job_id VARCHAR,
    root_job_id VARCHAR NOT NULL,
    depth INTEGER NOT NULL,
    state INTEGER NOT NULL,
    submitted_at_ms INTEGER NOT NULL,
    root_submitted_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    scheduling_deadline_epoch_ms INTEGER,
    error VARCHAR,
    exit_code INTEGER,
    num_tasks INTEGER NOT NULL,
    name VARCHAR DEFAULT '' NOT NULL,
    PRIMARY KEY (job_id),
    FOREIGN KEY(user_id) REFERENCES users (user_id),
    FOREIGN KEY(parent_job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
)
"""

_JOBS_COLUMNS = (
    "job_id, user_id, parent_job_id, root_job_id, depth, state, "
    "submitted_at_ms, root_submitted_at_ms, started_at_ms, finished_at_ms, "
    "scheduling_deadline_epoch_ms, error, exit_code, num_tasks, name"
)

_JOBS_INDEXES = (
    "CREATE INDEX idx_jobs_parent ON jobs (parent_job_id)",
    "CREATE INDEX idx_jobs_state ON jobs (state, submitted_at_ms DESC)",
    "CREATE INDEX idx_jobs_depth_state ON jobs (depth, state, submitted_at_ms DESC)",
    "CREATE INDEX idx_jobs_user_state ON jobs (user_id, state)",
    "CREATE INDEX idx_jobs_root_depth ON jobs (root_job_id, depth)",
    "CREATE INDEX idx_jobs_depth_submitted ON jobs (depth, submitted_at_ms DESC)",
    "CREATE INDEX idx_jobs_name ON jobs (name)",
)


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def _delete_reservation_holders(raw_conn) -> None:
    """Delete synthetic reservation-holder jobs (and cascade to their descendants).

    Must run before the schema strip (needs ``is_reservation_holder``) and before the
    conversion below (so converted rows never include a holder's config).
    """
    if not _has_column(raw_conn, "jobs", "is_reservation_holder"):
        return
    # PRAGMA foreign_keys takes effect only outside a transaction; it is pinned ON
    # at connect, but set it explicitly so the cascade is robust to ambient state.
    raw_conn.commit()
    raw_conn.execute("PRAGMA foreign_keys=ON")
    raw_conn.execute("DELETE FROM jobs WHERE is_reservation_holder = 1")
    raw_conn.commit()


def _merged_constraints_json(reservation_json: str, constraints_json: str | None) -> str | None:
    """Fold a job's ``reservation_json`` into its ``constraints_json`` as soft hints.

    Returns the new constraints JSON, or ``None`` when there is nothing to add (no
    accelerator entries, or the hint is already present — keeping the step
    idempotent). Malformed reservation JSON is logged and skipped.
    """
    try:
        config = json_format.ParseDict(json.loads(reservation_json), job_pb2.ReservationConfig())
    except (ValueError, json_format.ParseError):
        logger.warning("0029: skipping unparseable reservation_json: %r", reservation_json)
        return None
    available = availability_constraints_from_reservation(config)
    if not available:
        return None
    existing = constraints_from_json(constraints_json)
    existing_keys = {c.key for c in existing}
    fresh = [c for c in available if c.key not in existing_keys]
    if not fresh:
        return None
    return constraints_to_json([c.to_proto() for c in existing] + [c.to_proto() for c in fresh])


def _convert_reservations_to_availability(raw_conn) -> None:
    """Convert each non-holder job's reservation into soft availability constraints."""
    if not _has_column(raw_conn, "job_config", "reservation_json"):
        return
    rows = raw_conn.execute(
        "SELECT job_id, reservation_json, constraints_json FROM job_config WHERE reservation_json IS NOT NULL"
    ).fetchall()
    for job_id, reservation_json, constraints_json in rows:
        merged = _merged_constraints_json(reservation_json, constraints_json)
        if merged is not None:
            raw_conn.execute("UPDATE job_config SET constraints_json = ? WHERE job_id = ?", (merged, job_id))
    raw_conn.commit()


def _rebuild_jobs(raw_conn) -> None:
    """Rebuild ``jobs`` without the reservation columns, indexes, and CHECK."""
    has_reservation_cols = _has_column(raw_conn, "jobs", "is_reservation_holder") or _has_column(
        raw_conn, "jobs", "has_reservation"
    )
    if not has_reservation_cols:
        return

    # Foreign keys must be off during the rebuild so dropping/renaming ``jobs``
    # does not trip the FKs that reference it. SQLite ignores PRAGMA
    # foreign_keys changes inside a transaction, so toggle it outside one.
    raw_conn.commit()
    raw_conn.execute("PRAGMA foreign_keys=OFF")
    try:
        raw_conn.execute("BEGIN IMMEDIATE")
        raw_conn.execute(_JOBS_DDL)
        raw_conn.execute(f"INSERT INTO jobs_new ({_JOBS_COLUMNS}) SELECT {_JOBS_COLUMNS} FROM jobs")
        raw_conn.execute("DROP TABLE jobs")
        raw_conn.execute("ALTER TABLE jobs_new RENAME TO jobs")
        for stmt in _JOBS_INDEXES:
            raw_conn.execute(stmt)
        raw_conn.commit()
    except Exception:
        raw_conn.execute("ROLLBACK")
        raise
    finally:
        raw_conn.execute("PRAGMA foreign_keys=ON")


def _drop_job_config_columns(raw_conn) -> None:
    """Drop the two ``job_config`` reservation columns in place.

    Nothing else references them once the partial index is gone (no CHECK, no
    inbound FK), so a plain ``DROP COLUMN`` suffices.
    """
    raw_conn.execute("DROP INDEX IF EXISTS idx_job_config_has_reservation")
    if _has_column(raw_conn, "job_config", "has_reservation"):
        raw_conn.execute("ALTER TABLE job_config DROP COLUMN has_reservation")
    if _has_column(raw_conn, "job_config", "reservation_json"):
        raw_conn.execute("ALTER TABLE job_config DROP COLUMN reservation_json")


def migrate(raw_conn) -> None:
    # Order matters: delete holders first (so conversion never sees their config),
    # convert real reservations while reservation_json still exists, then strip schema.
    _delete_reservation_holders(raw_conn)
    _convert_reservations_to_availability(raw_conn)
    _rebuild_jobs(raw_conn)
    _drop_job_config_columns(raw_conn)
    raw_conn.execute("DROP TABLE IF EXISTS reservation_claims")
