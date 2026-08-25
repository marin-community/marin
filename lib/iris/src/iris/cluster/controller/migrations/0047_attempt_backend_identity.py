# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add per-attempt backend identity and terminal reason to ``task_attempts``.

Persist the backend object an attempt owned — the Kubernetes pod name, pod UID,
and node — plus a bounded terminal-cause string, so a past failed attempt can be
described after its pod is garbage-collected. This closes the cw-us-east-02a
blind spot (#7542): a pod wedged in ``Init:Error`` on a stage-workdir bundle-fetch
404 surfaced only as a generic ``Pending``, and the pod name was derivable only by
reading ``_pod_name`` in Python.

Empty string for pre-migration rows and for the RPC-worker backend (which
identifies an attempt by ``worker_id``). Idempotent: each column add no-ops when it
already exists (a fresh DB has them from the baseline schema).
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    for column in ("pod_name", "pod_uid", "node_name", "terminal_reason"):
        if not _has_column(raw_conn, "task_attempts", column):
            raw_conn.execute(f"ALTER TABLE task_attempts ADD COLUMN {column} VARCHAR NOT NULL DEFAULT ''")
