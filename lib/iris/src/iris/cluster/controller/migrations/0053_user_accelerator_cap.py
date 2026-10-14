# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the hard per-identity accelerator cap to ``user_budgets``.

``max_accelerators`` is the most accelerators (GPU + TPU chips) a budget identity's
active tasks may hold at once; 0 means no cap. ``cap_set_at_ms`` records when an
admin set the cap at runtime (``iris user cap set``); a non-NULL value tells the
startup tier reconcile not to overwrite the cap from cluster config.

Existing rows get no cap and no runtime marker, so the next startup may seed them
from config. Idempotent: each column add no-ops when the column already exists.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    if not _has_column(raw_conn, "user_budgets", "max_accelerators"):
        raw_conn.execute("ALTER TABLE user_budgets ADD COLUMN max_accelerators INTEGER NOT NULL DEFAULT 0")
    if not _has_column(raw_conn, "user_budgets", "cap_set_at_ms"):
        raw_conn.execute("ALTER TABLE user_budgets ADD COLUMN cap_set_at_ms INTEGER")
