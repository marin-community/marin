# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the ``users`` table and its foreign-key anchors on ``jobs`` / ``user_budgets``.

Roles are now resolved from an in-memory, config-derived ``RolePolicy`` (see
``controller/auth.py``): cluster config is the sole source of truth, so the
``users`` table — which held only ``(user_id, role)`` and existed only as an FK
anchor — is dead, along with the ``_reconcile_admin_grants`` projection that kept
it in sync.

FK enforcement is ON for every controller connection (``PRAGMA foreign_keys=ON``
in ``db.py``), so ``users`` cannot simply be dropped while ``jobs.user_id`` and
``user_budgets.user_id`` still carry ``REFERENCES users(user_id)``: the implicit
row-delete a ``DROP TABLE`` performs would raise an FK violation, and even if it
didn't, those columns' FK text would then dangle at a non-existent parent and fail
the next insert. SQLite cannot drop a constraint in place, so we follow the
documented table-rebuild procedure (the same shape as ``0028``): rebuild ``jobs``
and ``user_budgets`` without the ``users`` FK (``jobs`` keeps its ``parent_job_id``
self-FK and every index), then drop ``users``. The rebuild runs with
``foreign_keys=OFF`` so dropping the old ``jobs`` does not cascade into its child
tables (``tasks`` / ``job_config`` / ``endpoints`` / …); those keep their rows and
their by-name FK to ``jobs``, which re-binds to the rebuilt table after the rename.

Idempotent: a fresh DB's baseline already builds ``jobs`` / ``user_budgets`` with
no ``users`` FK and never creates ``users``, so the guard short-circuits and the
migration no-ops (also safe to retry after a crash).
"""

# jobs, rebuilt without ``FOREIGN KEY(user_id) REFERENCES users``. Matches the
# baseline DDL byte-for-byte (SQLAlchemy-emitted) minus that one constraint; the
# ``parent_job_id`` self-FK and all indexes are preserved.
_JOBS_NEW_DDL = """
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
	backend_id VARCHAR DEFAULT '' NOT NULL,
	cluster VARCHAR DEFAULT 'local' NOT NULL,
	PRIMARY KEY (job_id),
	FOREIGN KEY(parent_job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
)
"""

_JOBS_COLUMNS = (
    "job_id, user_id, parent_job_id, root_job_id, depth, state, submitted_at_ms, "
    "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
    "error, exit_code, num_tasks, name, backend_id, cluster"
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

# user_budgets, rebuilt without ``FOREIGN KEY(user_id) REFERENCES users``.
_USER_BUDGETS_NEW_DDL = """
CREATE TABLE user_budgets_new (
	user_id VARCHAR NOT NULL,
	budget_limit INTEGER DEFAULT '0' NOT NULL,
	max_band INTEGER DEFAULT '2' NOT NULL,
	updated_at_ms INTEGER NOT NULL,
	PRIMARY KEY (user_id)
)
"""

_USER_BUDGETS_COLUMNS = "user_id, budget_limit, max_band, updated_at_ms"


def _references_users(raw_conn, table: str) -> bool:
    # PRAGMA foreign_key_list columns: (id, seq, table, from, to, on_update, on_delete, match)
    return any(str(row[2]) == "users" for row in raw_conn.execute(f"PRAGMA foreign_key_list({table})").fetchall())


def migrate(raw_conn) -> None:
    needs_rebuild = _references_users(raw_conn, "jobs") or _references_users(raw_conn, "user_budgets")
    if not needs_rebuild:
        # Fresh (or already-migrated) DB: no users FK anywhere. Drop a stray users
        # table if one somehow lingers with no referencing FK, then no-op.
        raw_conn.execute("DROP TABLE IF EXISTS users")
        return

    # foreign_keys can only be toggled outside a transaction; close any implicit
    # one first. With it OFF, dropping the old jobs does not cascade into children.
    raw_conn.commit()
    raw_conn.execute("PRAGMA foreign_keys=OFF")
    raw_conn.execute("BEGIN IMMEDIATE")
    try:
        # jobs: rebuild without the users FK, preserving rows, the self-FK, indexes.
        raw_conn.execute(_JOBS_NEW_DDL)
        raw_conn.execute(f"INSERT INTO jobs_new ({_JOBS_COLUMNS}) SELECT {_JOBS_COLUMNS} FROM jobs")
        raw_conn.execute("DROP TABLE jobs")
        raw_conn.execute("ALTER TABLE jobs_new RENAME TO jobs")
        for index_sql in _JOBS_INDEXES:
            raw_conn.execute(index_sql)

        # user_budgets: rebuild without the users FK.
        raw_conn.execute(_USER_BUDGETS_NEW_DDL)
        raw_conn.execute(
            f"INSERT INTO user_budgets_new ({_USER_BUDGETS_COLUMNS}) "
            f"SELECT {_USER_BUDGETS_COLUMNS} FROM user_budgets"
        )
        raw_conn.execute("DROP TABLE user_budgets")
        raw_conn.execute("ALTER TABLE user_budgets_new RENAME TO user_budgets")

        # Nothing references users now — drop it.
        raw_conn.execute("DROP TABLE IF EXISTS users")

        # Verify the rebuild left no dangling references (e.g. the jobs self-FK).
        violations = raw_conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"0040_drop_users left foreign-key violations: {violations}")
        raw_conn.commit()
    except Exception:
        raw_conn.execute("ROLLBACK")
        raise
    finally:
        # Restore enforcement for the recorded-migration write and all later use.
        raw_conn.execute("PRAGMA foreign_keys=ON")
