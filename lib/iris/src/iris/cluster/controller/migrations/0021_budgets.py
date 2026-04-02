# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Add priority_band column to tasks and create user_budgets table."""
    # Add priority_band column (idempotent check).
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "priority_band" not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN priority_band INTEGER NOT NULL DEFAULT 2")

    conn.executescript(
        """
        -- User budgets table
        CREATE TABLE IF NOT EXISTS user_budgets (
            user_id TEXT PRIMARY KEY REFERENCES users(user_id),
            budget_limit INTEGER NOT NULL DEFAULT 0,
            max_band INTEGER NOT NULL DEFAULT 2,
            updated_at_ms INTEGER NOT NULL
        );

        -- Seed budget rows for existing users
        INSERT OR IGNORE INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms)
        SELECT user_id, 0, 2, created_at_ms FROM users;

        -- Rebuild pending-tasks index to include priority_band
        DROP INDEX IF EXISTS idx_tasks_pending;
        CREATE INDEX idx_tasks_pending ON tasks(state,
            priority_band ASC,
            priority_neg_depth ASC,
            priority_root_submitted_ms ASC,
            submitted_at_ms ASC,
            priority_insertion ASC
        );
        """
    )
