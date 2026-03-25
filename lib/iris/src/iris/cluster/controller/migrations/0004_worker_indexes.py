# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
DROP TRIGGER IF EXISTS trg_txn_log_retention;
CREATE TRIGGER IF NOT EXISTS trg_txn_log_retention
AFTER INSERT ON txn_log
WHEN (SELECT COUNT(*) FROM txn_log) > 1100
BEGIN
  DELETE FROM txn_log WHERE id <= (
    SELECT id FROM txn_log ORDER BY id DESC LIMIT 1 OFFSET 1000
  );
END;

CREATE INDEX IF NOT EXISTS idx_workers_healthy_active ON workers(healthy, active);
"""
    )
