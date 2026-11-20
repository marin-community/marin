# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist the resolved task termination grace for existing and new jobs."""


def migrate(raw_conn) -> None:
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(job_config)").fetchall()}
    if "termination_grace_ms" not in columns:
        raw_conn.execute("ALTER TABLE job_config ADD COLUMN termination_grace_ms INTEGER NOT NULL DEFAULT 10000")
