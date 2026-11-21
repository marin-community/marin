# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ``tags_json`` to ``job_config``.

``tags_json`` holds a job's ``JobTags`` as a JSON object. Existing rows stay
NULL (untagged).

Idempotent: re-run from scratch if the controller crashes mid-migration.
"""


def migrate(raw_conn) -> None:
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(job_config)").fetchall()}
    if "tags_json" not in columns:
        raw_conn.execute("ALTER TABLE job_config ADD COLUMN tags_json TEXT")
