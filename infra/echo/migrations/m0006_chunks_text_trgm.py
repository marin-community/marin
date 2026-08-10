# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Index chunks.text for fast case-insensitive substring (grep) scans.

`grep` matches with `text ILIKE '%pattern%'`, which neither the full-text GIN index nor a
btree can serve, so it sequentially scanned every chunk (~1s on the current corpus). A
pg_trgm GIN index makes the substring match indexable.
"""

import sqlalchemy

DDL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_chunks_text_trgm ON chunks USING gin (text gin_trgm_ops);
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
