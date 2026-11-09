# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the selected load of sampled SFT-mix documents from Postgres."""

from typing import Any

from fastapi import FastAPI, HTTPException
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mix_sources (
    load_id TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, family TEXT NOT NULL,
    share DOUBLE PRECISION NOT NULL, documents BIGINT NOT NULL, cache_path TEXT NOT NULL,
    PRIMARY KEY (load_id, name)
);
CREATE TABLE IF NOT EXISTS mix_samples (
    load_id TEXT NOT NULL, source TEXT NOT NULL, doc_index BIGINT NOT NULL,
    tokens BIGINT NOT NULL, chars BIGINT NOT NULL, text TEXT NOT NULL,
    PRIMARY KEY (load_id, source, doc_index)
);
CREATE TABLE IF NOT EXISTS mix_sample_tokens (
    load_id TEXT NOT NULL, source TEXT NOT NULL, doc_index BIGINT NOT NULL,
    head_tokens BIGINT NOT NULL, token_ids INTEGER[] NOT NULL,
    PRIMARY KEY (load_id, source, doc_index)
);
CREATE TABLE IF NOT EXISTS mix_vocab (
    load_id TEXT NOT NULL, token_id BIGINT NOT NULL, piece TEXT NOT NULL,
    PRIMARY KEY (load_id, token_id)
);
CREATE TABLE IF NOT EXISTS mix_current (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton), load_id TEXT NOT NULL
)
"""

# mix_current holds at most one row: the load being served.
SOURCES_SQL = """
SELECT s.name, s.kind, s.family, s.share, s.documents, s.cache_path,
    COALESCE(
        (
            SELECT json_agg(
                json_build_object('doc_index', m.doc_index, 'tokens', m.tokens, 'chars', m.chars)
                ORDER BY m.doc_index
            )
            FROM mix_samples m WHERE m.load_id = s.load_id AND m.source = s.name
        ),
        '[]'::json
    ) AS samples
FROM mix_sources s JOIN mix_current c USING (load_id)
ORDER BY s.kind DESC, s.name
"""

SAMPLE_SQL = """
SELECT m.source, m.doc_index, m.tokens, m.chars, m.text
FROM mix_samples m JOIN mix_current c USING (load_id)
WHERE m.source = :source AND m.doc_index = :doc_index
"""

# pieces[i] is the decoded text of token_ids[i].
TOKENS_SQL = """
SELECT t.head_tokens, t.token_ids,
    (
        SELECT array_agg(v.piece ORDER BY u.position)
        FROM unnest(t.token_ids) WITH ORDINALITY AS u(token_id, position)
        JOIN mix_vocab v ON v.load_id = t.load_id AND v.token_id = u.token_id
    ) AS pieces
FROM mix_sample_tokens t JOIN mix_current c USING (load_id)
WHERE t.source = :source AND t.doc_index = :doc_index
"""


def migrate(connection: Connection) -> None:
    for statement in SCHEMA_SQL.split(";"):
        connection.execute(text(statement))


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/sources")
    def sources() -> list[dict[str, Any]]:
        """Every source of the served load with the sizes of its sampled documents."""
        with engine.connect() as connection:
            return [dict(row) for row in connection.execute(text(SOURCES_SQL)).mappings()]

    @api.get("/sample")
    def sample(source: str, doc_index: int) -> dict[str, Any]:
        with engine.connect() as connection:
            row = (
                connection.execute(text(SAMPLE_SQL), {"source": source, "doc_index": doc_index}).mappings().one_or_none()
            )
        if row is None:
            raise HTTPException(status_code=404, detail="No such sample in the served load")
        return dict(row)

    @api.get("/tokens")
    def tokens(source: str, doc_index: int) -> dict[str, Any]:
        """The token ids kept for one sample and each id's decoded piece; ids after ``head_tokens`` are the tail."""
        with engine.connect() as connection:
            row = (
                connection.execute(text(TOKENS_SQL), {"source": source, "doc_index": doc_index}).mappings().one_or_none()
            )
        if row is None:
            raise HTTPException(status_code=404, detail="No tokens for this sample in the served load")
        return dict(row)

    return api
