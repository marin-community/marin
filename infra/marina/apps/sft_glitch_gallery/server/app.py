# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the selected load of flaw categories and their example documents from Postgres."""

from typing import Any

from fastapi import FastAPI, HTTPException
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gallery_categories (
    load_id TEXT NOT NULL, key TEXT NOT NULL, title TEXT NOT NULL, severity TEXT NOT NULL,
    description TEXT NOT NULL, prevalence TEXT NOT NULL, position BIGINT NOT NULL,
    PRIMARY KEY (load_id, key)
);
CREATE TABLE IF NOT EXISTS gallery_examples (
    load_id TEXT NOT NULL, category TEXT NOT NULL, source TEXT NOT NULL, doc_index BIGINT NOT NULL,
    tokens BIGINT NOT NULL, note TEXT NOT NULL, text TEXT NOT NULL,
    match_start BIGINT NOT NULL, match_length BIGINT NOT NULL, position BIGINT NOT NULL,
    PRIMARY KEY (load_id, category, source, doc_index)
);
CREATE TABLE IF NOT EXISTS gallery_current (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton), load_id TEXT NOT NULL
)
"""

# gallery_current holds at most one row: the load being served.
CATEGORIES_SQL = """
SELECT c.key, c.title, c.severity, c.description, c.prevalence,
    COALESCE(
        (
            SELECT json_agg(
                json_build_object('source', e.source, 'doc_index', e.doc_index, 'tokens', e.tokens, 'note', e.note)
                ORDER BY e.position
            )
            FROM gallery_examples e WHERE e.load_id = c.load_id AND e.category = c.key
        ),
        '[]'::json
    ) AS examples
FROM gallery_categories c JOIN gallery_current USING (load_id)
ORDER BY c.position
"""

EXAMPLE_SQL = """
SELECT e.category, e.source, e.doc_index, e.tokens, e.note, e.text, e.match_start, e.match_length
FROM gallery_examples e JOIN gallery_current USING (load_id)
WHERE e.category = :category AND e.source = :source AND e.doc_index = :doc_index
"""


def migrate(connection: Connection) -> None:
    for statement in SCHEMA_SQL.split(";"):
        connection.execute(text(statement))


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/categories")
    def categories() -> list[dict[str, Any]]:
        """Every flaw category of the served load, in display order, with its examples listed without text."""
        with engine.connect() as connection:
            return [dict(row) for row in connection.execute(text(CATEGORIES_SQL)).mappings()]

    @api.get("/example")
    def example(category: str, source: str, doc_index: int) -> dict[str, Any]:
        """One example's text window and where the flaw sits in it."""
        parameters = {"category": category, "source": source, "doc_index": doc_index}
        with engine.connect() as connection:
            row = connection.execute(text(EXAMPLE_SQL), parameters).mappings().one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="No such example in the served load")
        return dict(row)

    return api
