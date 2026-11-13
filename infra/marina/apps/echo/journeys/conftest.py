# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One row of each kind for the pages to show.

The journey uses the shared disposable Postgres registry and released source packages.

Nothing here reaches the network: the corpus rows are written directly, and the journey
searches them with ``grep`` and the wiki listing, neither of which embeds a query.
"""


import pytest
import sqlalchemy
from marina.db import DatabaseSpec, engine_for

from echo import app as echo_app
from echo import schema

APP = "echo"
# A unit vector: pgvector's cosine operator class has no meaning for a zero vector, and the
# journey never asks for a distance anyway.
EMBEDDING = [1.0] + [0.0] * (schema.EMBED_DIM - 1)
CHUNK = {
    "id": 1,
    "source": "github",
    "kind": "issue",
    "url": "https://github.com/marin-community/marin/issues/1",
    "title": "ragged_all_to_all overflows on the second shard",
    "text": "The ragged_all_to_all kernel writes past the end of the receive buffer.",
    "author": "someone",
    "date": "2026-01-02T03:04:05+00:00",
}
WIKI = {
    "author": "someone",
    "title": "Grafana access for a new engineer",
    "use_when": "when a teammate cannot open the Marin Grafana",
    "tags": ["ops", "grafana"],
    "body": "Grafana reads its viewer list from the eng-all group; membership takes 15 minutes.",
    "embedding": EMBEDDING,
}


@pytest.fixture(scope="session", autouse=True)
def echo_corpus(marina_database: DatabaseSpec) -> None:
    """Echo's schema, holding one activity chunk and one wiki note."""
    engine = engine_for(marina_database, APP)
    try:
        with engine.begin() as migration_connection:
            echo_app.migrate(migration_connection)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.delete(schema.chunks))
            conn.execute(sqlalchemy.delete(schema.wiki_entries))
            conn.execute(schema.chunks.insert().values(**CHUNK))
            conn.execute(schema.wiki_entries.insert().values(**WIKI))
    finally:
        engine.dispose()
