# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One row of each kind for the pages to show.

Echo is a Python app, so its journey needs a Postgres: ``MARINA_DATABASE_URL`` names one,
and the journey kernel serves it the way ``marina dev`` would.

Nothing here reaches the network: the corpus rows are written directly, and the journey
searches them with ``grep`` and the wiki listing, neither of which embeds a query.
"""

import os

import pytest
import sqlalchemy
from marina.db import database_from_env, engine_for

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
def echo_corpus() -> None:
    """Echo's schema, holding one activity chunk and one wiki note."""
    database = database_from_env(os.environ)
    if database is None:
        pytest.skip("set MARINA_DATABASE_URL to a Postgres with pgvector to walk through Echo")
    engine = engine_for(database, APP)
    try:
        echo_app.migrate(engine)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.delete(schema.chunks))
            conn.execute(sqlalchemy.delete(schema.wiki_entries))
            conn.execute(schema.chunks.insert().values(**CHUNK))
            conn.execute(schema.wiki_entries.insert().values(**WIKI))
    finally:
        engine.dispose()
