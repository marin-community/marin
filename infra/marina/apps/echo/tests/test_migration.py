# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The migration against a real Postgres: Echo's tables land in Echo's own schema."""

import sqlalchemy
from marina.db import UrlDatabase, engine_for

from echo import app as echo_app
from echo import migrate, schema

APP = "echo"
SCHEMA_TABLES = sqlalchemy.text("SELECT table_name FROM information_schema.tables WHERE table_schema = :schema")


def test_migrate_creates_the_schema_once_and_records_what_it_applied(database_url: str) -> None:
    engine = engine_for(UrlDatabase(url=database_url), APP)
    try:
        echo_app.migrate(engine)
        # A second run must be a no-op: the deploy job runs it on every image change.
        echo_app.migrate(engine)
        with engine.connect() as conn:
            tables = {row[0] for row in conn.execute(SCHEMA_TABLES, {"schema": APP})}
            applied = [
                row[0] for row in conn.execute(sqlalchemy.text("SELECT name FROM schema_migrations ORDER BY name"))
            ]
            wiki_columns = {
                row[0]
                for row in conn.execute(
                    sqlalchemy.text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = :schema AND table_name = 'wiki_entries'"
                    ),
                    {"schema": APP},
                )
            }
    finally:
        engine.dispose()

    assert set(schema.metadata.tables) <= tables
    assert applied == migrate.migration_names()
    # The generated column and the vector column are what searching the wiki needs.
    assert {"search_document", "embedding"} <= wiki_columns
