# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pg8000.dbapi
import pytest
import sqlalchemy
from marina.db import CloudSqlDatabase, QueryEngineCache
from sqlalchemy import text


@pytest.mark.parametrize("retirement", ["evict", "close"])
def test_query_cache_retirement_keeps_active_connector_until_connections_return(
    database_url: str, monkeypatch: pytest.MonkeyPatch, retirement: str
) -> None:
    url = sqlalchemy.make_url(database_url)
    assert url.username is not None and url.database is not None
    username = url.username
    database = url.database
    connectors = []

    class LocalConnector:
        """Replace the Cloud SQL transport with the disposable local Postgres."""

        def __init__(self, **_options: object):
            self.closed = False
            connectors.append(self)

        def connect(self, *_args: object, **_options: object):
            if self.closed:
                raise RuntimeError("connector closed while requests still need connections")
            return pg8000.dbapi.connect(
                user=username,
                password=url.password,
                host=url.host,
                port=url.port,
                database=database,
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("marina.db.Connector", LocalConnector)
    cache = QueryEngineCache(CloudSqlDatabase("local", database, username), capacity=1)
    try:
        with cache.engine("public", username) as first:
            with first.connect() as connection:
                original_pid = connection.execute(text("SELECT pg_backend_pid()")).scalar_one()
                if retirement == "evict":
                    with cache.engine("pg_catalog", username) as other:
                        with other.connect() as other_connection:
                            assert other_connection.execute(text("SELECT current_schema()")).scalar_one() == "pg_catalog"
                else:
                    cache.close()
                assert not connectors[0].closed
                # An active request can need a second connection after eviction.
                # Closing the connector with the first connection checked out breaks this.
                with first.connect() as sibling:
                    assert sibling.execute(text("SELECT pg_backend_pid()")).scalar_one() != original_pid
                assert connection.execute(text("SELECT current_schema()")).scalar_one() == "public"
        assert connectors[0].closed
        if retirement == "evict":
            with cache.engine("public", username) as replacement:
                with replacement.connect() as connection:
                    assert connection.execute(text("SELECT pg_backend_pid()")).scalar_one() != original_pid
    finally:
        cache.close()
    assert all(connector.closed for connector in connectors)
