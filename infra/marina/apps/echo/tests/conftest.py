# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Iterator

import pytest
import sqlalchemy
from marina.db import UrlDatabase, engine_for


@pytest.fixture
def schema_engine(database_url: str) -> Iterator[sqlalchemy.Engine]:
    """An engine on a new empty schema in the throwaway database; the schema is dropped afterwards."""
    name = f"echo_test_{uuid.uuid4().hex[:8]}"
    engine = engine_for(UrlDatabase(database_url), name)
    try:
        yield engine
    finally:
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'DROP SCHEMA "{name}" CASCADE'))
        engine.dispose()
