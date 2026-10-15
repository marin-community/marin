# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from marina.db import DatabaseSpec, engine_for
from plantt import app as plantt


@pytest.fixture(scope="session", autouse=True)
def plantt_database(marina_database: DatabaseSpec) -> Iterator[None]:
    engine = engine_for(marina_database, "plantt")
    try:
        with engine.begin() as connection:
            plantt.migrate(connection)
        yield
    finally:
        engine.dispose()
