# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A launch's worth of records in the database an EvalDash journey drives.

The kernel serves the database ``MARINA_DATABASE_URL`` names. The records are the deterministic
fixture set, written to a temporary directory and reconciled into the app's schema before the
kernel starts, so the walk needs no network, no Iris, no finelog and no object storage.
"""

import asyncio
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from evaldash import app as evaldash_app
from evaldash import fixtures
from marina.db import DATABASE_URL_ENV, UrlDatabase, engine_for

APP = "evaldash"


@pytest.fixture(scope="session", autouse=True)
def seeded_records(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Fixture records on disk, reconciled into the app's schema; yields the records prefix."""
    evaldash_database = os.environ.get(DATABASE_URL_ENV)
    if not evaldash_database:
        pytest.skip(f"set {DATABASE_URL_ENV} to a Postgres to walk through EvalDash")
    prefix = str(tmp_path_factory.mktemp("evaldash-records"))
    fixtures.build_fixtures(prefix)
    engine = engine_for(UrlDatabase(url=evaldash_database), APP)
    try:
        evaldash_app.migrate(engine)
        ingestor = evaldash_app.PostgresIngestor(
            evaldash_app.PgRecordStore(engine),
            (prefix,),
            evaldash_app.DEFAULT_INGEST_INTERVAL,
            evaldash_app.DEFAULT_REVALIDATE_AFTER,
        )
        # Playwright's sync API keeps an event loop running in the main thread once the browser
        # is up, so the ingest runs on its own thread with its own loop.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, ingestor.run_once()).result()
    finally:
        engine.dispose()
    # The kernel reads this when it mounts the app, so the served instance scans these records
    # rather than the remote defaults.
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv(evaldash_app.PREFIXES_ENV, prefix)
        yield prefix
