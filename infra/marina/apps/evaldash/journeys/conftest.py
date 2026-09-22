# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A launch's worth of records in the database an EvalDash journey drives.

The kernel serves the database ``MARINA_DATABASE_URL`` names. The records are the deterministic
fixture set, written to a temporary directory and reconciled into the app's schema before the
first EvalDash page loads, so the walk needs no network, no Iris, no finelog and no object
storage. The records prefix is pointed at when this module is collected: the kernel is shared
by every app's journeys and may already be serving by the time EvalDash's fixtures run.
"""

import asyncio
import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from evaldash import app as evaldash_app
from evaldash import fixtures
from marina.db import DATABASE_URL_ENV, UrlDatabase, engine_for

APP = "evaldash"
RECORDS_PREFIX = tempfile.mkdtemp(prefix="evaldash-records-")
os.environ[evaldash_app.PREFIXES_ENV] = RECORDS_PREFIX


@pytest.fixture(scope="session", autouse=True)
def seeded_records(request: pytest.FixtureRequest) -> Iterator[str]:
    """Fixture records on disk, reconciled into the app's schema; yields the records prefix."""
    evaldash_database = os.environ.get(DATABASE_URL_ENV)
    if not evaldash_database:
        pytest.skip(f"set {DATABASE_URL_ENV} to a Postgres to walk through EvalDash")
    prefix = RECORDS_PREFIX
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
    # The kernel may have mounted the app before these rows existed; one refresh pass makes
    # the served catalog current instead of waiting on the reload loop.
    kernel = request.getfixturevalue("marina_kernel")
    httpx.post(f"{kernel.origin}/evaldash/api/refresh", timeout=60).raise_for_status()
    yield prefix
