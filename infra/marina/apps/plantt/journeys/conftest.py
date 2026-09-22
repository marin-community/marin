# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterator

import pytest
from marina.applets import AppletStore
from marina.database_setup import ensure_applet_provisioning
from marina.db import DATABASE_URL_ENV, UrlDatabase, engine_for
from marina.testing import test_database
from plantt import app as plantt


@pytest.fixture(scope="session", autouse=True)
def plantt_database() -> Iterator[None]:
    previous = os.environ.get(DATABASE_URL_ENV)
    with test_database() as url:
        os.environ[DATABASE_URL_ENV] = url
        database = UrlDatabase(url)
        applet_store = AppletStore(database)
        engine = engine_for(database, "plantt")
        try:
            ensure_applet_provisioning(applet_store.engine)
            applet_store.migrate()
            plantt.migrate(engine)
            yield
        finally:
            applet_store.engine.dispose()
            engine.dispose()
            if previous is None:
                os.environ.pop(DATABASE_URL_ENV, None)
            else:
                os.environ[DATABASE_URL_ENV] = previous
