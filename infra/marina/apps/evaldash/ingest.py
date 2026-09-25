# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One scheduled EvalDash object-store reconciliation pass."""

import asyncio
import os

from marina.db import database_from_env, engine_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from .app import EvaldashConfig, PgRecordStore, PostgresIngestor, StoreMode
from .results_db import verify_schema

APP_NAME = "evaldash"


def main() -> int:
    """Reconcile every configured record prefix into EvalDash's PostgreSQL catalog once."""
    config = EvaldashConfig.from_env(os.environ)
    if config.store is not StoreMode.POSTGRES:
        raise RuntimeError("the scheduled EvalDash ingest requires the postgres store")
    database = database_from_env(os.environ)
    if database is None:
        raise RuntimeError("no database configured: set MARINA_DATABASE_URL or CLOUDSQL_CONNECTION")

    configure_coreweave_s3()
    engine = engine_for(database, APP_NAME)
    try:
        verify_schema(engine)
        store = PgRecordStore(engine)
        ingestor = PostgresIngestor(store, config.prefixes, config.ingest_interval, config.revalidate_after)
        failures = asyncio.run(ingestor.run_once())
        if failures:
            raise RuntimeError(f"EvalDash ingest failed for prefixes: {', '.join(failures)}")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
