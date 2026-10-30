# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One scheduled EvalDash object-store reconciliation pass."""

import asyncio
import os

from marina.jobs import job_engine
from rigging.filesystem.s3_compat import configure_coreweave_s3

from .app import EvaldashConfig, PostgresIngestor, StoreMode
from .results_db import verify_schema


def main() -> int:
    """Reconcile every configured record prefix into EvalDash's PostgreSQL catalog once."""
    config = EvaldashConfig.from_env(os.environ)
    if config.store is not StoreMode.POSTGRES:
        raise RuntimeError("the scheduled EvalDash ingest requires the postgres store")
    configure_coreweave_s3()
    engine = job_engine()
    try:
        verify_schema(engine)
        ingestor = PostgresIngestor(engine, config.prefixes, config.ingest_interval, config.revalidate_after)
        failures = asyncio.run(ingestor.run_once())
        if failures:
            raise RuntimeError(f"EvalDash ingest failed for prefixes: {', '.join(failures)}")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
