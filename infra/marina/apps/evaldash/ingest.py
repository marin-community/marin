# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One scheduled EvalDash object-store reconciliation pass."""

import asyncio
import os

from marina.jobs import job_engine
from rigging.filesystem.s3_compat import configure_coreweave_s3

from .app import EvaldashConfig, PostgresIngestor, StoreMode
from .playground_reports import ingest_reports
from .results_db import verify_schema


def main() -> int:
    """Copy new playground reports, then reconcile every record prefix, into EvalDash's PostgreSQL schema once."""
    config = EvaldashConfig.from_env(os.environ)
    if config.store is not StoreMode.POSTGRES:
        raise RuntimeError("the scheduled EvalDash ingest requires the postgres store")
    configure_coreweave_s3()
    engine = job_engine()
    try:
        verify_schema(engine)
        # Reports first: copying them is quick, and a slow or failing record pass must not delay them.
        failures = ingest_reports(engine, config.report_prefixes)
        ingestor = PostgresIngestor(engine, config.prefixes, config.ingest_interval, config.revalidate_after)
        failures += asyncio.run(ingestor.run_once())
        if failures:
            raise RuntimeError(f"EvalDash ingest failed for prefixes: {', '.join(failures)}")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
