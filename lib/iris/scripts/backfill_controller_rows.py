#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill the iris.controller_db__* Finelog rows from a controller DB checkpoint, once per cluster.

Rows cannot be removed once written: Finelog has no row delete. Run the dry run
first. The rules, guards and what the rows hold are in
``iris.cluster.controller.observability_backfill`` and
``lib/iris/docs/observability-rows.md`` ("Backfilling rows from before the live path").

Usage, as an Iris job on the cluster it backfills (resolves the cluster's log server):

    uv run python lib/iris/scripts/backfill_controller_rows.py dry-run \\
        --cluster marin --live-start-ms <ms> --checkpoint gs://.../controller-state/<epoch_ms>

or from a local copy (``--db-dir``, a directory holding controller.sqlite3) and an
explicit ``--log-server`` URL.
"""

import logging
import sys
import tempfile
from pathlib import Path

import click
from finelog.client import LogClient
from iris.client.client import iris_ctx
from iris.cluster.controller.checkpoint import download_checkpoint_to_local
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.observability_backfill import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_PAUSE_SECONDS,
    TARGETS,
    BackfillRefused,
    plan_backfill,
    summary,
    write_backfill,
)
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from rigging.timing import Timestamp


def _confirmation(cluster: str, live_start_ms: int) -> str:
    return f"{cluster}:{live_start_ms}"


def _log_client(log_server: str | None) -> LogClient:
    if log_server:
        return LogClient.connect(log_server)
    ctx = iris_ctx()
    assert ctx.client is not None, "resolving the log server needs an in-cluster iris client"
    return LogClient.connect(LOG_SERVER_ENDPOINT_NAME, resolver=ctx.client.resolve_endpoint)


@click.command()
@click.argument("mode", type=click.Choice(["dry-run", "run", "check"]))
@click.option("--cluster", required=True, help="The cluster config name; every row's cluster column")
@click.option("--live-start-ms", required=True, type=int, help="When this cluster's live rows started (epoch ms)")
@click.option("--checkpoint", help="A checkpoint directory URL, {remote_state_dir}/controller-state/<epoch_ms>")
@click.option("--db-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), help="A local DB copy")
@click.option("--log-server", help="Finelog URL; default: the cluster's /system/log-server, from inside a job")
@click.option("--rerun", is_flag=True, help="An earlier backfill run wrote rows for this cluster (G4)")
@click.option("--i-understand-rows-cannot-be-removed", "confirmation", help="Required by run; the dry run prints it")
@click.option("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS, show_default=True)
@click.option("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS, show_default=True)
def main(mode, cluster, live_start_ms, checkpoint, db_dir, log_server, rerun, confirmation, batch_rows, pause_seconds):
    """MODE: dry-run (counts, writes nothing), run (writes), check (exit 1 if any DB entity has no row)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if (checkpoint is None) == (db_dir is None):
        raise click.UsageError("give exactly one of --checkpoint and --db-dir")
    if mode == "run" and confirmation != _confirmation(cluster, live_start_ms):
        raise click.UsageError("run needs the --i-understand-rows-cannot-be-removed value the dry run prints")
    with tempfile.TemporaryDirectory(prefix="controller-db-copy-") as scratch:
        # The copy holds every column, secrets included; it is deleted with the scratch directory.
        if checkpoint is not None:
            db_dir = Path(scratch) / "db"
            if not download_checkpoint_to_local(checkpoint, db_dir, checkpoint_dir=checkpoint):
                raise click.ClickException(f"no controller DB at {checkpoint}")
        db = ControllerDB(db_dir=db_dir)
        client = _log_client(log_server)
        try:
            plans = plan_backfill(
                db,
                client,
                cluster=cluster,
                live_start_ms=live_start_ms,
                rerun=rerun or mode == "check",
                now=Timestamp.now().as_naive_utc(),
            )
            for line in summary(plans, batch_rows=batch_rows, pause_seconds=pause_seconds):
                click.echo(line)
            if mode == "dry-run":
                click.echo(
                    f"To write: run --i-understand-rows-cannot-be-removed={_confirmation(cluster, live_start_ms)}"
                )
            elif mode == "check":
                missing = sum(len(plan.to_write) for plan in plans)
                click.echo(f"check: {missing} DB entities have no row in Finelog")
                if missing:
                    sys.exit(1)
            else:
                tables = {
                    target: client.get_table(target.namespace, target.row_class, storage_policy=target.storage_policy)
                    for target in TARGETS
                }
                write_backfill(plans, tables, batch_rows=batch_rows, pause_seconds=pause_seconds)
                click.echo("run: done; rerun check to compare with the DB")
        except BackfillRefused as refused:
            raise click.ClickException(str(refused)) from refused
        finally:
            client.close()
            db.close()


if __name__ == "__main__":
    main()
