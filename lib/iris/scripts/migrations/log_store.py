#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate Iris logs from the old SQLite log store to DuckDBLogStore Parquet.

The old controller stores logs in ``logs.db`` (SQLite):

    CREATE TABLE logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL,
        source TEXT NOT NULL,
        data TEXT NOT NULL,
        epoch_ms INTEGER NOT NULL,
        level INTEGER NOT NULL DEFAULT 0
    );

This script:
  1. SCPs ``logs.db`` from the running controller
  2. Reads rows, constructs LogEntry protos, and feeds them through a real
     DuckDBLogStore instance which handles segmentation, sorting, and Parquet
     writing
  3. SCPs the resulting Parquet files back to the controller's log directory

Idempotent: if Parquet segments already exist on the controller they are
skipped during upload.

Usage:
  uv run python lib/iris/scripts/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml

  # Use an already-downloaded sqlite file
  uv run python lib/iris/scripts/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml --local-db /tmp/logs.db

  # Dry run
  uv run python lib/iris/scripts/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml --dry-run

  # Keep only the most recent N rows
  uv run python lib/iris/scripts/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml --max-rows 5000000
"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import click

from iris.rpc import logging_pb2

logger = logging.getLogger("migrate-log-store")

CONTROLLER_STATE_DIR = "/var/cache/iris/controller"
CONTROLLER_LOG_DIR = f"{CONTROLLER_STATE_DIR}/logs"

# Batch size for feeding rows into the LogStore. Large enough for throughput,
# small enough to avoid blowing up memory with proto objects.
BATCH_SIZE = 50_000


def _run(
    *args: str,
    timeout: float = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = list(args)
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def discover_controller_vm(project: str, label_prefix: str) -> tuple[str, str] | None:
    """Find controller VM by its Iris label. Returns (vm_name, zone) or None."""
    r = _run(
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter=labels.iris-{label_prefix}-controller=true",
        "--format=json(name,zone)",
        timeout=30,
    )
    instances = json.loads(r.stdout)
    if not instances:
        return None
    vm = instances[0]
    zone = vm["zone"].rsplit("/", 1)[-1]
    return vm["name"], zone


def scp_from_controller(vm: str, zone: str, project: str, remote_path: str, local_path: Path) -> None:
    """SCP a file from the controller VM to a local path."""
    _run(
        "gcloud",
        "compute",
        "scp",
        f"{vm}:{remote_path}",
        str(local_path),
        f"--zone={zone}",
        f"--project={project}",
        timeout=600,
    )


def scp_to_controller(local_path: Path, vm: str, zone: str, project: str, remote_path: str) -> None:
    """SCP a local file to the controller VM."""
    _run(
        "gcloud",
        "compute",
        "scp",
        str(local_path),
        f"{vm}:{remote_path}",
        f"--zone={zone}",
        f"--project={project}",
        timeout=600,
    )


def ssh_run(vm: str, zone: str, project: str, command: str, timeout: float = 30) -> str:
    """Run a command on the controller VM via SSH."""
    r = _run(
        "gcloud",
        "compute",
        "ssh",
        vm,
        f"--zone={zone}",
        f"--project={project}",
        f"--command={command}",
        timeout=timeout,
    )
    return r.stdout.strip()


def read_log_count(db_path: Path) -> int:
    """Count rows in the logs table."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    finally:
        conn.close()


def migrate_sqlite_to_log_store(
    db_path: Path,
    output_dir: Path,
    *,
    max_rows: int = 0,
) -> int:
    """Read logs from SQLite and write them through a DuckDBLogStore.

    Returns the number of rows migrated.
    """
    from iris.cluster.log_store.duckdb_store import DuckDBLogStore

    store = DuckDBLogStore(
        log_dir=output_dir,
        # No remote offload — we just want local Parquet files.
        remote_log_dir="",
        # Flush aggressively so everything lands on disk.
        flush_interval_sec=0,
    )

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if max_rows > 0:
            query = f"SELECT key, source, data, epoch_ms, level FROM logs ORDER BY id DESC LIMIT {max_rows}"
        else:
            query = "SELECT key, source, data, epoch_ms, level FROM logs ORDER BY id"
        cursor = conn.execute(query)

        total = 0
        batch: dict[str, list[logging_pb2.LogEntry]] = defaultdict(list)
        batch_size = 0

        for key, source, data, epoch_ms, level in cursor:
            entry = logging_pb2.LogEntry(source=source, data=data, level=level)
            entry.timestamp.epoch_ms = epoch_ms
            batch[key].append(entry)
            batch_size += 1

            if batch_size >= BATCH_SIZE:
                store.append_batch(list(batch.items()))
                total += batch_size
                batch.clear()
                batch_size = 0
                if total % 500_000 == 0:
                    logger.info("Migrated %d rows...", total)

        # Flush remaining batch.
        if batch_size > 0:
            store.append_batch(list(batch.items()))
            total += batch_size
    finally:
        conn.close()

    # close() seals the head buffer and flushes everything to Parquet.
    store.close()
    return total


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--local-db",
    "local_db_path",
    type=click.Path(),
    default=None,
    help="Use local logs.db instead of SCP from controller",
)
@click.option("--max-rows", default=0, type=int, help="Max rows to migrate (0 = all, newest first)")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--skip-upload", is_flag=True, default=False, help="Skip SCP back to controller")
@click.option("--output-dir", type=click.Path(), default=None, help="Local output dir for Parquet (default: temp dir)")
@click.option("--project", default="hai-gcp-models")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    local_db_path: str | None,
    max_rows: int,
    dry_run: bool,
    skip_upload: bool,
    output_dir: str | None,
    project: str,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    from iris.cluster.config import load_config

    cluster_config = load_config(Path(config_path))
    remote_state_dir = cluster_config.storage.remote_state_dir
    label_prefix = cluster_config.platform.label_prefix or "iris"

    if not remote_state_dir:
        raise click.ClickException("storage.remote_state_dir is required in the cluster config")

    # --- Discover controller ---
    vm_name: str | None = None
    zone: str | None = None
    if not local_db_path or not skip_upload:
        click.echo("Discovering controller VM...")
        vm_info = discover_controller_vm(project, label_prefix)
        if vm_info is None:
            raise click.ClickException(f"No controller VM found with label iris-{label_prefix}-controller=true")
        vm_name, zone = vm_info
        click.echo(f"  VM: {vm_name} ({zone})")

    # --- Get the SQLite DB ---
    if local_db_path:
        db_path = Path(local_db_path)
        if not db_path.exists():
            raise click.ClickException(f"Local DB not found: {db_path}")
    else:
        assert vm_name and zone
        remote_sqlite = f"{CONTROLLER_LOG_DIR}/logs.db"
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        tmp.close()

        if dry_run:
            click.echo("[dry-run] Would SCP logs.db — cannot proceed")
            return
        click.echo(f"SCP {vm_name}:{remote_sqlite} -> {db_path}")
        scp_from_controller(vm_name, zone, project, remote_sqlite, db_path)

    # --- Read + convert ---
    total_rows = read_log_count(db_path)
    click.echo(f"SQLite logs table: {total_rows} rows")
    if max_rows > 0:
        click.echo(f"Migrating newest {min(max_rows, total_rows)} rows")

    if total_rows == 0:
        click.echo("Nothing to migrate.")
        return

    if dry_run:
        click.echo(f"[dry-run] Would convert {total_rows} rows to Parquet via DuckDBLogStore")
        return

    parquet_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="iris_log_migration_"))
    click.echo(f"Writing Parquet via DuckDBLogStore to {parquet_dir}")

    migrated = migrate_sqlite_to_log_store(db_path, parquet_dir, max_rows=max_rows)
    click.echo(f"Migrated {migrated} rows")

    segments = sorted(parquet_dir.glob("logs_*_*.parquet"))
    total_bytes = sum(s.stat().st_size for s in segments)
    click.echo(f"Wrote {len(segments)} segments ({total_bytes / 1024 / 1024:.1f} MB)")

    # --- SCP back to controller ---
    if not skip_upload and vm_name and zone:
        click.echo(f"Ensuring log directory on controller: {CONTROLLER_LOG_DIR}")
        ssh_run(vm_name, zone, project, f"sudo mkdir -p {CONTROLLER_LOG_DIR}")

        click.echo(f"Uploading {len(segments)} segments to controller...")
        for seg in segments:
            remote_dest = f"{CONTROLLER_LOG_DIR}/{seg.name}"
            exists_check = ssh_run(
                vm_name,
                zone,
                project,
                f"test -f {remote_dest} && echo EXISTS || echo MISSING",
            )
            if exists_check == "EXISTS":
                click.echo(f"  {seg.name} already exists, skipping")
                continue
            click.echo(f"  Uploading {seg.name} ({seg.stat().st_size / 1024 / 1024:.1f} MB)")
            scp_to_controller(seg, vm_name, zone, project, remote_dest)
        click.echo("Upload complete.")
    elif skip_upload:
        click.echo(f"Parquet segments at: {parquet_dir}")

    click.echo()
    click.echo(f"Done. {migrated} rows -> {len(segments)} segments ({total_bytes / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
