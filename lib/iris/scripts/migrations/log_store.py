#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate Iris logs from the old SQLite log store to DuckDBLogStore Parquet.

The old controller stores logs in a SQLite table:

    CREATE TABLE logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL,
        source TEXT NOT NULL,
        data TEXT NOT NULL,
        epoch_ms INTEGER NOT NULL,
        level INTEGER NOT NULL DEFAULT 0
    );

This script:
  1. SCPs the SQLite logs DB from the running controller
  2. Reads all rows and converts them to sorted Parquet segments
     matching the DuckDBLogStore schema (id → seq)
  3. SCPs the Parquet files back to the controller's local log directory

Idempotent: if a Parquet segment covering the same seq range already exists
on the controller, it is skipped.

Usage:
  uv run python lib/iris/scripts/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml

  # Use an already-downloaded sqlite file
  uv run python lib/iris/scripts/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml --local-db /tmp/logs.sqlite3

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
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("migrate-log-store")

CONTROLLER_STATE_DIR = "/var/cache/iris/controller"
CONTROLLER_LOG_DIR = f"{CONTROLLER_STATE_DIR}/logs"

# Must match duckdb_store._PARQUET_SCHEMA exactly.
_PARQUET_SCHEMA = pa.schema([
    ("seq", pa.int64()),
    ("key", pa.string()),
    ("source", pa.string()),
    ("data", pa.string()),
    ("epoch_ms", pa.int64()),
    ("level", pa.int32()),
])

# Match duckdb_store constants.
SEGMENT_TARGET_BYTES = 100 * 1024 * 1024  # 100 MB
ROW_GROUP_SIZE = 16_384


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
        "gcloud", "compute", "instances", "list",
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
        "gcloud", "compute", "scp",
        f"{vm}:{remote_path}", str(local_path),
        f"--zone={zone}", f"--project={project}",
        timeout=600,
    )


def scp_to_controller(local_path: Path, vm: str, zone: str, project: str, remote_path: str) -> None:
    """SCP a local file to the controller VM."""
    _run(
        "gcloud", "compute", "scp",
        str(local_path), f"{vm}:{remote_path}",
        f"--zone={zone}", f"--project={project}",
        timeout=600,
    )


def ssh_run(vm: str, zone: str, project: str, command: str, timeout: float = 30) -> str:
    """Run a command on the controller VM via SSH."""
    r = _run(
        "gcloud", "compute", "ssh", vm,
        f"--zone={zone}", f"--project={project}",
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


def read_logs_from_sqlite(
    db_path: Path,
    *,
    max_rows: int = 0,
) -> pa.Table:
    """Read logs from the legacy SQLite store into a pyarrow Table.

    Maps SQLite ``id`` to Parquet ``seq``. Rows are sorted by (key, seq) for
    optimal row-group statistics in the DuckDBLogStore read path.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if max_rows > 0:
            # Take the most recent rows by id.
            query = f"SELECT id, key, source, data, epoch_ms, level FROM logs ORDER BY id DESC LIMIT {max_rows}"
        else:
            query = "SELECT id, key, source, data, epoch_ms, level FROM logs ORDER BY id"
        cursor = conn.execute(query)

        seqs: list[int] = []
        keys: list[str] = []
        sources: list[str] = []
        datas: list[str] = []
        epochs: list[int] = []
        levels: list[int] = []

        for row in cursor:
            seqs.append(row[0])
            keys.append(row[1])
            sources.append(row[2])
            datas.append(row[3])
            epochs.append(row[4])
            levels.append(row[5])
    finally:
        conn.close()

    table = pa.table(
        [
            pa.array(seqs, type=pa.int64()),
            pa.array(keys, type=pa.string()),
            pa.array(sources, type=pa.string()),
            pa.array(datas, type=pa.string()),
            pa.array(epochs, type=pa.int64()),
            pa.array(levels, type=pa.int32()),
        ],
        schema=_PARQUET_SCHEMA,
    )
    # Sort by (key, seq) so DuckDB can skip row groups efficiently.
    return table.sort_by([("key", "ascending"), ("seq", "ascending")])


def write_parquet_segments(table: pa.Table, output_dir: Path) -> list[Path]:
    """Split a table into Parquet segments matching DuckDBLogStore naming.

    Filenames: ``logs_{min_seq:019d}_{max_seq:019d}.parquet``
    Each segment targets ~100 MB (compressed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if table.num_rows == 0:
        return []

    # Estimate rows per segment from average row size.
    # Write a small sample to get compressed size, then extrapolate.
    sample_size = min(1000, table.num_rows)
    sample = table.slice(0, sample_size)
    sample_path = output_dir / "_sample.parquet.tmp"
    pq.write_table(sample, sample_path, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    bytes_per_row = sample_path.stat().st_size / sample_size
    sample_path.unlink()

    rows_per_segment = max(1000, int(SEGMENT_TARGET_BYTES / bytes_per_row))
    logger.info(
        "Estimated %.0f bytes/row, %d rows/segment for %d total rows",
        bytes_per_row, rows_per_segment, table.num_rows,
    )

    written: list[Path] = []
    offset = 0
    while offset < table.num_rows:
        chunk = table.slice(offset, rows_per_segment)
        seq_col = chunk.column("seq")
        min_seq = seq_col[0].as_py()
        max_seq = seq_col[-1].as_py()

        filename = f"logs_{min_seq:019d}_{max_seq:019d}.parquet"
        filepath = output_dir / filename
        pq.write_table(
            chunk, filepath,
            compression="zstd",
            row_group_size=ROW_GROUP_SIZE,
            write_page_index=True,
        )
        written.append(filepath)
        logger.info(
            "Wrote %s (%d rows, %.1f MB)",
            filename, chunk.num_rows, filepath.stat().st_size / 1024 / 1024,
        )
        offset += rows_per_segment

    return written


def verify_parquet_segments(segments: list[Path]) -> bool:
    """Verify each Parquet segment is readable and has the correct schema."""
    for seg in segments:
        try:
            meta = pq.read_metadata(seg)
            schema = meta.schema.to_arrow_schema()
            if schema != _PARQUET_SCHEMA:
                logger.error("Schema mismatch in %s: %s", seg.name, schema)
                return False
            if meta.num_rows == 0:
                logger.warning("Segment %s has 0 rows", seg.name)
        except Exception:
            logger.exception("Failed to read %s", seg.name)
            return False
    logger.info("All %d segments verified OK", len(segments))
    return True


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--local-db", "local_db_path", type=click.Path(), default=None,
              help="Use local sqlite file instead of SCP from controller")
@click.option("--max-rows", default=0, type=int, help="Max rows to migrate (0 = all, newest first)")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--skip-verify", is_flag=True, default=False)
@click.option("--skip-upload", is_flag=True, default=False, help="Skip SCP back to controller")
@click.option("--output-dir", type=click.Path(), default=None,
              help="Local output dir for Parquet (default: temp dir)")
@click.option("--project", default="hai-gcp-models")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    local_db_path: str | None,
    max_rows: int,
    dry_run: bool,
    skip_verify: bool,
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
        # The old log store lives in the controller DB itself.
        remote_sqlite = f"{CONTROLLER_STATE_DIR}/controller.sqlite3"
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        db_path = Path(tmp.name)
        tmp.close()

        if dry_run:
            click.echo("[dry-run] Would SCP controller.sqlite3 — cannot proceed")
            return
        click.echo(f"SCP {vm_name}:{remote_sqlite} -> {db_path}")
        scp_from_controller(vm_name, zone, project, remote_sqlite, db_path)

    # --- Read logs ---
    total_rows = read_log_count(db_path)
    click.echo(f"SQLite logs table: {total_rows} rows")
    if max_rows > 0:
        click.echo(f"Migrating newest {min(max_rows, total_rows)} rows")

    if total_rows == 0:
        click.echo("Nothing to migrate.")
        return

    if dry_run:
        click.echo(f"[dry-run] Would convert {total_rows} rows to Parquet")
        return

    click.echo("Reading logs from SQLite...")
    table = read_logs_from_sqlite(db_path, max_rows=max_rows)
    click.echo(f"Loaded {table.num_rows} rows into memory")

    # --- Write Parquet segments ---
    parquet_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="iris_log_migration_"))
    click.echo(f"Writing Parquet segments to {parquet_dir}")
    segments = write_parquet_segments(table, parquet_dir)
    click.echo(f"Wrote {len(segments)} segments")

    total_bytes = sum(s.stat().st_size for s in segments)
    click.echo(f"Total Parquet size: {total_bytes / 1024 / 1024:.1f} MB")

    # --- Verify ---
    if not skip_verify:
        if not verify_parquet_segments(segments):
            click.echo("VERIFICATION FAILED", err=True)
            sys.exit(1)

    # --- SCP back to controller ---
    if not skip_upload and vm_name and zone:
        click.echo(f"Ensuring log directory exists on controller: {CONTROLLER_LOG_DIR}")
        ssh_run(vm_name, zone, project, f"sudo mkdir -p {CONTROLLER_LOG_DIR}")

        click.echo(f"Uploading {len(segments)} Parquet segments to controller...")
        for seg in segments:
            remote_dest = f"{CONTROLLER_LOG_DIR}/{seg.name}"
            # Check if segment already exists on controller.
            exists_check = ssh_run(
                vm_name, zone, project,
                f"test -f {remote_dest} && echo EXISTS || echo MISSING",
            )
            if exists_check == "EXISTS":
                click.echo(f"  {seg.name} already exists, skipping")
                continue
            click.echo(f"  Uploading {seg.name} ({seg.stat().st_size / 1024 / 1024:.1f} MB)")
            scp_to_controller(seg, vm_name, zone, project, remote_dest)

        click.echo("Upload complete.")
    elif skip_upload:
        click.echo(f"Parquet segments available at: {parquet_dir}")

    # --- Summary ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Log Store Migration Summary")
    click.echo("=" * 60)
    click.echo(f"  SQLite rows:   {table.num_rows}")
    click.echo(f"  Segments:      {len(segments)}")
    click.echo(f"  Parquet size:  {total_bytes / 1024 / 1024:.1f} MB")
    if not skip_upload and vm_name:
        click.echo(f"  Uploaded to:   {vm_name}:{CONTROLLER_LOG_DIR}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
