#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate Iris bundle store from legacy SQLite to flat-file fsspec storage.

Idempotent: skips bundles that already exist at the destination.

Usage:
  uv run python lib/iris/scripts/migrations/bundle_store.py \
      --config lib/iris/examples/marin.yaml

  # Use an already-downloaded sqlite file
  uv run python lib/iris/scripts/migrations/bundle_store.py \
      --config lib/iris/examples/marin.yaml --local-db /tmp/bundles.sqlite3

  # Dry run
  uv run python lib/iris/scripts/migrations/bundle_store.py \
      --config lib/iris/examples/marin.yaml --dry-run
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
import fsspec.core

from iris.cluster.bundle import BundleStore, bundle_id_for_zip

logger = logging.getLogger("migrate-bundle-store")

CONTROLLER_STATE_DIR = "/var/cache/iris/controller"


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
        timeout=300,
    )


def scp_to_controller(local_path: Path, vm: str, zone: str, project: str, remote_path: str) -> None:
    """SCP a local file to the controller VM."""
    _run(
        "gcloud", "compute", "scp",
        str(local_path), f"{vm}:{remote_path}",
        f"--zone={zone}", f"--project={project}",
        timeout=300,
    )


def read_bundles_from_sqlite(db_path: Path) -> dict[str, bytes]:
    """Read all bundles from a legacy SQLite bundle store. Returns {bundle_id: zip_bytes}."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT bundle_id, zip_bytes FROM bundles")
        return {row[0]: bytes(row[1]) for row in cursor}
    finally:
        conn.close()


def write_bundles_to_fsspec(bundles: dict[str, bytes], storage_dir: str, dry_run: bool = False) -> int:
    """Write bundle blobs as individual .zip files to fsspec storage. Returns count written."""
    fs, fs_path = fsspec.core.url_to_fs(storage_dir)
    fs.mkdirs(fs_path, exist_ok=True)

    written = 0
    for bundle_id, blob in bundles.items():
        dest = f"{fs_path}/{bundle_id}.zip"
        if fs.exists(dest):
            logger.info("Bundle %s already exists, skipping", bundle_id)
            continue
        if dry_run:
            logger.info("[dry-run] Would write %s (%d bytes)", dest, len(blob))
        else:
            with fs.open(dest, "wb") as f:
                f.write(blob)
            written += 1
            if written % 50 == 0:
                logger.info("Written %d bundles...", written)
    return written


def verify_bundles(bundles: dict[str, bytes], storage_dir: str) -> list[str]:
    """Verify all bundles are readable from the new BundleStore. Returns list of failed bundle_ids."""
    store = BundleStore(storage_dir=storage_dir)
    failures: list[str] = []

    for bundle_id, original_blob in bundles.items():
        try:
            read_blob = store.get_zip(bundle_id)
            actual_id = bundle_id_for_zip(read_blob)
            if actual_id != bundle_id:
                failures.append(bundle_id)
                logger.error("Hash mismatch for %s: got %s", bundle_id, actual_id)
            elif read_blob != original_blob:
                failures.append(bundle_id)
                logger.error("Content mismatch for %s", bundle_id)
        except Exception:
            failures.append(bundle_id)
            logger.exception("Failed to read bundle %s from new store", bundle_id)

    return failures


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--local-db", "local_db_path", type=click.Path(), default=None,
              help="Use local sqlite file instead of SCP from controller")
@click.option("--storage-dir", "storage_dir_override", default=None,
              help="Override destination (default: {remote_state_dir}/bundles)")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--skip-verify", is_flag=True, default=False)
@click.option("--project", default="hai-gcp-models")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    local_db_path: str | None,
    storage_dir_override: str | None,
    dry_run: bool,
    skip_verify: bool,
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

    storage_dir = storage_dir_override or f"{remote_state_dir.rstrip('/')}/bundles"
    click.echo(f"Destination: {storage_dir}")

    # --- Get the sqlite DB ---
    if local_db_path:
        db_path = Path(local_db_path)
        if not db_path.exists():
            raise click.ClickException(f"Local DB not found: {db_path}")
    else:
        click.echo("Discovering controller VM...")
        vm_info = discover_controller_vm(project, label_prefix)
        if vm_info is None:
            raise click.ClickException(f"No controller VM found with label iris-{label_prefix}-controller=true")
        vm_name, zone = vm_info
        click.echo(f"  VM: {vm_name} ({zone})")

        remote_sqlite = f"{CONTROLLER_STATE_DIR}/bundles.sqlite3"
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        db_path = Path(tmp.name)
        tmp.close()

        if dry_run:
            click.echo("[dry-run] Would SCP bundles.sqlite3 — cannot proceed")
            return
        click.echo(f"SCP {vm_name}:{remote_sqlite} -> {db_path}")
        scp_from_controller(vm_name, zone, project, remote_sqlite, db_path)

    # --- Read + write ---
    bundles = read_bundles_from_sqlite(db_path)
    total_bytes = sum(len(b) for b in bundles.values())
    click.echo(f"Found {len(bundles)} bundles ({total_bytes / 1024 / 1024:.1f} MB)")

    if not bundles:
        click.echo("Nothing to migrate.")
        return

    written = write_bundles_to_fsspec(bundles, storage_dir, dry_run=dry_run)
    click.echo(f"Wrote {written} new bundles ({len(bundles) - written} already existed)")

    # --- Verify ---
    if not skip_verify and not dry_run:
        failures = verify_bundles(bundles, storage_dir)
        if failures:
            click.echo(f"FAILED: {len(failures)} bundles could not be verified", err=True)
            sys.exit(1)
        click.echo(f"All {len(bundles)} bundles verified OK")


if __name__ == "__main__":
    main()
