#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate Iris bundle store from SQLite to flat-file fsspec storage.

Copies bundles.sqlite3 off the running controller via SCP, reads all bundles,
writes them as individual {bundle_id}.zip files to the new BundleStore storage
path, then verifies every bundle is readable via BundleStore.get_zip().

Usage:

  # Migrate bundles to the GCS path derived from the cluster config
  uv run python scripts/migrations/migrate_bundle_store.py \\
      --config lib/iris/examples/marin.yaml

  # Dry run — show what would happen
  uv run python scripts/migrations/migrate_bundle_store.py \\
      --config lib/iris/examples/marin.yaml --dry-run

  # Use an already-downloaded sqlite file instead of SCP
  uv run python scripts/migrations/migrate_bundle_store.py \\
      --config lib/iris/examples/marin.yaml --local-db /tmp/bundles.sqlite3

  # Override the destination storage dir
  uv run python scripts/migrations/migrate_bundle_store.py \\
      --config lib/iris/examples/marin.yaml \\
      --storage-dir gs://marin-us-central2/iris/marin/state/bundles
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

logger = logging.getLogger("migrate-bundle-store")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTROLLER_STATE_DIR = "/var/cache/iris/controller"


def _run(
    *args: str,
    timeout: float = 120,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = list(args)
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, check=check, cwd=str(REPO_ROOT))


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
        timeout=300,
    )


def read_bundles_from_sqlite(db_path: Path) -> dict[str, bytes]:
    """Read all bundles from a legacy SQLite bundle store. Returns {bundle_id: zip_bytes}."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT bundle_id, zip_bytes FROM bundles")
        bundles = {row[0]: bytes(row[1]) for row in cursor}
    finally:
        conn.close()
    return bundles


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
    from iris.cluster.bundle import BundleStore, bundle_id_for_zip

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
                logger.error("Content mismatch for %s: sizes %d vs %d", bundle_id, len(original_blob), len(read_blob))
        except Exception:
            failures.append(bundle_id)
            logger.exception("Failed to read bundle %s from new store", bundle_id)

    return failures


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Iris cluster config YAML")
@click.option(
    "--local-db",
    "local_db_path",
    type=click.Path(),
    default=None,
    help="Use local sqlite file instead of SCP from controller",
)
@click.option(
    "--storage-dir",
    "storage_dir_override",
    default=None,
    help="Override destination storage dir (default: {remote_state_dir}/bundles)",
)
@click.option("--dry-run", is_flag=True, default=False, help="Show what would happen without writing")
@click.option("--skip-verify", is_flag=True, default=False, help="Skip read-back verification")
@click.option("--project", default="hai-gcp-models", help="GCP project ID")
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
    click.echo(f"Destination storage: {storage_dir}")

    # --- Get the sqlite DB ---
    if local_db_path:
        db_path = Path(local_db_path)
        if not db_path.exists():
            raise click.ClickException(f"Local DB not found: {db_path}")
        click.echo(f"Using local DB: {db_path}")
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

        click.echo(f"SCP {vm_name}:{remote_sqlite} -> {db_path}")
        if dry_run:
            click.echo("  [dry-run] Would SCP bundles.sqlite3")
            click.echo("  [dry-run] Cannot proceed without DB — exiting")
            return
        scp_from_controller(vm_name, zone, project, remote_sqlite, db_path)

    # --- Read bundles from sqlite ---
    click.echo("Reading bundles from SQLite...")
    bundles = read_bundles_from_sqlite(db_path)
    total_bytes = sum(len(b) for b in bundles.values())
    click.echo(f"  Found {len(bundles)} bundles ({total_bytes / 1024 / 1024:.1f} MB)")

    if not bundles:
        click.echo("No bundles to migrate.")
        return

    # --- Write to fsspec storage ---
    click.echo(f"Writing bundles to {storage_dir}...")
    written = write_bundles_to_fsspec(bundles, storage_dir, dry_run=dry_run)
    if dry_run:
        click.echo(f"  [dry-run] Would write {len(bundles)} bundles")
    else:
        click.echo(f"  Wrote {written} new bundles ({len(bundles) - written} already existed)")

    # --- Verify ---
    if skip_verify or dry_run:
        click.echo("Skipping verification.")
    else:
        click.echo("Verifying all bundles are readable from the new BundleStore...")
        failures = verify_bundles(bundles, storage_dir)
        if failures:
            click.echo(f"  FAILED: {len(failures)} bundles could not be verified:")
            for bid in failures[:10]:
                click.echo(f"    {bid}")
            if len(failures) > 10:
                click.echo(f"    ... and {len(failures) - 10} more")
            sys.exit(1)
        else:
            click.echo(f"  All {len(bundles)} bundles verified OK")

    # --- Summary ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Migration Summary")
    click.echo("=" * 60)
    click.echo(f"  Source:      {db_path}")
    click.echo(f"  Destination: {storage_dir}")
    click.echo(f"  Bundles:     {len(bundles)} ({total_bytes / 1024 / 1024:.1f} MB)")
    click.echo(f"  New writes:  {written}")
    if not dry_run and not skip_verify:
        click.echo("  Verified:    YES — safe to restart controller with new BundleStore code")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
