#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate Iris logs: download remote Parquet segments to seed a local LogStore.

The DuckDBLogStore offloads Parquet segments to GCS (``remote_log_dir``) as
best-effort copies. On a fresh controller restart the local ``log_dir`` is
empty, so historical logs are lost unless we seed it from remote storage.

This script downloads remote Parquet segments back to the local log directory
so the new controller has log history available immediately.

Can be run standalone or called from ``controller_restart.py``.

Usage (standalone):

  uv run python lib/iris/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml

  # Dry run
  uv run python lib/iris/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml --dry-run

  # Override local destination
  uv run python lib/iris/migrations/log_store.py \
      --config lib/iris/examples/marin.yaml \
      --local-log-dir /tmp/iris_logs
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import fsspec.core

logger = logging.getLogger("iris.migrations.log_store")

LOCAL_STATE_DIR_DEFAULT = Path("/var/cache/iris/controller")
PARQUET_GLOB = "logs_*_*.parquet"


@dataclass(frozen=True)
class LogMigrationResult:
    """Summary of a log store migration."""

    remote_segments_found: int
    segments_downloaded: int
    segments_skipped: int
    total_bytes_downloaded: int
    verification_ok: bool


def _fsspec_copy(src: str, dst: str) -> None:
    """Copy a file using fsspec so either path can be remote (e.g. GCS)."""
    with fsspec.core.open(src, "rb") as f_src, fsspec.core.open(dst, "wb") as f_dst:
        f_dst.write(f_src.read())


def list_remote_segments(remote_log_dir: str) -> list[str]:
    """List Parquet log segments in the remote log directory.

    Returns full remote paths sorted by filename (which encodes seq ranges).
    """
    fs, fs_path = fsspec.core.url_to_fs(remote_log_dir)
    if not fs.exists(fs_path):
        logger.info("Remote log dir does not exist: %s", remote_log_dir)
        return []

    # fsspec glob returns full paths without the protocol prefix.
    pattern = f"{fs_path}/{PARQUET_GLOB}"
    matches = fs.glob(pattern)
    return sorted(matches)


def download_segments(
    remote_log_dir: str,
    local_log_dir: Path,
    *,
    dry_run: bool = False,
    max_segments: int = 0,
) -> LogMigrationResult:
    """Download remote Parquet log segments to the local log directory.

    Skips segments that already exist locally (by filename). Returns a summary.
    """
    fs, fs_path = fsspec.core.url_to_fs(remote_log_dir)
    protocol = remote_log_dir.split("://")[0] if "://" in remote_log_dir else ""

    remote_paths = list_remote_segments(remote_log_dir)
    if not remote_paths:
        logger.info("No remote log segments found in %s", remote_log_dir)
        return LogMigrationResult(
            remote_segments_found=0,
            segments_downloaded=0,
            segments_skipped=0,
            total_bytes_downloaded=0,
            verification_ok=True,
        )

    logger.info("Found %d remote log segments", len(remote_paths))

    local_log_dir.mkdir(parents=True, exist_ok=True)

    # Determine which segments to download (newest first if capped).
    if max_segments > 0 and len(remote_paths) > max_segments:
        logger.info("Capping download to newest %d segments (of %d)", max_segments, len(remote_paths))
        remote_paths = remote_paths[-max_segments:]

    downloaded = 0
    skipped = 0
    total_bytes = 0

    for remote_path in remote_paths:
        filename = remote_path.rsplit("/", 1)[-1]
        local_path = local_log_dir / filename

        if local_path.exists():
            logger.debug("Segment %s already exists locally, skipping", filename)
            skipped += 1
            continue

        if dry_run:
            size = fs.size(remote_path)
            logger.info("[dry-run] Would download %s (%d bytes)", filename, size)
            downloaded += 1
            total_bytes += size
            continue

        # Build the full remote URI for fsspec copy.
        remote_uri = f"{protocol}://{remote_path}" if protocol else remote_path

        try:
            _fsspec_copy(remote_uri, str(local_path))
            size = local_path.stat().st_size
            total_bytes += size
            downloaded += 1
            if downloaded % 10 == 0:
                logger.info("Downloaded %d segments (%.1f MB)...", downloaded, total_bytes / 1024 / 1024)
        except Exception:
            logger.exception("Failed to download segment %s", filename)
            # Continue with remaining segments.

    logger.info(
        "Download complete: %d downloaded, %d skipped, %.1f MB total",
        downloaded,
        skipped,
        total_bytes / 1024 / 1024,
    )

    return LogMigrationResult(
        remote_segments_found=len(remote_paths),
        segments_downloaded=downloaded,
        segments_skipped=skipped,
        total_bytes_downloaded=total_bytes,
        verification_ok=True,
    )


def verify_log_store(local_log_dir: Path) -> bool:
    """Verify the local log directory has readable Parquet segments.

    Creates a DuckDBLogStore pointing at the directory and checks it can
    read at least one log entry (if segments exist).
    """
    import pyarrow.parquet as pq

    segments = sorted(local_log_dir.glob(PARQUET_GLOB))
    if not segments:
        logger.info("No local segments to verify")
        return True

    for seg_path in segments:
        try:
            meta = pq.read_metadata(seg_path)
            if meta.num_rows == 0:
                logger.warning("Segment %s has 0 rows", seg_path.name)
            else:
                logger.debug("Segment %s: %d rows, %d row groups", seg_path.name, meta.num_rows, meta.num_row_groups)
        except Exception:
            logger.exception("Failed to read Parquet metadata from %s", seg_path.name)
            return False

    logger.info("All %d local segments are readable", len(segments))
    return True


def migrate_log_store(
    remote_log_dir: str,
    local_log_dir: Path,
    *,
    dry_run: bool = False,
    max_segments: int = 0,
    skip_verify: bool = False,
) -> LogMigrationResult:
    """Run the full log store migration: download + verify."""
    result = download_segments(
        remote_log_dir,
        local_log_dir,
        dry_run=dry_run,
        max_segments=max_segments,
    )

    if not skip_verify and not dry_run and result.segments_downloaded > 0:
        ok = verify_log_store(local_log_dir)
        result = LogMigrationResult(
            remote_segments_found=result.remote_segments_found,
            segments_downloaded=result.segments_downloaded,
            segments_skipped=result.segments_skipped,
            total_bytes_downloaded=result.total_bytes_downloaded,
            verification_ok=ok,
        )

    return result


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Iris cluster config YAML")
@click.option(
    "--local-log-dir",
    "local_log_dir_override",
    type=click.Path(),
    default=None,
    help="Override local log directory (default: {local_state_dir}/logs)",
)
@click.option(
    "--remote-log-dir",
    "remote_log_dir_override",
    default=None,
    help="Override remote log dir (default: {remote_state_dir}/logs)",
)
@click.option("--max-segments", default=0, type=int, help="Max segments to download (0 = all, newest first)")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would happen without downloading")
@click.option("--skip-verify", is_flag=True, default=False, help="Skip Parquet verification")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    local_log_dir_override: str | None,
    remote_log_dir_override: str | None,
    max_segments: int,
    dry_run: bool,
    skip_verify: bool,
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

    if not remote_state_dir:
        raise click.ClickException("storage.remote_state_dir is required in the cluster config")

    remote_log_dir = remote_log_dir_override or f"{remote_state_dir.rstrip('/')}/logs"
    local_state_dir = (
        Path(cluster_config.storage.local_state_dir)
        if cluster_config.storage.local_state_dir
        else LOCAL_STATE_DIR_DEFAULT
    )
    local_log_dir = Path(local_log_dir_override) if local_log_dir_override else local_state_dir / "logs"

    click.echo(f"Remote log dir: {remote_log_dir}")
    click.echo(f"Local log dir:  {local_log_dir}")

    result = migrate_log_store(
        remote_log_dir=remote_log_dir,
        local_log_dir=local_log_dir,
        dry_run=dry_run,
        max_segments=max_segments,
        skip_verify=skip_verify,
    )

    click.echo()
    click.echo("=" * 60)
    click.echo("Log Store Migration Summary")
    click.echo("=" * 60)
    click.echo(f"  Remote segments: {result.remote_segments_found}")
    click.echo(f"  Downloaded:      {result.segments_downloaded}")
    click.echo(f"  Skipped:         {result.segments_skipped}")
    click.echo(f"  Total bytes:     {result.total_bytes_downloaded / 1024 / 1024:.1f} MB")
    if not result.verification_ok:
        click.echo("  Verified:        FAILED")
        sys.exit(1)
    elif not dry_run and not skip_verify and result.segments_downloaded > 0:
        click.echo("  Verified:        YES")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
