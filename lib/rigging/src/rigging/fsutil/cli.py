# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``fsutil`` command line: list, read, size, copy, and remove across Marin's buckets.

Every path is a full URL (``gs://``, ``s3://``, or a local path). There is no implicit
current bucket, so the same command means the same thing from any shell, and a copy can
name two different backends. Bare ``fsutil`` opens the interactive browser.
"""

import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import click
from fsspec import AbstractFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from rigging.filesystem.buckets import MissingCredentials, filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import s3_credentials, s3_endpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.listing import (
    ROOT,
    Entry,
    ListingPhase,
    Preview,
    list_entries,
    read_decompressed_preview,
    read_preview,
    total_size,
)
from rigging.fsutil.parquet import PREVIEW_ROWS, MissingParquetReader, is_parquet, parquet_lines
from rigging.fsutil.render import aligned_lines, file_lines, format_size, format_time, table_lines
from rigging.fsutil.tui import run as run_browser
from rigging.fsutil.usage import (
    DEFAULT_PREFIX_DEPTH,
    DEFAULT_USAGE_WORKERS,
    ScanProgress,
    parse_byte_size,
    render_usage_report,
    scan_usage,
)

logger = logging.getLogger(__name__)

# Streaming chunk for cross-backend copies, which cannot use a filesystem's own
# server-side copy.
_COPY_CHUNK = 8 * 1024 * 1024
_RM_WORKERS = 8
_S3_DELETE_BATCH = 1000
_GCS_DELETE_BATCH = 20
_INTERACTIVE_PROGRESS_INTERVAL = 0.2
_LOG_PROGRESS_INTERVAL = 10.0


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Log fsspec/botocore activity.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Browse and manipulate Marin's object storage (GCS, CoreWeave, R2).

    Paths are full URLs: gs://bucket/key, s3://bucket/key, or a local path.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    if ctx.invoked_subcommand is None:
        ctx.invoke(browse)


@cli.command()
def buckets() -> None:
    """List the declared buckets and whether their backend is reachable."""
    rows = []
    for name, spec in sorted(data_buckets().items()):
        if spec.store == StoreType.GCS:
            endpoint, status = "-", "application default"
        else:
            endpoint = s3_endpoint(spec.store)
            status = "set" if s3_credentials(spec.store) else "missing"
        rows.append([name, str(spec.store), endpoint, status])
    for line in table_lines(["bucket", "backend", "endpoint", "credentials"], rows):
        click.echo(line)


@cli.command("ls")
@click.argument("url", default=ROOT)
@click.option("-l", "--long", is_flag=True, help="Show size and modification time.")
def list_command(url: str, long: bool) -> None:
    """List URL's immediate children or glob matches. With no URL, list the known buckets."""
    entries = list_entries(url)
    if not long:
        for entry in entries:
            click.echo(f"{entry.name}/" if entry.is_dir else entry.name)
        return
    _print_long_entries(entries)


@cli.command()
@click.argument("url")
@click.option("--raw", is_flag=True, help="Write bytes to stdout without formatting.")
def cat(url: str, raw: bool) -> None:
    """Print a file, rendering tabular JSON, JSONL, and parquet as a table."""
    if raw:
        data = _read_raw(url)
        sys.stdout.buffer.write(data)
        return
    for line in _formatted_lines(url, PREVIEW_ROWS):
        click.echo(line)


@cli.command()
@click.argument("url")
@click.option(
    "-n",
    "--lines",
    default=PREVIEW_ROWS,
    show_default=True,
    help="Number of lines to print, or rows for a parquet file.",
)
def head(url: str, lines: int) -> None:
    """Print the first lines of a file, or the first rows of a parquet file."""
    rendered = _formatted_lines(url, lines)
    for line in rendered if is_parquet(StoragePath(url).name) else rendered[:lines]:
        click.echo(line)


@cli.command()
@click.argument("url")
def stat(url: str) -> None:
    """Print an object's metadata as the backend reports it."""
    fs, path = filesystem_for(url)
    info = fs.info(path)
    for line in aligned_lines([[str(key), str(value)] for key, value in sorted(info.items())]):
        click.echo(line)


@cli.command()
@click.argument("url")
def du(url: str) -> None:
    """Total the bytes and objects under a prefix."""
    size, count = total_size(url)
    click.echo(f"{format_size(size)}  ({size} bytes, {count} objects)  {url}")


@cli.command()
@click.argument("url")
@click.option(
    "--prefix-threshold",
    default="1TiB",
    show_default=True,
    help="Descend into prefixes at or above this size; accepts values such as 1TB or 512GiB.",
)
@click.option(
    "--prefix-depth",
    default=DEFAULT_PREFIX_DEPTH,
    show_default=True,
    type=click.IntRange(min=1, max=20),
    help="Maximum path components retained for grouping.",
)
@click.option("--workers", default=DEFAULT_USAGE_WORKERS, show_default=True, type=click.IntRange(min=1, max=128))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Write Markdown here.")
def usage(url: str, prefix_threshold: str, prefix_depth: int, workers: int, output: Path | None) -> None:
    """Scan URL metadata and rank old, large prefixes for cleanup."""
    try:
        threshold_bytes = parse_byte_size(prefix_threshold)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--prefix-threshold") from error

    click.echo(f"Scanning object metadata under {url} ...", err=True)
    last_update = time.monotonic()
    rendered_width = 0
    latest_progress: ScanProgress | None = None
    interactive = click.get_text_stream("stderr").isatty()

    def show_progress(progress: ScanProgress) -> None:
        nonlocal last_update, latest_progress, rendered_width
        latest_progress = progress
        now = time.monotonic()
        interval = _INTERACTIVE_PROGRESS_INTERVAL if interactive else _LOG_PROGRESS_INTERVAL
        complete = progress.phase == ListingPhase.SCANNING and (
            progress.prefixes_completed == progress.prefixes_discovered
        )
        if not complete and now - last_update < interval:
            return
        line = _scan_progress_line(progress)
        if interactive:
            rendered_width = max(rendered_width, len(line))
            click.echo(f"\r{line.ljust(rendered_width)}", nl=False, err=True)
        else:
            click.echo(line, err=True)
        last_update = now

    try:
        scan = scan_usage(url, workers=workers, prefix_depth=prefix_depth, progress=show_progress)
    except MissingCredentials as error:
        raise click.ClickException(str(error)) from error
    if latest_progress is not None:
        line = _finished_scan_line(latest_progress)
        if interactive:
            click.echo(f"\r{line.ljust(max(rendered_width, len(line)))}", err=True)
        else:
            click.echo(line, err=True)
    report = render_usage_report(scan, threshold_bytes=threshold_bytes, generated_at=datetime.now(UTC))
    if output is None:
        click.echo(report)
        return
    output.write_text(report)
    click.echo(f"Wrote {output} ({scan.root.total.object_count:,} objects)", err=True)


def _scan_progress_line(progress: ScanProgress) -> str:
    rate = progress.stats.object_count / progress.elapsed_seconds if progress.elapsed_seconds else 0.0
    details = (
        f"{progress.listing_pages:,} pages | {progress.stats.object_count:,} objects | "
        f"{format_size(progress.stats.size_bytes)} | {rate:,.0f} obj/s"
    )
    if progress.phase == ListingPhase.DISCOVERING:
        spinner = "|/-\\"[progress.listing_pages % 4]
        return f"{spinner} Discovering {progress.current_prefix} | {details}"

    remaining = max(0, progress.prefixes_discovered - progress.prefixes_completed)
    return f"Scanning {progress.current_prefix} | {remaining:,} prefixes open | {details}"


def _finished_scan_line(progress: ScanProgress) -> str:
    return (
        f"Scan complete | {progress.listing_pages:,} pages | {progress.stats.object_count:,} objects | "
        f"{format_size(progress.stats.size_bytes)} | {_format_duration(progress.elapsed_seconds)}"
    )


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


@cli.command()
@click.argument("pattern")
def find(pattern: str) -> None:
    """List paths matching a glob pattern, e.g. 'gs://marin-us-central2/x/**/*.json'."""
    fs, path = filesystem_for(pattern)
    scheme = StoragePath(pattern).scheme
    for match in fs.glob(path):
        click.echo(f"{scheme}://{match}" if scheme else match)


@cli.command()
@click.argument("src")
@click.argument("dst")
@click.option("-r", "-R", "--recursive", is_flag=True, help="Copy a prefix and everything under it.")
def cp(src: str, dst: str, recursive: bool) -> None:
    """Copy between any two locations, including across backends."""
    src_fs, src_path = filesystem_for(src)
    dst_fs, dst_path = filesystem_for(dst)

    if not src_fs.exists(src_path):
        raise click.ClickException(f"{src} does not exist")

    if not recursive:
        if src_fs.isdir(src_path):
            raise click.ClickException(f"{src} is a directory; pass -r to copy it recursively")
        _copy_file(src_fs, src_path, dst_fs, dst_path)
        click.echo(f"{src} -> {dst}")
        return

    copied = 0
    for match in src_fs.find(src_path):
        _copy_file(src_fs, match, dst_fs, _destination(match, src_path, dst_path))
        copied += 1
    click.echo(f"{src} -> {dst} ({copied} objects)")


@cli.command()
@click.argument("url")
@click.option("-r", "-R", "--recursive", is_flag=True, help="Remove a prefix and everything under it.")
def rm(url: str, recursive: bool) -> None:
    """Remove an object, or recursively remove a prefix."""
    fs, path = filesystem_for(url)
    is_dir = fs.isdir(path)
    if is_dir and not recursive:
        raise click.ClickException(f"{url} is a directory; pass -r to remove it recursively")
    if not is_dir:
        fs.rm(path)
        click.echo(url)
        return
    if StoragePath(url).is_local:
        if fs.info(path).get("islink"):
            fs.rm_file(path)
        else:
            fs.rm(path, recursive=True)
        click.echo(url)
        return

    click.echo(f"Scanning {url} ...", err=True)
    entries = fs.find(path, detail=True)
    files = list(entries)
    total_bytes = sum(entry.get("size", 0) or 0 for entry in entries.values())
    batches = _delete_batches(fs, files)
    with ThreadPoolExecutor(max_workers=_RM_WORKERS) as executor:
        label = f"Removing {len(files)} objects ({format_size(total_bytes)})"
        with click.progressbar(length=len(files), label=label, show_pos=True) as progress:
            for start in range(0, len(batches), _RM_WORKERS):
                removals = {
                    executor.submit(_remove_batch, fs, batch): len(batch)
                    for batch in batches[start : start + _RM_WORKERS]
                }
                for removal in as_completed(removals):
                    removal.result()
                    progress.update(removals[removal])
    fs.invalidate_cache()
    click.echo(url)


def _delete_batches(fs: AbstractFileSystem, files: list[str]) -> list[list[str]]:
    if isinstance(fs, S3FileSystem):
        batch_size = _S3_DELETE_BATCH
    elif isinstance(fs, GCSFileSystem):
        batch_size = _GCS_DELETE_BATCH
    else:
        batch_size = 1
    return [files[start : start + batch_size] for start in range(0, len(files), batch_size)]


def _remove_batch(fs: AbstractFileSystem, files: list[str]) -> None:
    if not isinstance(fs, S3FileSystem):
        fs.rm(files)
        return

    objects = []
    buckets = set()
    for path in files:
        bucket, key, version = fs.split_path(path)
        buckets.add(bucket)
        item = {"Key": key}
        if version is not None:
            item["VersionId"] = version
        objects.append(item)
    assert len(buckets) == 1
    response = fs.call_s3(
        "delete_objects",
        Bucket=buckets.pop(),
        Delete={"Objects": objects, "Quiet": True},
    )
    errors = response.get("Errors", [])
    if errors:
        details = ", ".join(f"{error['Key']}: {error['Code']}" for error in errors)
        raise RuntimeError(f"S3 bulk delete failed: {details}")


@cli.command()
@click.argument("url", default=ROOT)
def browse(url: str) -> None:
    """Open the interactive browser, starting at URL (default: the bucket list)."""
    run_browser(url)


def _print_long_entries(entries: list[Entry]) -> None:
    rows = []
    for entry in entries:
        name = f"{entry.name}/" if entry.is_dir else entry.name
        rows.append([format_size(entry.size), format_time(entry.mtime), name])
    for line in table_lines(["size", "modified", "name"], rows):
        click.echo(line)


def _formatted_lines(url: str, rows: int) -> list[str]:
    """Render *url* for display, reading parquet through its footer and the rest by head.

    A parquet file states its own row count in the returned lines, so it takes *rows*
    rather than a line budget the caller applies afterwards.
    """
    name = StoragePath(url).name
    if is_parquet(name):
        try:
            return parquet_lines(url, rows)
        except MissingParquetReader as e:
            raise click.ClickException(str(e)) from e
    return file_lines(name, _read(url))


def _read(url: str) -> bytes:
    """Read a bounded, decompressed preview of *url*."""
    preview = read_decompressed_preview(url)
    _report_truncation(preview)
    return preview.data


def _read_raw(url: str) -> bytes:
    """Read stored bytes from *url* and report truncation on stderr."""
    preview = read_preview(url)
    _report_truncation(preview)
    return preview.data


def _report_truncation(preview: Preview) -> None:
    if not preview.truncated:
        return
    if preview.full_size is None:
        click.echo(f"[truncated: read first {format_size(len(preview.data))} of decompressed data]", err=True)
        return
    click.echo(
        f"[truncated: read {format_size(len(preview.data))} of {format_size(preview.full_size)}]",
        err=True,
    )


def _destination(match: str, src_path: str, dst_path: str) -> str:
    """Where *match* lands under *dst_path*, mirroring its position under *src_path*.

    A recursive copy whose source turns out to be a single object has nothing below the
    source to mirror, so the object keeps its own name under the destination.
    """
    relative = match[len(src_path) :].lstrip("/") or match.rsplit("/", 1)[-1]
    return f"{dst_path.rstrip('/')}/{relative}"


def _copy_file(src_fs, src_path: str, dst_fs, dst_path: str) -> None:
    """Stream one object between two filesystems, which may be different backends."""
    parent, separator, _ = dst_path.rpartition("/")
    if separator:
        dst_fs.makedirs(parent, exist_ok=True)
    with src_fs.open(src_path, "rb") as source, dst_fs.open(dst_path, "wb") as target:
        shutil.copyfileobj(source, target, _COPY_CHUNK)


def main() -> None:
    try:
        cli()
    except MissingCredentials as e:
        raise SystemExit(f"fsutil: {e}") from e


if __name__ == "__main__":
    main()
