# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``fsutil`` command line: list, read, size, copy, and remove across Marin's buckets.

Every path is a full URL (``gs://``, ``s3://``, or a local path). There is no implicit
current bucket, so the same command means the same thing from any shell, and a copy can
name two different backends. Bare ``fsutil`` opens the interactive browser.
"""

import logging
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
from rigging.fsutil.hashing import file_md5, format_digest
from rigging.fsutil.listing import (
    ROOT,
    Entry,
    Preview,
    list_entries,
    read_decompressed_preview,
    read_preview,
    total_size,
)
from rigging.fsutil.parquet import PREVIEW_ROWS, MissingParquetReader, is_parquet, parquet_lines
from rigging.fsutil.render import aligned_lines, file_lines, format_size, format_time, table_lines
from rigging.fsutil.transfer import (
    TransferError,
    copy_plan,
    execute_copy_plan,
    execute_sync,
    remove_sources,
    sync_plan,
)
from rigging.fsutil.tui import run as run_browser
from rigging.fsutil.usage import (
    DEFAULT_PREFIX_DEPTH,
    DEFAULT_USAGE_WORKERS,
    MAX_USAGE_WORKERS,
    ScanProgress,
    parse_byte_size,
    render_usage_report,
    scan_usage,
)

logger = logging.getLogger(__name__)

_RM_WORKERS = 8
_S3_DELETE_BATCH = 1000
_GCS_DELETE_BATCH = 20
_INTERACTIVE_PROGRESS_INTERVAL = 0.2
_LOG_PROGRESS_INTERVAL = 10.0
_ACTIVITY_BAR_WIDTH = 8
_MAX_PROGRESS_PATH_LENGTH = 60


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


@click.command()
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


@click.command("ls")
@click.argument("urls", nargs=-1)
@click.option("-l", "--long", is_flag=True, help="Show size and modification time.")
@click.option("-r", "-R", "--recursive", is_flag=True, help="List descendants beneath literal paths.")
def list_command(urls: tuple[str, ...], long: bool, recursive: bool) -> None:
    """List paths, immediate children, or glob matches. With no path, list known buckets."""
    targets = urls or (ROOT,)
    for index, url in enumerate(targets):
        if len(targets) > 1:
            if index:
                click.echo()
            click.echo(f"{url}:")
        entries = list_entries(url, recursive=recursive)
        if long:
            _print_long_entries(entries)
        else:
            for entry in entries:
                click.echo(f"{entry.name}/" if entry.is_dir else entry.name)


@click.command()
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


@click.command()
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


@click.command()
@click.argument("url")
def stat(url: str) -> None:
    """Print an object's metadata as the backend reports it."""
    fs, path = filesystem_for(url)
    info = fs.info(path)
    for line in aligned_lines([[str(key), str(value)] for key, value in sorted(info.items())]):
        click.echo(line)


@click.command()
@click.argument("url")
def du(url: str) -> None:
    """Total the bytes and objects under a prefix."""
    size, count = total_size(url)
    click.echo(f"{format_size(size)}  ({size} bytes, {count} objects)  {url}")


@click.command()
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
@click.option(
    "--workers",
    default=DEFAULT_USAGE_WORKERS,
    show_default=True,
    type=click.IntRange(min=1, max=MAX_USAGE_WORKERS),
)
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
        complete = progress.prefixes_completed == progress.prefixes_discovered
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
    object_rate = progress.stats.object_count / progress.elapsed_seconds if progress.elapsed_seconds else 0.0
    byte_rate = progress.stats.size_bytes / progress.elapsed_seconds if progress.elapsed_seconds else 0.0
    filled = max(1, round(_ACTIVITY_BAR_WIDTH * progress.workers_active / progress.workers_total))
    activity = f"[{'█' * filled}{'░' * (_ACTIVITY_BAR_WIDTH - filled)}]"
    remaining = max(0, progress.prefixes_discovered - progress.prefixes_completed)
    return (
        f"{activity} {progress.workers_active}/{progress.workers_total} threads | {remaining:,} open | "
        f"{progress.listing_pages:,} pages | {progress.stats.object_count:,} objects | "
        f"{format_size(progress.stats.size_bytes)} | {object_rate:,.0f} obj/s | "
        f"{format_size(round(byte_rate))}/s | scanning {_cropped_progress_path(progress.current_prefix)}"
    )


def _cropped_progress_path(path: str) -> str:
    if len(path) <= _MAX_PROGRESS_PATH_LENGTH:
        return path
    root, separator, tail = path.partition("/")
    head = f"{root}{separator}"
    suffix_length = _MAX_PROGRESS_PATH_LENGTH - len(head) - 1
    if suffix_length <= 0:
        return f"…{path[-(_MAX_PROGRESS_PATH_LENGTH - 1) :]}"
    return f"{head}…{tail[-suffix_length:]}"


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


@click.command()
@click.argument("pattern")
def find(pattern: str) -> None:
    """List paths matching a glob pattern, e.g. 'gs://marin-us-central2/x/**/*.json'."""
    fs, path = filesystem_for(pattern)
    scheme = StoragePath(pattern).scheme
    for match in fs.glob(path):
        click.echo(f"{scheme}://{match}" if scheme else match)


@click.command()
@click.argument("urls", nargs=-1, required=True)
@click.option("-r", "-R", "--recursive", is_flag=True, help="Copy a prefix and everything under it.")
@click.option("-n", "--no-clobber", is_flag=True, help="Skip files that already exist at the destination.")
def cp(urls: tuple[str, ...], recursive: bool, no_clobber: bool) -> None:
    """Copy one or more sources to a file, directory, or object prefix."""
    source_urls, destination_url = _transfer_arguments(urls)
    try:
        plan = copy_plan(source_urls, destination_url, recursive=recursive, no_clobber=no_clobber)
        execute_copy_plan(plan)
    except TransferError as error:
        raise click.ClickException(str(error)) from error
    for action in plan.copies:
        click.echo(f"{action.source.url} -> {action.destination.url}")
    for action in plan.skipped:
        click.echo(f"skip {action.destination.url}")


@click.command()
@click.argument("urls", nargs=-1, required=True)
@click.option("-r", "-R", "--recursive", is_flag=True, help="Move a prefix and everything under it.")
def mv(urls: tuple[str, ...], recursive: bool) -> None:
    """Move or rename one or more sources, including across backends."""
    source_urls, destination_url = _transfer_arguments(urls)
    try:
        plan = copy_plan(source_urls, destination_url, recursive=recursive, no_clobber=False)
        execute_copy_plan(plan)
        remove_sources(plan.sources)
    except TransferError as error:
        raise click.ClickException(str(error)) from error
    for action in plan.copies:
        click.echo(f"{action.source.url} -> {action.destination.url}")


@click.command()
@click.argument("source")
@click.argument("destination")
@click.option("--delete", is_flag=True, help="Delete destination files absent from the source.")
@click.option("--dry-run", is_flag=True, help="Print operations without changing the destination.")
@click.option(
    "--checksum",
    is_flag=True,
    help="Read equal-sized source and destination files and compare MD5 digests.",
)
def rsync(source: str, destination: str, delete: bool, dry_run: bool, checksum: bool) -> None:
    """Synchronize the files beneath two directories or object prefixes."""
    try:
        plan = sync_plan(source, destination, delete=delete, checksum=checksum)
        if not dry_run:
            execute_sync(plan)
    except TransferError as error:
        raise click.ClickException(str(error)) from error
    for action in plan.copies:
        click.echo(f"copy {action.source.url} -> {action.destination.url}")
    for action in plan.deletes:
        click.echo(f"delete {action.location.url}")


@click.command("hash")
@click.argument("urls", nargs=-1, required=True)
@click.option("--hex", "hexadecimal", is_flag=True, help="Print hexadecimal digests instead of base64.")
def hash_command(urls: tuple[str, ...], hexadecimal: bool) -> None:
    """Calculate MD5 content hashes for complete files or objects."""
    rows = []
    for url in urls:
        rows.append([url, format_digest(file_md5(url), hexadecimal=hexadecimal)])
    for line in table_lines(["url", "md5"], rows):
        click.echo(line)


@click.command()
@click.argument("urls", nargs=-1, required=True)
@click.option("-r", "-R", "--recursive", is_flag=True, help="Remove a prefix and everything under it.")
def rm(urls: tuple[str, ...], recursive: bool) -> None:
    """Remove objects, or recursively remove prefixes."""
    for url in urls:
        _remove(url, recursive)


def _remove(url: str, recursive: bool) -> None:
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


@click.command()
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


def _transfer_arguments(urls: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    if len(urls) < 2:
        raise click.UsageError("requires at least one source and one destination")
    return urls[:-1], urls[-1]


_COMMANDS = (buckets, list_command, cat, head, stat, du, usage, find, cp, mv, rsync, hash_command, rm, browse)

for _command in _COMMANDS:
    cli.add_command(_command)


def main() -> None:
    try:
        cli()
    except MissingCredentials as e:
        raise SystemExit(f"fsutil: {e}") from e


if __name__ == "__main__":
    main()
