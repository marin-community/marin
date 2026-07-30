# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``fsutil`` command line: list, read, size, and copy across Marin's buckets.

Every path is a full URL (``gs://``, ``s3://``, or a local path). There is no implicit
current bucket, so the same command means the same thing from any shell, and a copy can
name two different backends. Bare ``fsutil`` opens the interactive browser.
"""

import logging
import shutil
import sys

import click
from rich.console import Console
from rich.table import Table

from rigging.filesystem.buckets import MissingCredentials, filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import s3_credentials, s3_endpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.listing import ROOT, Preview, list_entries, read_decompressed_preview, read_preview, total_size
from rigging.fsutil.render import file_lines, format_size, print_entries
from rigging.fsutil.tui import run as run_browser

logger = logging.getLogger(__name__)

# Streaming chunk for cross-backend copies, which cannot use a filesystem's own
# server-side copy.
_COPY_CHUNK = 8 * 1024 * 1024

console = Console()


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
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("bucket")
    table.add_column("backend")
    table.add_column("endpoint")
    table.add_column("credentials")
    for name, spec in sorted(data_buckets().items()):
        if spec.store == StoreType.GCS:
            endpoint, status = "-", "application default"
        else:
            endpoint = s3_endpoint(spec.store)
            status = "set" if s3_credentials(spec.store) else "[red]missing[/red]"
        table.add_row(name, str(spec.store), endpoint, status)
    console.print(table)


@cli.command("ls")
@click.argument("url", default=ROOT)
@click.option("-l", "--long", is_flag=True, help="Show size and modification time.")
def list_command(url: str, long: bool) -> None:
    """List the immediate children of URL. With no URL, list the known buckets."""
    print_entries(console, list_entries(url), long=long)


@cli.command()
@click.argument("url")
@click.option("--raw", is_flag=True, help="Write bytes to stdout without formatting.")
def cat(url: str, raw: bool) -> None:
    """Print a file, rendering tabular JSON and JSONL as a table."""
    if raw:
        data = _read_raw(url)
        sys.stdout.buffer.write(data)
        return
    data = _read(url)
    for line in file_lines(StoragePath(url).name, data):
        click.echo(line)


@cli.command()
@click.argument("url")
@click.option("-n", "--lines", default=20, show_default=True, help="Number of lines to print.")
def head(url: str, lines: int) -> None:
    """Print the first lines of a file."""
    data = _read(url)
    for line in file_lines(StoragePath(url).name, data)[:lines]:
        click.echo(line)


@cli.command()
@click.argument("url")
def stat(url: str) -> None:
    """Print an object's metadata as the backend reports it."""
    fs, path = filesystem_for(url)
    info = fs.info(path)
    table = Table(box=None, pad_edge=False, show_header=False)
    for key, value in sorted(info.items()):
        table.add_row(str(key), str(value))
    console.print(table)


@cli.command()
@click.argument("url")
def du(url: str) -> None:
    """Total the bytes and objects under a prefix."""
    size, count = total_size(url)
    click.echo(f"{format_size(size)}  ({size} bytes, {count} objects)  {url}")


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
@click.argument("url", default=ROOT)
def browse(url: str) -> None:
    """Open the interactive browser, starting at URL (default: the bucket list)."""
    run_browser(url)


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
