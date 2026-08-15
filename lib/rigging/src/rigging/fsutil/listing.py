# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directory listings over bucket-routed filesystems.

One :class:`Entry` model serves both the CLI and the TUI, and one navigable tree
spans every backend: the root lists the declared buckets, and each level below is
an ordinary object-store listing routed through
:func:`rigging.filesystem.buckets.filesystem_for`.
"""

import dataclasses
import glob
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.compression import compression_for

# The root of the browsable tree: the list of declared buckets rather than any one
# filesystem. Not a URL, so it never reaches fsspec.
ROOT = ""

_SCHEME_FOR_STORE = {StoreType.GCS: "gs", StoreType.R2: "s3", StoreType.COREWEAVE: "s3"}
DIRECTORY_TYPE = "directory"

# Preview reads are bounded: browsing should never pull a multi-gigabyte shard down a home
# connection because someone pressed enter on it. `fsutil cp` fetches whole objects.
MAX_PREVIEW_BYTES = 10 * 1024 * 1024

# Bucket listings are network-bound.
DEFAULT_LISTING_WORKERS = 128
_S3_MAX_SPLIT_DEPTH = 3


@dataclasses.dataclass(frozen=True)
class Entry:
    """One child of a listed location.

    Attributes:
        url: Full URL of the child, navigable on its own.
        name: Display name relative to the listed location.
        size: Object size in bytes; ``None`` for a directory or bucket.
        mtime: Last modification time, when the backend reports one.
        is_dir: Whether the child has children (a prefix or a bucket).
    """

    url: str
    name: str
    size: int | None
    mtime: datetime | None
    is_dir: bool


@dataclasses.dataclass(frozen=True)
class ListingPage:
    """One metadata page and aggregate progress for its listing."""

    path: str
    entries: list[dict[str, Any]]
    pages_completed: int
    prefixes_completed: int
    prefixes_discovered: int
    workers_active: int
    workers_total: int


@dataclasses.dataclass(frozen=True)
class Preview:
    data: bytes
    truncated: bool
    full_size: int | None


@dataclasses.dataclass(frozen=True)
class _ListingStats:
    size: int
    count: int
    directories: list[str]


@dataclasses.dataclass(frozen=True)
class _CompletedDirectoryListing:
    path: str
    entries: list[dict[str, Any]]
    workers_active: int


class _S3ListingMode(StrEnum):
    PROBE = "probe"
    EXPAND = "expand"
    FLAT = "flat"


@dataclasses.dataclass(frozen=True)
class _S3ListingTask:
    path: str
    mode: _S3ListingMode
    depth: int = 0
    continuation_token: str | None = None


def bucket_url(bucket: str) -> str:
    """The URL of *bucket*, with the scheme its declared backend is served by.

    Raises:
        KeyError: if no cluster config declares *bucket*.
    """
    spec = data_buckets()[bucket]
    return f"{_SCHEME_FOR_STORE[spec.store]}://{spec.name}"


def list_entries(url: str, *, recursive: bool = False) -> list[Entry]:
    """List *url*'s immediate children or the entries matching a glob pattern.

    ``url`` may be :data:`ROOT`, in which case the declared buckets are the children.
    Glob matches are named relative to the non-pattern prefix, so matches remain
    distinguishable when their basenames are the same. Recursive listings of literal
    URLs name every descendant relative to ``url``; recursive root and glob listings
    retain their ordinary non-recursive behavior.
    """
    if url == ROOT:
        return [
            Entry(url=bucket_url(name), name=name, size=None, mtime=None, is_dir=True) for name in sorted(data_buckets())
        ]

    parsed = StoragePath(url)
    fs, path = filesystem_for(url)
    if recursive and not glob.has_magic(path):
        entries = []
        for item in fs.find(path, detail=True, withdirs=True).values():
            if not is_child(path, item["name"]):
                continue
            qualified_url = _qualified_url(parsed, item["name"])
            name = _relative_name(item["name"], path)
            entries.append(_entry(parsed, item, name=name, url=qualified_url))
    elif glob.has_magic(path):
        root = _glob_root(path)
        entries = []
        for item in fs.glob(path, detail=True).values():
            qualified_url = _qualified_url(parsed, item["name"])
            name = _relative_name(item["name"], root)
            entries.append(_entry(parsed, item, name=name, url=qualified_url))
    else:
        entries = [_entry(parsed, item) for item in fs.ls(path, detail=True) if is_child(path, item["name"])]
    entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
    return entries


def _glob_root(pattern: str) -> str:
    first_magic = min(pattern.find(character) for character in "*?[" if character in pattern)
    return pattern[:first_magic].rpartition("/")[0]


def _qualified_url(parsed: StoragePath, path: str) -> str:
    if parsed.scheme is None or "://" in path:
        return path
    return f"{parsed.scheme}://{path}"


def _relative_name(name: str, root: str) -> str:
    rooted = root.startswith("/")
    root = root.rstrip("/")
    if name == root:
        return ""
    prefix = f"{root}/" if root else "/" if rooted else ""
    if not name.startswith(prefix):
        raise ValueError(f"{name} is not under {root}")
    return name[len(prefix) :]


def is_child(listed: str, name: str) -> bool:
    """Whether *name* is a child of the listed path rather than the path itself.

    Object stores commonly report a zero-byte marker object for the prefix being
    listed; it carries no information and would render as an empty-named row.
    """
    return name.strip("/") != listed.strip("/")


def _entry(parsed: StoragePath, item: dict, *, name: str | None = None, url: str | None = None) -> Entry:
    item_name = item["name"].rstrip("/")
    name = name or item_name.rsplit("/", 1)[-1]
    is_dir = item["type"] == DIRECTORY_TYPE
    return Entry(
        url=url or str(parsed / name),
        name=name,
        size=None if is_dir else item.get("size", 0),
        mtime=entry_mtime(item),
        is_dir=is_dir,
    )


def entry_mtime(item: dict) -> datetime | None:
    """The entry's modification time under whichever key the backend uses.

    gcsfs reports ``updated``/``mtime``, s3fs ``LastModified``; directories carry none.
    """
    for key in ("LastModified", "mtime", "updated"):
        value = item.get(key)
        if isinstance(value, datetime):
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return None


def parent_url(url: str) -> str:
    """The location one level up: a prefix's parent, a bucket's :data:`ROOT`, or ``ROOT``."""
    if url == ROOT:
        return ROOT
    parsed = StoragePath(url)
    if not parsed.segments:
        return ROOT
    return str(parsed.parent)


def read_preview(url: str) -> Preview:
    """Read a bounded preview of the stored bytes in *url*."""
    fs, path = filesystem_for(url)
    return _read_preview(fs, path, compression=None, full_size=fs.size(path))


def read_decompressed_preview(url: str) -> Preview:
    """Read a bounded preview and decompress supported file suffixes."""
    fs, path = filesystem_for(url)
    compression = compression_for(path)
    full_size = fs.size(path) if compression is None else None
    return _read_preview(fs, path, compression=compression, full_size=full_size)


def _read_preview(fs, path: str, *, compression: str | None, full_size: int | None) -> Preview:
    with fs.open(path, "rb", compression=compression) as file:
        data = file.read(MAX_PREVIEW_BYTES + 1)
    return Preview(data=data[:MAX_PREVIEW_BYTES], truncated=len(data) > MAX_PREVIEW_BYTES, full_size=full_size)


def total_size(url: str) -> tuple[int, int]:
    """Return ``(bytes, object_count)`` under *url*."""
    fs, path = _listing_filesystem(url, DEFAULT_LISTING_WORKERS)
    info = fs.info(path)
    if info["type"] != DIRECTORY_TYPE:
        return info.get("size", 0) or 0, 1

    total = 0
    count = 0
    for page in _metadata_listing_pages(fs, path, DEFAULT_LISTING_WORKERS):
        stats = _listing_stats(page.path, page.entries)
        total += stats.size
        count += stats.count
    return total, count


def metadata_listing_pages(url: str, *, workers: int = DEFAULT_LISTING_WORKERS) -> Iterator[ListingPage]:
    """Yield independent metadata pages covering every object below *url*."""
    fs, path = _listing_filesystem(url, workers)
    yield from _metadata_listing_pages(fs, path, workers)


def _listing_filesystem(url: str, workers: int) -> tuple[Any, str]:
    fs, path = filesystem_for(url)
    if _is_s3_filesystem(fs):
        fs.config_kwargs = {**fs.config_kwargs, "max_pool_connections": workers}
    return fs, path


def _is_s3_filesystem(fs) -> bool:
    protocol = getattr(fs, "protocol", ())
    protocols = (protocol,) if isinstance(protocol, str) else protocol
    return "s3" in protocols


def _metadata_listing_pages(fs, path: str, workers: int) -> Iterator[ListingPage]:
    if _is_s3_filesystem(fs):
        yield from _s3_listing_pages(fs, path, workers)
        return

    root_entries = fs.ls(path, detail=True)
    directories = _listing_stats(path, root_entries).directories
    prefixes_discovered = len(directories) + 1
    yield ListingPage(
        path=path,
        entries=root_entries,
        pages_completed=1,
        prefixes_completed=1,
        prefixes_discovered=prefixes_discovered,
        workers_active=1,
        workers_total=workers,
    )

    pages_completed = 1
    prefixes_completed = 1
    for completed in _directory_listing_pages(fs, directories, workers):
        pages_completed += 1
        prefixes_completed += 1
        prefixes_discovered += len(_listing_stats(completed.path, completed.entries).directories)
        yield ListingPage(
            path=completed.path,
            entries=completed.entries,
            pages_completed=pages_completed,
            prefixes_completed=prefixes_completed,
            prefixes_discovered=prefixes_discovered,
            workers_active=completed.workers_active,
            workers_total=workers,
        )


def _directory_listing_pages(fs, directories: Iterable[str], workers: int) -> Iterator[_CompletedDirectoryListing]:
    queued = deque(directories)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        while queued or pending:
            while queued and len(pending) < workers:
                directory = queued.popleft()
                pending[executor.submit(fs.ls, directory, detail=True)] = directory

            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                directory = pending.pop(future)
                entries = future.result()
                queued.extend(_listing_stats(directory, entries).directories)
                workers_active = len(pending) + 1
                yield _CompletedDirectoryListing(directory, entries, workers_active)


def _s3_listing_pages(fs, path: str, workers: int) -> Iterator[ListingPage]:
    queued = deque([_S3ListingTask(path, _S3ListingMode.PROBE)])
    scheduled_prefixes = {path}
    pages_completed = 0
    prefixes_completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        while queued or pending:
            while queued and len(pending) < workers:
                task = queued.popleft()
                delimiter = "/" if task.mode == _S3ListingMode.EXPAND else ""
                future = executor.submit(_s3_listing_page, fs, task.path, task.continuation_token, delimiter)
                pending[future] = task

            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                task = pending.pop(future)
                entries, continuation_token = future.result()
                pages_completed += 1
                if task.mode == _S3ListingMode.PROBE:
                    if continuation_token is None:
                        prefixes_completed += 1
                    elif task.depth < _S3_MAX_SPLIT_DEPTH:
                        entries = []
                        queued.appendleft(_S3ListingTask(task.path, _S3ListingMode.EXPAND, depth=task.depth))
                    else:
                        queued.appendleft(
                            _S3ListingTask(
                                task.path,
                                _S3ListingMode.FLAT,
                                depth=task.depth,
                                continuation_token=continuation_token,
                            )
                        )
                else:
                    if continuation_token is None:
                        prefixes_completed += 1
                    else:
                        queued.appendleft(
                            _S3ListingTask(
                                task.path,
                                task.mode,
                                depth=task.depth,
                                continuation_token=continuation_token,
                            )
                        )
                    if task.mode == _S3ListingMode.EXPAND:
                        for directory in _listing_stats(task.path, entries).directories:
                            if directory not in scheduled_prefixes:
                                scheduled_prefixes.add(directory)
                                queued.append(_S3ListingTask(directory, _S3ListingMode.PROBE, depth=task.depth + 1))

                yield ListingPage(
                    path=task.path,
                    entries=entries,
                    pages_completed=pages_completed,
                    prefixes_completed=prefixes_completed,
                    prefixes_discovered=len(scheduled_prefixes),
                    workers_active=len(pending) + 1,
                    workers_total=workers,
                )


def _s3_listing_page(
    fs, path: str, continuation_token: str | None, delimiter: str
) -> tuple[list[dict[str, Any]], str | None]:
    bucket, key, _ = fs.split_path(path)
    prefix = key if not key or key.endswith("/") else f"{key}/"
    kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": delimiter}
    if continuation_token is not None:
        kwargs["ContinuationToken"] = continuation_token
    response = fs.call_s3("list_objects_v2", **kwargs)
    entries = [
        {"name": f"{bucket}/{item['Prefix']}", "size": 0, "type": DIRECTORY_TYPE}
        for item in response.get("CommonPrefixes", [])
    ]
    entries.extend(
        {
            "name": f"{bucket}/{item['Key']}",
            "size": int(item.get("Size") or 0),
            "type": "file",
            "LastModified": item.get("LastModified"),
        }
        for item in response.get("Contents", [])
    )
    return entries, response.get("NextContinuationToken")


def _listing_stats(listed: str, entries: list[dict[str, Any]]) -> _ListingStats:
    total = 0
    count = 0
    directories = []
    for entry in entries:
        name = entry["name"]
        if not is_child(listed, name):
            continue
        if entry["type"] == DIRECTORY_TYPE:
            directories.append(name)
            continue
        total += entry.get("size", 0) or 0
        count += 1
    return _ListingStats(size=total, count=count, directories=directories)
