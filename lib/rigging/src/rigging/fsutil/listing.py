# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directory listings over bucket-routed filesystems.

One :class:`Entry` model serves both the CLI and the TUI, and one navigable tree
spans every backend: the root lists the declared buckets, and each level below is
an ordinary object-store listing routed through
:func:`rigging.filesystem.filesystem_for`.
"""

import dataclasses
import glob
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.compression import compression_for

# The root of the browsable tree: the list of declared buckets rather than any one
# filesystem. Not a URL, so it never reaches fsspec.
ROOT = ""

_SCHEME_FOR_STORE = {StoreType.GCS: "gs", StoreType.R2: "s3", StoreType.COREWEAVE: "s3"}
_DIRECTORY_TYPE = "directory"

# Preview reads are bounded: browsing should never pull a multi-gigabyte shard down a home
# connection because someone pressed enter on it. `fsutil cp` fetches whole objects.
MAX_PREVIEW_BYTES = 10 * 1024 * 1024

# Bucket listings are network-bound, so a modest pool hides request latency without
# creating enough concurrent requests to overwhelm an object-store endpoint.
_DU_WORKERS = 32


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
class Preview:
    data: bytes
    truncated: bool
    full_size: int | None


@dataclasses.dataclass(frozen=True)
class _ListingStats:
    size: int
    count: int
    directories: list[str]


def bucket_url(bucket: str) -> str:
    """The URL of *bucket*, with the scheme its declared backend is served by.

    Raises:
        KeyError: if no cluster config declares *bucket*.
    """
    spec = data_buckets()[bucket]
    return f"{_SCHEME_FOR_STORE[spec.store]}://{spec.name}"


def list_entries(url: str) -> list[Entry]:
    """List *url*'s immediate children or the entries matching a glob pattern.

    ``url`` may be :data:`ROOT`, in which case the declared buckets are the children.
    Glob matches are named relative to the non-pattern prefix, so matches remain
    distinguishable when their basenames are the same.
    """
    if url == ROOT:
        return [
            Entry(url=bucket_url(name), name=name, size=None, mtime=None, is_dir=True) for name in sorted(data_buckets())
        ]

    parsed = StoragePath(url)
    fs, path = filesystem_for(url)
    if glob.has_magic(path):
        root = _glob_root(path)
        base = StoragePath(_qualified_url(parsed, root))
        entries = []
        for item in fs.glob(path, detail=True).values():
            qualified_url = _qualified_url(parsed, item["name"])
            name = StoragePath(qualified_url).relative_to(base)
            entries.append(_entry(parsed, item, name=name, url=qualified_url))
    else:
        entries = [_entry(parsed, item) for item in fs.ls(path, detail=True) if _is_child(path, item["name"])]
    entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
    return entries


def _glob_root(pattern: str) -> str:
    first_magic = min(pattern.find(character) for character in "*?[" if character in pattern)
    return pattern[:first_magic].rpartition("/")[0]


def _qualified_url(parsed: StoragePath, path: str) -> str:
    if parsed.scheme is None or "://" in path:
        return path
    return f"{parsed.scheme}://{path}"


def _is_child(listed: str, name: str) -> bool:
    """Whether *name* is a child of the listed path rather than the path itself.

    Object stores commonly report a zero-byte marker object for the prefix being
    listed; it carries no information and would render as an empty-named row.
    """
    return name.strip("/") != listed.strip("/")


def _entry(parsed: StoragePath, item: dict, *, name: str | None = None, url: str | None = None) -> Entry:
    item_name = item["name"].rstrip("/")
    name = name or item_name.rsplit("/", 1)[-1]
    is_dir = item["type"] == _DIRECTORY_TYPE
    return Entry(
        url=url or str(parsed / name),
        name=name,
        size=None if is_dir else item.get("size", 0),
        mtime=_mtime(item),
        is_dir=is_dir,
    )


def _mtime(item: dict) -> datetime | None:
    """The entry's modification time under whichever key the backend uses.

    gcsfs reports ``updated``/``mtime``, s3fs ``LastModified``; directories carry none.
    """
    for key in ("LastModified", "mtime", "updated"):
        value = item.get(key)
        if isinstance(value, datetime):
            return value
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
    fs, path = filesystem_for(url)
    root_entries = fs.ls(path, detail=True)
    if len(root_entries) == 1 and not _is_child(path, root_entries[0]["name"]):
        entry = root_entries[0]
        if entry["type"] != _DIRECTORY_TYPE:
            return entry.get("size", 0) or 0, 1

    root_stats = _listing_stats(path, root_entries)
    total = root_stats.size
    count = root_stats.count
    queued = deque(root_stats.directories)
    if not queued:
        return total, count

    with ThreadPoolExecutor(max_workers=_DU_WORKERS) as executor:
        pending = {}
        while queued or pending:
            while queued and len(pending) < _DU_WORKERS:
                directory = queued.popleft()
                pending[executor.submit(fs.ls, directory, detail=True)] = directory

            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                directory = pending.pop(future)
                stats = _listing_stats(directory, future.result())
                total += stats.size
                count += stats.count
                queued.extend(stats.directories)
    return total, count


def _listing_stats(listed: str, entries: list[dict[str, Any]]) -> _ListingStats:
    total = 0
    count = 0
    directories = []
    for entry in entries:
        name = entry["name"]
        if not _is_child(listed, name):
            continue
        if entry["type"] == _DIRECTORY_TYPE:
            directories.append(name)
            continue
        total += entry.get("size", 0) or 0
        count += 1
    return _ListingStats(size=total, count=count, directories=directories)
