# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directory listings over bucket-routed filesystems.

One :class:`Entry` model serves both the CLI and the TUI, and one navigable tree
spans every backend: the root lists the declared buckets, and each level below is
an ordinary object-store listing routed through
:func:`rigging.filesystem.filesystem_for`.
"""

import dataclasses
from datetime import datetime

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.compression import compression_for

# The root of the browsable tree: the list of declared buckets rather than any one
# filesystem. Not a URL, so it never reaches fsspec.
ROOT = ""

_SCHEME_FOR_STORE = {StoreType.GCS: "gs", StoreType.R2: "s3", StoreType.COREWEAVE: "s3"}

# Preview reads are bounded: browsing should never pull a multi-gigabyte shard down a home
# connection because someone pressed enter on it. `fsutil cp` fetches whole objects.
MAX_PREVIEW_BYTES = 10 * 1024 * 1024


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


def bucket_url(bucket: str) -> str:
    """The URL of *bucket*, with the scheme its declared backend is served by.

    Raises:
        KeyError: if no cluster config declares *bucket*.
    """
    spec = data_buckets()[bucket]
    return f"{_SCHEME_FOR_STORE[spec.store]}://{spec.name}"


def list_entries(url: str) -> list[Entry]:
    """List the immediate children of *url*, directories first then names ascending.

    ``url`` may be :data:`ROOT`, in which case the declared buckets are the children.
    The listed location itself is never included, so a listing is always strictly a
    level down.
    """
    if url == ROOT:
        return [
            Entry(url=bucket_url(name), name=name, size=None, mtime=None, is_dir=True) for name in sorted(data_buckets())
        ]

    parsed = StoragePath(url)
    fs, path = filesystem_for(url)
    entries = [_entry(parsed, item) for item in fs.ls(path, detail=True) if _is_child(path, item["name"])]
    entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
    return entries


def _is_child(listed: str, name: str) -> bool:
    """Whether *name* is a child of the listed path rather than the path itself.

    Object stores commonly report a zero-byte marker object for the prefix being
    listed; it carries no information and would render as an empty-named row.
    """
    return name.strip("/") != listed.strip("/")


def _entry(parsed: StoragePath, item: dict) -> Entry:
    name = item["name"].rstrip("/").rsplit("/", 1)[-1]
    is_dir = item["type"] == "directory"
    return Entry(
        url=str(parsed / name),
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
    """Return ``(bytes, object_count)`` under *url*, walking every prefix beneath it."""
    fs, path = filesystem_for(url)
    if fs.isfile(path):
        return fs.size(path), 1

    total = 0
    count = 0
    for _, _, files in fs.walk(path, detail=True):
        for info in files.values():
            total += info.get("size", 0) or 0
            count += 1
    return total, count
