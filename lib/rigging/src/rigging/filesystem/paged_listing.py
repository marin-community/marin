# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paged object listings over ordinary S3 and GCS fsspec filesystems.

The filesystem router attaches the specialized operations needed by storage
scans at ``fs.listing``:

- ``level_pages(path)`` returns one delimiter level as ``(file_entries, subdir_paths)``
  pages; each subdir path is re-listable on the same instance.
- ``flat_pages(path)`` returns a flat (recursive) listing as pages of object detail
  dicts covering every object beneath *path*, so a whole subtree can be measured
  or streamed without descending prefix by prefix.

Paths are the protocol-stripped ``bucket/key`` form the filesystems themselves
use, and every listing excludes the listed prefix's own marker object.
"""

from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.timing import ExponentialBackoff, retry_with_backoff

DIRECTORY_TYPE = "directory"

_S3_LISTING_MAX_ATTEMPTS = 4
_S3_LISTING_BACKOFF = ExponentialBackoff(initial=0.5, maximum=5.0, factor=2.0)

# The largest page size the GCS JSON API serves; the server returns fewer plus a
# continuation token when a page would exceed its own limits.
_GCS_PAGE_SIZE = 5000


def is_child(listed: str, name: str) -> bool:
    """Whether *name* differs from *listed* after trimming separator markers.

    Object stores commonly report a zero-byte marker object for the prefix being
    listed; it carries no information and would render as an empty-named row.
    """
    return name.strip("/") != listed.strip("/")


def s3_listing_page(
    fs, path: str, continuation_token: str | None, delimiter: str
) -> tuple[list[dict[str, Any]], str | None]:
    """One ListObjectsV2 page under *path*: ``(detail dicts, next token)``.

    ``delimiter`` of ``"/"`` lists one level (subdirectories arrive as
    directory-type entries); ``""`` lists the subtree flat.
    """
    bucket, key, _ = fs.split_path(path)
    prefix = key if not key or key.endswith("/") else f"{key}/"
    kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": delimiter}
    if continuation_token is not None:
        kwargs["ContinuationToken"] = continuation_token
    response = retry_with_backoff(
        lambda: fs.call_s3("list_objects_v2", **kwargs),
        retryable=is_transient_s3_error,
        max_attempts=_S3_LISTING_MAX_ATTEMPTS,
        backoff=_S3_LISTING_BACKOFF,
        operation=f"ListObjectsV2 {path}",
    )
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


def _gcs_listing_page(
    fs: Any, path: str, continuation_token: str | None, delimiter: str
) -> tuple[list[dict[str, Any]], str | None]:
    bucket, key, _ = fs.split_path(path)
    prefix = key if not key or key.endswith("/") else f"{key}/"
    page = fs.call(
        "GET",
        "b/{}/o",
        bucket,
        prefix=prefix or None,
        delimiter=delimiter or None,
        maxResults=_GCS_PAGE_SIZE,
        pageToken=continuation_token,
        json_out=True,
    )
    entries = [{"name": f"{bucket}/{item}", "size": 0, "type": DIRECTORY_TYPE} for item in page.get("prefixes", [])]
    entries.extend(fs._process_object(bucket, item) for item in page.get("items", []))
    return entries, page.get("nextPageToken")


def _split_level(listed: str, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Partition one level's entries into object detail dicts and subdirectory paths."""
    files: list[dict[str, Any]] = []
    subdirs: list[str] = []
    for entry in entries:
        if not is_child(listed, entry["name"]):
            continue
        if entry.get("type") == DIRECTORY_TYPE:
            subdirs.append(entry["name"])
        else:
            files.append(entry)
    return files, subdirs


class _PagedListing:
    """Paged object listings backed by a filesystem-specific page function."""

    def __init__(self, listing_page: Callable[[str, str | None, str], tuple[list[dict[str, Any]], str | None]]) -> None:
        self._listing_page = listing_page

    def level_pages(self, path: str) -> Iterator[tuple[list[dict[str, Any]], list[str]]]:
        for entries in self._pages(path, "/"):
            yield _split_level(path, entries)

    def flat_pages(self, path: str) -> Iterator[list[dict[str, Any]]]:
        for entries in self._pages(path, ""):
            yield [entry for entry in entries if is_child(path, entry["name"])]

    def _pages(self, path: str, delimiter: str) -> Iterator[list[dict[str, Any]]]:
        token: str | None = None
        while True:
            entries, token = self._listing_page(path, token, delimiter)
            yield entries
            if token is None:
                return


def with_listing(fs: Any) -> Any:
    """Attach backend-aware paged listing operations to a cloud filesystem."""
    protocol = getattr(fs, "protocol", ())
    protocols = (protocol,) if isinstance(protocol, str) else protocol
    if "s3" in protocols:
        fs.listing = _PagedListing(partial(s3_listing_page, fs))
    elif any(item in ("gs", "gcs") for item in protocols):
        fs.listing = _PagedListing(partial(_gcs_listing_page, fs))
    return fs
