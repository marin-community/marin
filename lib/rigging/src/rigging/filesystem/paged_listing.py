# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""S3 and GCS filesystems with paged listing methods.

Rigging serves these subclasses everywhere a filesystem is returned —
:func:`rigging.filesystem.buckets.filesystem_for` constructs the S3 one for
declared R2/CoreWeave buckets, and :mod:`rigging.filesystem.factory` registers
both as the process-wide fsspec implementations — so any ``s3://`` or ``gs://``
filesystem can stream a huge namespace in bounded pages:

- ``level_pages(path)`` — one delimiter level as ``(file_entries, subdir_paths)``
  pages; each subdir path is re-listable on the same instance.
- ``flat_pages(path)`` — a flat (recursive) listing as pages of object detail
  dicts covering every object beneath *path*, so a whole subtree can be measured
  or streamed without descending prefix by prefix.

Paths are the protocol-stripped ``bucket/key`` form the filesystems themselves
use, and every listing excludes the listed prefix's own marker object.
"""

from collections.abc import Iterator
from typing import Any

import s3fs
from gcsfs.core import GCSFileSystem

from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.timing import ExponentialBackoff, retry_with_backoff

DIRECTORY_TYPE = "directory"

_S3_LISTING_MAX_ATTEMPTS = 4
_S3_LISTING_BACKOFF = ExponentialBackoff(initial=0.5, maximum=5.0, factor=2.0)

# The largest page size the GCS JSON API serves; the server returns fewer plus a
# continuation token when a page would exceed its own limits.
_GCS_FLAT_PAGE_SIZE = 5000


def is_child(listed: str, name: str) -> bool:
    """Whether *name* is a child of the listed path rather than the path itself.

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


class S3ListingFileSystem(s3fs.S3FileSystem):
    """s3fs plus paged listing methods; both shapes page by continuation token."""

    def level_pages(self, path: str) -> Iterator[tuple[list[dict[str, Any]], list[str]]]:
        token: str | None = None
        while True:
            entries, token = s3_listing_page(self, path, token, "/")
            yield _split_level(path, entries)
            if token is None:
                return

    def flat_pages(self, path: str) -> Iterator[list[dict[str, Any]]]:
        token: str | None = None
        while True:
            entries, token = s3_listing_page(self, path, token, "")
            yield [entry for entry in entries if is_child(path, entry["name"])]
            if token is None:
                return


class GcsListingFileSystem(GCSFileSystem):
    """gcsfs plus paged listing methods.

    gcsfs itself exposes only whole-level ``ls`` and all-at-once ``find``, so
    ``flat_pages`` issues the raw ``objects.list`` calls page by page and
    normalizes items the same way ``ls`` does.
    """

    def level_pages(self, path: str) -> Iterator[tuple[list[dict[str, Any]], list[str]]]:
        entries = self.ls(path, detail=True)
        # Drop the dircache entry so a long scan over many prefixes stays bounded.
        self.invalidate_cache(path)
        yield _split_level(path, entries)

    def flat_pages(self, path: str) -> Iterator[list[dict[str, Any]]]:
        bucket, key, _ = self.split_path(path)
        prefix = key if not key or key.endswith("/") else f"{key}/"
        token: str | None = None
        while True:
            page = self.call(
                "GET",
                "b/{}/o",
                bucket,
                prefix=prefix or None,
                maxResults=_GCS_FLAT_PAGE_SIZE,
                pageToken=token,
                json_out=True,
            )
            entries = [self._process_object(bucket, item) for item in page.get("items", [])]
            yield [entry for entry in entries if is_child(path, entry["name"])]
            token = page.get("nextPageToken")
            if token is None:
                return
