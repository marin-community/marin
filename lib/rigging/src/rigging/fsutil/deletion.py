# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streamed parallel removal of object-store prefixes.

Deletion is driven by :func:`rigging.fsutil.listing.metadata_listing_pages`, which lists
prefixes in parallel and yields each page as it lands. Objects are removed as the listing
discovers them, so the first delete leaves before a large prefix finishes listing and
memory stays bounded by the requests in flight rather than by the size of the namespace.

Each submitted batch is exactly one delete request, so ``workers`` means concurrent
delete requests on every backend.
"""

import dataclasses
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Protocol, cast

from fsspec import AbstractFileSystem

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.protocols import is_gcs_filesystem, is_s3_filesystem
from rigging.filesystem.s3_errors import is_transient_s3_error, is_transient_s3_error_code
from rigging.fsutil.listing import DEFAULT_LISTING_WORKERS, DIRECTORY_TYPE, metadata_listing_pages
from rigging.timing import ExponentialBackoff, retry_with_backoff

# Deletes are network-bound, but a bucket enforces its own write ceiling: GCS admits
# roughly a thousand writes per second before it throttles. Sixteen requests in flight
# saturates that without spending the run inside backoff.
DEFAULT_DELETE_WORKERS = 16
MAX_DELETE_WORKERS = 256

# Objects per request, at each backend's documented maximum: DeleteObjects accepts a
# thousand keys, and the GCS batch endpoint accepts a hundred sub-requests.
S3_DELETE_BATCH = 1000
GCS_DELETE_BATCH = 100

_DELETE_MAX_ATTEMPTS = 4
_DELETE_BACKOFF = ExponentialBackoff(initial=0.5, maximum=5.0, factor=2.0)

# gcsfs serves listings and deletes from one session, whose aiohttp connector admits 100
# sockets by default. A listing sized for its own command would hold most of the pool and
# leave the deletes queued behind it. s3fs sizes its pool per command, so it needs no cap.
_GCS_CONNECTION_LIMIT = 100


class _ThrottledBulkDelete(Exception):
    """Every key in one bulk delete failed with a transient code."""


class _BulkDeleteFailed(Exception):
    """At least one key in a bulk delete failed with a permanent code."""


def _is_retryable_delete(error: BaseException) -> bool:
    """Classify a bulk delete failure.

    The two outcomes this module raises are decided by type. Anything else came from the
    transport, which :func:`is_transient_s3_error` classifies. That classifier falls back
    to a substring scan, so a permanent failure whose message quotes a throttling code
    would otherwise read as transient and re-send the whole batch.
    """
    if isinstance(error, _ThrottledBulkDelete):
        return True
    if isinstance(error, _BulkDeleteFailed):
        return False
    return is_transient_s3_error(error)


class _BatchDeleteFilesystem(Protocol):
    """gcsfs's delete signature, which fsspec's base class does not declare.

    ``batchsize`` is the number of sub-requests gcsfs puts in one batch HTTP round trip.
    """

    def rm(self, path: list[str], recursive: bool = False, maxdepth: int | None = None, batchsize: int = 20) -> None: ...


@dataclasses.dataclass(frozen=True)
class DeleteProgress:
    """Monotonic progress observed as delete requests complete."""

    objects_deleted: int
    bytes_deleted: int
    objects_listed: int
    listing_pages: int
    requests_active: int
    requests_total: int
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class DeleteResult:
    """Totals for one completed prefix removal."""

    url: str
    objects_deleted: int
    bytes_deleted: int
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class _Batch:
    paths: list[str]
    size_bytes: int


@dataclasses.dataclass
class _Listed:
    pages: int = 0
    objects: int = 0


@dataclasses.dataclass
class _Deleted:
    objects: int = 0
    size_bytes: int = 0


def delete_prefix(
    url: str,
    *,
    workers: int = DEFAULT_DELETE_WORKERS,
    listing_workers: int | None = None,
    progress: Callable[[DeleteProgress], None] | None = None,
) -> DeleteResult:
    """Remove every object below *url*, deleting while the listing streams.

    Args:
        url: Prefix to empty.
        workers: Delete requests in flight at one time.
        listing_workers: Threads that list prefixes. Derived from the backend's
            connection pool when omitted.
        progress: Called as each delete request completes.

    Returns:
        The totals for the removal.
    """
    fs, _ = filesystem_for(url)
    batch_size = _delete_batch_size(fs)
    listing_workers = _resolve_listing_workers(fs, workers, listing_workers)
    listed = _Listed()
    deleted = _Deleted()
    started = time.monotonic()

    def report(requests_active: int) -> None:
        if progress is None:
            return
        progress(
            DeleteProgress(
                objects_deleted=deleted.objects,
                bytes_deleted=deleted.size_bytes,
                objects_listed=listed.objects,
                listing_pages=listed.pages,
                requests_active=requests_active,
                requests_total=workers,
                elapsed_seconds=time.monotonic() - started,
            )
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future, _Batch] = {}
        for batch in _object_batches(url, batch_size, listing_workers, listed):
            while len(pending) >= workers:
                _drain(pending, deleted, report)
            pending[executor.submit(_delete_batch, fs, batch.paths)] = batch
        while pending:
            _drain(pending, deleted, report)

    fs.invalidate_cache()
    return DeleteResult(
        url=url,
        objects_deleted=deleted.objects,
        bytes_deleted=deleted.size_bytes,
        elapsed_seconds=time.monotonic() - started,
    )


def _object_batches(url: str, batch_size: int, listing_workers: int, listed: _Listed) -> Iterator[_Batch]:
    """Yield full delete batches as the streamed listing fills them."""
    paths: list[str] = []
    size_bytes = 0
    for page in metadata_listing_pages(url, workers=listing_workers):
        listed.pages = page.pages_completed
        for entry in page.entries:
            if entry["type"] == DIRECTORY_TYPE:
                continue
            paths.append(entry["name"])
            size_bytes += entry.get("size", 0) or 0
            listed.objects += 1
            if len(paths) == batch_size:
                yield _Batch(paths=paths, size_bytes=size_bytes)
                paths = []
                size_bytes = 0
    if paths:
        yield _Batch(paths=paths, size_bytes=size_bytes)


def _drain(pending: dict[Future, _Batch], deleted: _Deleted, report: Callable[[int], None]) -> None:
    """Retire the first finished requests, freeing their slots for new batches."""
    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in finished:
        batch = pending.pop(future)
        future.result()
        deleted.objects += len(batch.paths)
        deleted.size_bytes += batch.size_bytes
        report(len(pending))


def _resolve_listing_workers(fs: AbstractFileSystem, workers: int, requested: int | None) -> int:
    """How many threads may list while *workers* delete requests are in flight."""
    if requested is not None:
        return requested
    if is_gcs_filesystem(fs):
        return max(1, _GCS_CONNECTION_LIMIT - workers)
    return DEFAULT_LISTING_WORKERS


def _delete_batch_size(fs: AbstractFileSystem) -> int:
    """Objects per delete request, so that one batch costs one request."""
    if is_s3_filesystem(fs):
        return S3_DELETE_BATCH
    if is_gcs_filesystem(fs):
        return GCS_DELETE_BATCH
    return 1


def _delete_batch(fs: AbstractFileSystem, paths: list[str]) -> None:
    if is_s3_filesystem(fs):
        _delete_s3_objects(fs, paths)
        return
    if is_gcs_filesystem(fs):
        # gcsfs splits the list into batch sub-requests of its own, so sending a whole
        # batch keeps the call to a single round trip.
        cast(_BatchDeleteFilesystem, fs).rm(paths, batchsize=GCS_DELETE_BATCH)
        return
    fs.rm(paths)


def _delete_s3_objects(fs: AbstractFileSystem, paths: list[str]) -> None:
    objects = []
    buckets = set()
    for path in paths:
        bucket, key, version = fs.split_path(path)
        buckets.add(bucket)
        item = {"Key": key}
        if version is not None:
            item["VersionId"] = version
        objects.append(item)
    assert len(buckets) == 1
    bucket = buckets.pop()
    retry_with_backoff(
        lambda: _delete_s3_batch(fs, bucket, objects),
        retryable=_is_retryable_delete,
        max_attempts=_DELETE_MAX_ATTEMPTS,
        backoff=_DELETE_BACKOFF,
        operation=f"DeleteObjects {bucket}",
    )


def _delete_s3_batch(fs: AbstractFileSystem, bucket: str, objects: list[dict[str, str]]) -> None:
    """Delete one batch, treating a throttled key as a failure of the whole request.

    S3 answers 200 and names each failed key in ``Errors``, so throttling never raises on
    its own. The retry repeats the whole batch, which is safe because deleting a key that
    is already gone succeeds.
    """
    response = fs.call_s3("delete_objects", Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
    errors = response.get("Errors", [])
    if not errors:
        return
    details = ", ".join(f"{error['Key']}: {error['Code']}" for error in errors)
    if all(is_transient_s3_error_code(error.get("Code")) for error in errors):
        raise _ThrottledBulkDelete(f"S3 bulk delete throttled: {details}")
    raise _BulkDeleteFailed(f"S3 bulk delete failed: {details}")
