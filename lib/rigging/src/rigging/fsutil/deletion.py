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

from rigging.filesystem.buckets import filesystem_for
from rigging.fsutil.listing import DEFAULT_LISTING_WORKERS, DIRECTORY_TYPE, metadata_listing_pages

# Deletes are network-bound, but a bucket enforces its own write ceiling: GCS admits
# roughly a thousand writes per second before it throttles. Sixteen requests in flight
# saturates that without spending the run inside backoff.
DEFAULT_DELETE_WORKERS = 16
MAX_DELETE_WORKERS = 256


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
    listing_workers = fs.deletion.listing_workers(workers, listing_workers, DEFAULT_LISTING_WORKERS)
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
        for batch in _object_batches(url, fs.deletion.batch_size, listing_workers, listed):
            while len(pending) >= workers:
                _drain(pending, deleted, report)
            pending[executor.submit(fs.deletion.delete, batch.paths)] = batch
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
