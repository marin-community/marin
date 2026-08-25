# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-aware bulk deletion attached to bucket-routed filesystems."""

from collections.abc import Callable
from functools import partial
from typing import Any, NotRequired, Protocol, TypedDict, cast

from rigging.filesystem.protocols import normalize_protocols
from rigging.filesystem.s3_errors import is_transient_s3_error, is_transient_s3_error_code
from rigging.timing import ExponentialBackoff, retry_with_backoff

_S3_BATCH_SIZE = 1000
_GCS_BATCH_SIZE = 100
_GCS_CONNECTION_LIMIT = 100
_DELETE_MAX_ATTEMPTS = 4
_DELETE_BACKOFF = ExponentialBackoff(initial=0.5, maximum=5.0, factor=2.0)


class _S3DeleteError(TypedDict):
    Key: str
    Code: str


class _S3DeleteObject(TypedDict):
    Key: str
    VersionId: NotRequired[str]


class _S3DeleteResponse(TypedDict):
    Errors: NotRequired[list[_S3DeleteError]]


class _S3BulkDeleteFilesystem(Protocol):
    def split_path(self, path: str) -> tuple[str, str, str | None]: ...

    def call_s3(self, method: str, **kwargs: object) -> _S3DeleteResponse: ...


class _GCSBulkDeleteFilesystem(Protocol):
    def rm(self, path: list[str], recursive: bool = False, maxdepth: int | None = None, batchsize: int = 20) -> None: ...


class _ThrottledBulkDelete(Exception):
    """Every key in one bulk delete failed with a transient code."""


class _BulkDeleteFailed(Exception):
    """At least one key in a bulk delete failed with a permanent code."""


class _BulkDeletion:
    """One backend's bulk-delete operation and request limits."""

    def __init__(
        self,
        batch_size: int,
        delete: Callable[[list[str]], None],
        *,
        shared_connection_limit: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self._delete = delete
        self._shared_connection_limit = shared_connection_limit

    def delete(self, paths: list[str]) -> None:
        """Delete one request-sized batch of protocol-stripped paths."""
        self._delete(paths)

    def listing_workers(self, delete_workers: int, requested: int | None, default: int) -> int:
        """Return listing concurrency alongside the active delete requests."""
        if requested is not None:
            return requested
        if self._shared_connection_limit is None:
            return default
        return max(1, self._shared_connection_limit - delete_workers)


def _is_retryable_delete(error: BaseException) -> bool:
    if isinstance(error, _ThrottledBulkDelete):
        return True
    if isinstance(error, _BulkDeleteFailed):
        return False
    return is_transient_s3_error(error)


def _delete_s3_objects(fs: _S3BulkDeleteFilesystem, paths: list[str]) -> None:
    objects: list[_S3DeleteObject] = []
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


def _delete_s3_batch(fs: _S3BulkDeleteFilesystem, bucket: str, objects: list[_S3DeleteObject]) -> None:
    """Delete one S3 batch and surface failures reported inside a 200 response."""
    response = fs.call_s3("delete_objects", Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
    errors = response.get("Errors", [])
    if not errors:
        return
    details = ", ".join(f"{error['Key']}: {error['Code']}" for error in errors)
    if all(is_transient_s3_error_code(error.get("Code")) for error in errors):
        raise _ThrottledBulkDelete(f"S3 bulk delete throttled: {details}")
    raise _BulkDeleteFailed(f"S3 bulk delete failed: {details}")


def _delete_gcs_objects(fs: _GCSBulkDeleteFilesystem, paths: list[str]) -> None:
    fs.rm(paths, batchsize=_GCS_BATCH_SIZE)


def with_bulk_deletion(fs: Any) -> Any:
    """Attach backend-aware bulk deletion at ``fs.deletion``."""
    protocols = normalize_protocols(getattr(fs, "protocol", ()))
    if "s3" in protocols:
        fs.deletion = _BulkDeletion(
            _S3_BATCH_SIZE,
            partial(_delete_s3_objects, cast(_S3BulkDeleteFilesystem, fs)),
        )
    elif any(item in ("gs", "gcs") for item in protocols):
        fs.deletion = _BulkDeletion(
            _GCS_BATCH_SIZE,
            partial(_delete_gcs_objects, cast(_GCSBulkDeleteFilesystem, fs)),
            shared_connection_limit=_GCS_CONNECTION_LIMIT,
        )
    else:
        fs.deletion = _BulkDeletion(1, fs.rm)
    return fs
