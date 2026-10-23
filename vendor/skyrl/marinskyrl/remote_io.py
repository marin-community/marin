"""Shared, bounded remote I/O for Hugging Face and object storage."""

from __future__ import annotations

from collections import deque
from collections.abc import Buffer, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import logging
from typing import Any, cast, Protocol, runtime_checkable

from fsspec.spec import AbstractFileSystem
from rigging.filesystem.factory import filesystem as create_filesystem
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath


logger = logging.getLogger(__name__)

S3_MULTIPART_PART_BYTES = 64 * 2**20
S3_MULTIPART_CONCURRENCY = 4


def create_s3_filesystem(**storage_options: Any) -> AbstractFileSystem:
    """Create an S3 filesystem with Rigging's request bounds and retries."""
    return create_filesystem("s3", **storage_options)


def filesystem_and_path(uri: str) -> tuple[AbstractFileSystem, str]:
    """Resolve a URI through Rigging's guarded fsspec factory."""
    return url_to_fs(uri)


@runtime_checkable
class MultipartS3FileSystem(Protocol):
    protocol: str | tuple[str, ...]

    def split_path(self, path: str) -> tuple[str, str, str | None]: ...

    def call_s3(self, method: str, *args: Any, **kwargs: Any) -> Any: ...


class OutputStream(Protocol):
    closed: bool

    def write(self, payload: Buffer) -> int: ...

    def close(self) -> None: ...


def _close_after_error(stream: OutputStream, error: BaseException, path: str, action: str) -> None:
    try:
        stream.close()
    except Exception as cleanup_error:
        error.add_note(f"Failed to {action} for {path}: {cleanup_error}")


class CommittableStream:
    """Output stream that publishes buffered data only on explicit commit."""

    def commit(self) -> None:
        raise NotImplementedError


class S3MultipartWriteStream(CommittableStream):
    """Bounded multipart writer backed by Rigging's guarded S3 filesystem."""

    part_bytes = S3_MULTIPART_PART_BYTES
    concurrency = S3_MULTIPART_CONCURRENCY

    def __init__(
        self,
        filesystem: MultipartS3FileSystem,
        path: str,
    ) -> None:
        bucket, key, _version_id = filesystem.split_path(path)
        self.filesystem = filesystem
        self.bucket = bucket
        self.key = key
        self.path = path
        self.closed = False
        self._position = 0
        self._buffer = bytearray()
        self._upload_id: str | None = None
        self._next_part_number = 1
        self._completed_parts: list[dict[str, int | str]] = []
        self._pending: deque[tuple[int, Future[dict[str, int | str]]]] = deque()
        self._executor = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="marinskyrl-s3")

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        if self.closed:
            raise ValueError("flush of closed remote stream")

    def write(self, payload: Buffer) -> int:
        if self.closed:
            raise ValueError("write to closed remote stream")
        view = memoryview(payload).cast("B")
        payload_bytes = len(view)
        try:
            while view:
                chunk_bytes = min(self.part_bytes - len(self._buffer), len(view))
                self._buffer.extend(view[:chunk_bytes])
                view = view[chunk_bytes:]
                if len(self._buffer) == self.part_bytes:
                    self._submit_part(bytes(self._buffer))
                    self._buffer.clear()
        except BaseException as error:
            _close_after_error(self, error, self.path, "abort multipart upload")
            raise
        self._position += payload_bytes
        return payload_bytes

    def commit(self) -> None:
        if self.closed:
            raise ValueError("commit of closed remote stream")
        try:
            if self._upload_id is None:
                self.filesystem.call_s3(
                    "put_object",
                    Bucket=self.bucket,
                    Key=self.key,
                    Body=bytes(self._buffer),
                )
                self._buffer.clear()
                self._executor.shutdown(wait=True, cancel_futures=True)
                self.closed = True
                return

            if self._buffer:
                self._submit_part(bytes(self._buffer))
                self._buffer.clear()
            while self._pending:
                self._finish_oldest_part()
            self._executor.shutdown(wait=True)
            self.filesystem.call_s3(
                "complete_multipart_upload",
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self._upload_id,
                MultipartUpload={"Parts": sorted(self._completed_parts, key=lambda part: int(part["PartNumber"]))},
            )
            self.closed = True
        except BaseException as error:
            _close_after_error(self, error, self.path, "abort multipart upload")
            raise

    def close(self) -> None:
        """Discard buffered data and abort an uncommitted multipart upload."""
        if self.closed:
            return
        for _part_number, future in self._pending:
            future.cancel()
        try:
            if self._upload_id is not None:
                self.filesystem.call_s3(
                    "abort_multipart_upload",
                    Bucket=self.bucket,
                    Key=self.key,
                    UploadId=self._upload_id,
                )
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._buffer.clear()
            self.closed = True

    def _submit_part(self, payload: bytes) -> None:
        if self._upload_id is None:
            response = self.filesystem.call_s3(
                "create_multipart_upload",
                Bucket=self.bucket,
                Key=self.key,
            )
            self._upload_id = str(response["UploadId"])

        part_number = self._next_part_number
        self._next_part_number += 1
        future = self._executor.submit(self._upload_part, part_number, payload)
        self._pending.append((part_number, future))
        if len(self._pending) >= self.concurrency:
            self._finish_oldest_part()

    def _upload_part(self, part_number: int, payload: bytes) -> dict[str, int | str]:
        try:
            response = self.filesystem.call_s3(
                "upload_part",
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self._upload_id,
                PartNumber=part_number,
                Body=payload,
            )
        except BaseException as error:
            error.add_note(f"Multipart upload failed for {self.path}: upload_id={self._upload_id} part={part_number}")
            raise
        return {"PartNumber": part_number, "ETag": str(response["ETag"])}

    def _finish_oldest_part(self) -> None:
        _part_number, future = self._pending.popleft()
        self._completed_parts.append(future.result())


def create_output_stream(
    filesystem: AbstractFileSystem,
    path: str,
) -> OutputStream:
    """Create a write stream, using bounded multipart transfer for S3."""
    protocol = getattr(filesystem, "protocol", ())
    protocols = protocol if isinstance(protocol, tuple) else (protocol,)
    if "s3" in protocols or "s3a" in protocols:
        if not isinstance(filesystem, MultipartS3FileSystem):
            raise TypeError("S3 filesystem does not provide multipart operations")
        return S3MultipartWriteStream(filesystem, path)
    return cast(OutputStream, filesystem.open(path, "wb"))


@contextmanager
def manage_output_stream(stream: OutputStream, path: str) -> Generator[OutputStream, None, None]:
    """Commit a successful output stream or close a failed partial write."""
    try:
        yield stream
        if isinstance(stream, CommittableStream):
            stream.commit()
        else:
            stream.close()
    except BaseException as error:
        _close_after_error(stream, error, path, "close remote write")
        error.add_note(f"Object write failed for {path}")
        raise


@contextmanager
def open_output_stream(
    filesystem: AbstractFileSystem,
    path: str,
) -> Generator[OutputStream, None, None]:
    """Open a managed output stream with bounded multipart transfer for S3."""
    stream = create_output_stream(filesystem, path)
    with manage_output_stream(stream, path) as managed:
        yield managed


def abort_multipart_uploads(path: str) -> int:
    """Abort incomplete uploads below a canonical S3 prefix and return their count."""
    checkpoint_path = StoragePath(path)
    if checkpoint_path.scheme != "s3" or str(checkpoint_path) != path or not checkpoint_path.key:
        raise ValueError(f"Expected a canonical S3 checkpoint path, got: {path}")

    filesystem = cast(MultipartS3FileSystem, create_s3_filesystem())
    response = filesystem.call_s3(
        "list_multipart_uploads",
        Bucket=checkpoint_path.bucket,
        Prefix=f"{checkpoint_path.key}/",
    )
    uploads = response.get("Uploads", [])
    for upload in uploads:
        filesystem.call_s3(
            "abort_multipart_upload",
            Bucket=checkpoint_path.bucket,
            Key=upload["Key"],
            UploadId=upload["UploadId"],
        )
    if uploads:
        logger.warning("Aborted %s stale multipart uploads below %s", len(uploads), path)
    return len(uploads)
