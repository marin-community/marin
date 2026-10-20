"""
File I/O utilities for handling both local filesystem and cloud storage (S3/GCS).

This module provides a unified interface for file operations that works with:
- Local filesystem paths
- S3 paths (s3://bucket/path)
- Google Cloud Storage paths (gs://bucket/path or gcs://bucket/path)

Uses fsspec for cloud storage abstraction.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger
from marinskyrl.remote_io import filesystem_and_path
from marinskyrl.resource_locator import is_cloud_uri


class DirectoryPublisher(Protocol):
    def __call__(self, local_path: str, cloud_path: str) -> None: ...


@dataclass
class PendingDirectoryUpload:
    """A cloud directory staged on local disk and ready for publication."""

    local_path: str
    output_path: str

    def publish(self) -> None:
        try:
            upload_directory(self.local_path, self.output_path)
        finally:
            try:
                shutil.rmtree(self.local_path)
            except OSError:
                logger.exception(f"Failed to remove checkpoint staging directory {self.local_path}")


class DeferredLocalWorkDir:
    """Stage cloud output locally while leaving publication to the caller."""

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._local_path: str | None = None
        self._cloud_backed = is_cloud_path(output_path)

    def __enter__(self) -> str:
        if self._cloud_backed:
            self._local_path = tempfile.mkdtemp(prefix="skyrl-checkpoint-")
        else:
            makedirs(self.output_path, exist_ok=True)
            self._local_path = self.output_path
        return self._local_path

    def __exit__(self, exc_type, _exc_value, _traceback) -> None:
        if exc_type is not None and self._cloud_backed and self._local_path is not None:
            shutil.rmtree(self._local_path, ignore_errors=True)

    def pending_upload(self) -> PendingDirectoryUpload | None:
        if not self._cloud_backed:
            return None
        assert self._local_path is not None
        return PendingDirectoryUpload(self._local_path, self.output_path)

    def discard(self) -> None:
        if self._cloud_backed and self._local_path is not None:
            shutil.rmtree(self._local_path, ignore_errors=True)


def is_cloud_path(path: str) -> bool:
    """Check if the given path is a cloud storage path."""
    return is_cloud_uri(path)


def _get_filesystem(path: str):
    """Get the appropriate filesystem for the given path."""
    fs, _ = filesystem_and_path(path)
    return fs


def _normalized_path(filesystem, path: str) -> str:
    return filesystem._strip_protocol(path) if is_cloud_path(path) else path


def open_file(path: str, mode: str = "rb"):
    """Open a file using fsspec, works with both local and cloud paths."""
    fs = _get_filesystem(path)
    norm = _normalized_path(fs, path)
    return fs.open(norm, mode)


def write_bytes_atomic(path: str, payload: bytes) -> None:
    """Write one object; local paths use fsync plus atomic replacement."""
    if is_cloud_path(path):
        with open_file(path, "wb") as destination:
            destination.write(payload)
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def read_bytes(path: str) -> bytes:
    """Read one local or cloud object."""
    with open_file(path, "rb") as source:
        return source.read()


def find_files(path: str) -> dict[str, int]:
    """Return recursive file paths and sizes below a local or cloud prefix."""
    filesystem = _get_filesystem(path)
    normalized = _normalized_path(filesystem, path)
    details = filesystem.find(normalized, detail=True, withdirs=False)
    return {str(file_path): int(detail["size"]) for file_path, detail in details.items()}


def file_size(path: str) -> int:
    """Return the size of one exact local or cloud object."""
    filesystem = _get_filesystem(path)
    normalized = _normalized_path(filesystem, path)
    detail = filesystem.info(normalized)
    if detail.get("type") == "directory":
        raise IsADirectoryError(path)
    return int(detail["size"])


def makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories. Only applies to local filesystem paths."""
    if not is_cloud_path(path):
        os.makedirs(path, exist_ok=exist_ok)


def exists(path: str) -> bool:
    """Check if a file or directory exists."""
    fs = _get_filesystem(path)
    return fs.exists(path)


def isdir(path: str) -> bool:
    """Check if path is a directory."""
    fs = _get_filesystem(path)
    return fs.isdir(path)


def list_dir(path: str) -> list[str]:
    """List contents of a directory."""
    fs = _get_filesystem(path)
    return fs.ls(path, detail=False)


def remove(path: str) -> None:
    """Remove a file or directory."""
    fs = _get_filesystem(path)
    if fs.isdir(path):
        fs.rm(path, recursive=True)
    else:
        fs.rm(path)


def _upload(local_path: str, cloud_path: str, *, recursive: bool) -> None:
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Destination must be a cloud path, got: {cloud_path}")
    fs = _get_filesystem(cloud_path)
    is_s3_path = cloud_path.startswith("s3://")
    destination = fs._strip_protocol(cloud_path) if is_s3_path else cloud_path
    try:
        fs.put(local_path, destination, recursive=recursive)
    except Exception as error:
        error.add_note(f"Cloud upload failed from {local_path} to {cloud_path}")
        raise


def upload_file(local_path: str, cloud_path: str) -> None:
    """Upload one local file to cloud storage."""
    _upload(local_path, cloud_path, recursive=False)


def upload_directory(local_path: str, cloud_path: str) -> None:
    """Upload a local directory to cloud storage."""
    _upload(local_path.rstrip("/") + "/", cloud_path, recursive=True)
    logger.info(f"Uploaded {local_path} to {cloud_path}")


def download_directory(cloud_path: str, local_path: str) -> None:
    """Download a cloud directory to local storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    # The trailing separator makes fsspec copy the directory CONTENTS instead of
    # nesting the directory under the destination. It must be appended AFTER
    # _strip_protocol, which rstrips separators and would silently undo it.
    if cloud_path.startswith("s3://"):
        source_path = fs._strip_protocol(cloud_path) + "/"
        fs.get(source_path, local_path, recursive=True)
    else:
        fs.get(cloud_path.rstrip("/") + "/", local_path, recursive=True)
    logger.info(f"Downloaded {cloud_path} to {local_path}")


def download_file(cloud_path: str, local_path: str) -> None:
    """Download one cloud object atomically to a local file."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    partial_path = f"{local_path}.partial"
    try:
        if cloud_path.startswith("s3://"):
            fs.get(fs._strip_protocol(cloud_path), partial_path, recursive=False)
        else:
            fs.get(cloud_path, partial_path, recursive=False)
        os.replace(partial_path, local_path)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    logger.info(f"Downloaded {cloud_path} to {local_path}")


@contextmanager
def local_read_files(input_paths: Sequence[str]):
    """Stage an explicit set of cloud objects locally without reading sibling objects."""
    paths = list(input_paths)
    cloud_paths = [is_cloud_path(path) for path in paths]
    if any(cloud_paths) and not all(cloud_paths):
        raise ValueError("input_paths must be entirely local or entirely cloud-backed")

    if not any(cloud_paths):
        missing_paths = [path for path in paths if not exists(path)]
        if missing_paths:
            raise FileNotFoundError(f"Paths do not exist: {missing_paths}")
        yield paths
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        local_paths = []
        for index, input_path in enumerate(paths):
            local_path = os.path.join(temp_dir, str(index), os.path.basename(input_path))
            download_file(input_path, local_path)
            local_paths.append(local_path)
        yield local_paths


@contextmanager
def local_output_dir(
    output_path: str,
    publisher: DirectoryPublisher,
):
    """Yield ``output_path`` locally, or temporary staging that publishes cloud output on success."""
    if is_cloud_path(output_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            publisher(temp_dir, output_path)
        return

    makedirs(output_path, exist_ok=True)
    yield output_path


@contextmanager
def local_work_dir(output_path: str):
    """Yield direct local output or cloud staging that uploads on success."""
    with local_output_dir(output_path, upload_directory) as work_dir:
        yield work_dir


@contextmanager
def local_read_dir(input_path: str):
    """
    Context manager that provides a local directory with content from input_path.

    For local paths, returns the path directly.
    For cloud paths, downloads content to a temporary directory.

    Args:
        input_path: The source path (local or cloud)

    Yields:
        str: Local directory path containing the content

    Example:
        with local_read_dir("s3://bucket/model") as read_dir:
            # Load files from read_dir
            model = AutoModel.from_pretrained(read_dir)
    """
    if is_cloud_path(input_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download everything from cloud path to temp_dir
            download_directory(input_path, temp_dir)
            logger.info(f"Downloaded directory contents from {input_path}")
            yield temp_dir
    else:
        # For local paths, use directly (but check it exists)
        if not exists(input_path):
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        yield input_path
