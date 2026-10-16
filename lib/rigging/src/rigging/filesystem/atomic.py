# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic local downloads and object-store write-and-rename operations."""

import contextlib
import os
import uuid
from collections.abc import Generator
from typing import Protocol

import rigging.filesystem.factory as factory
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.timing import ExponentialBackoff, retry_with_backoff


class _AtomicFileSystem(Protocol):
    def mv(self, source: str, destination: str, *, recursive: bool) -> object: ...

    def rm(self, path: str) -> object: ...


def unique_temp_path(output_path: str) -> str:
    return f"{output_path}.tmp.{uuid.uuid4().hex}"


def _mv_with_retry(filesystem: _AtomicFileSystem, source: str, destination: str) -> None:
    retry_with_backoff(
        lambda: filesystem.mv(source, destination, recursive=True),
        retryable=is_transient_s3_error,
        max_attempts=4,
        backoff=ExponentialBackoff(initial=1.0, maximum=8.0, factor=2.0),
        operation=f"atomic_rename fs.mv {source} -> {destination}",
    )


@contextlib.contextmanager
def atomic_rename(output_path: str, filesystem: _AtomicFileSystem | None = None) -> Generator[str, None, None]:
    """Write through a temporary sibling and rename it to ``output_path``.

    Callers may pass a configured filesystem to reuse backend-specific options.
    The temporary path is removed after a failed write or rename when possible.
    """
    temp_path = unique_temp_path(output_path)
    if filesystem is None:
        filesystem = factory.url_to_fs(output_path)[0]
    try:
        yield temp_path
        _mv_with_retry(filesystem, temp_path, output_path)
    except Exception:
        with contextlib.suppress(Exception):
            filesystem.rm(temp_path)
        raise


def fetch_file_atomic(source_url: str, destination_path: str) -> bool:
    """Download into a temporary sibling before replacing ``destination_path``.

    Returns ``False`` when the source does not exist. Other errors propagate
    after the temporary file is removed.
    """
    temp_path = unique_temp_path(destination_path)
    try:
        with factory.open_url(source_url, "rb") as source:
            data = source.read()
        with open(temp_path, "wb") as destination:
            destination.write(data)
        os.replace(temp_path, destination_path)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_path)
        raise
