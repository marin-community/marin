# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic local downloads and object-store write-and-rename operations."""

import contextlib
import os
import uuid
from collections.abc import Generator
from typing import Protocol, cast

from s3fs import S3FileSystem

import rigging.filesystem.factory as factory
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.timing import ExponentialBackoff, retry_with_backoff


class _AtomicFileSystem(Protocol):
    def mv(self, source: str | list[str], destination: str | list[str], *, recursive: bool) -> object: ...

    def rm(self, path: str) -> object: ...


def unique_temp_path(output_path: str) -> str:
    return f"{output_path}.tmp.{uuid.uuid4().hex}"


def _move(filesystem: _AtomicFileSystem, source: str, destination: str) -> None:
    if not isinstance(filesystem, S3FileSystem) or not filesystem.isdir(source):
        filesystem.mv(source, destination, recursive=True)
        return

    # S3 prefixes are synthetic directories, not objects that CopyObject can
    # move. Explicit file pairs also keep fsspec's strict copy-error handling.
    prefix = filesystem._strip_protocol(source).rstrip("/") + "/"
    sources = cast(list[str], filesystem.find(source, withdirs=False))
    destinations = [f"{destination.rstrip('/')}/{path[len(prefix):]}" for path in sources]
    filesystem.mv(sources, destinations, recursive=False)


def _mv_with_retry(filesystem: _AtomicFileSystem, source: str, destination: str) -> None:
    retry_with_backoff(
        lambda: _move(filesystem, source, destination),
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
