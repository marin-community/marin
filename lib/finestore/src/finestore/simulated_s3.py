# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An in-memory object store that behaves like S3 for FineStore tests and benchmarks.

Every request pays one round trip of simulated latency, ranged reads fetch one block
per uncached request, renames cost a copy plus a delete, and the commit pointer uses
compare-and-swap with the 409 conflict S3 returns when conditional writes overlap.
Request counts are recorded per operation so a test or benchmark can assert how many
round trips an operation costs, not only whether it succeeded.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from unittest import mock

import rigging.filesystem.factory as factory
from fsspec.core import OpenFile
from fsspec.implementations.memory import MemoryFile, MemoryFileSystem
from rigging.filesystem.conditional_object import ConditionalWriteError, VersionedBytes

import finestore.commit

DEFAULT_LATENCY = 0.1
# Captured before ``simulated_s3`` patches ``time.sleep`` to scale FineStore's retry backoff.
_REAL_SLEEP = time.sleep
DEFAULT_BLOCK_SIZE = 5 * 1024 * 1024
MULTIPART_THRESHOLD = 5 * 1024 * 1024


class Request(StrEnum):
    """One S3 request type; multipart uploads count each part as a PUT."""

    GET = "GET"
    PUT = "PUT"
    HEAD = "HEAD"
    LIST = "LIST"
    COPY = "COPY"
    DELETE = "DELETE"
    CAS_PUT = "CAS_PUT"


@dataclass
class RequestLog:
    """Thread-safe counts of simulated requests, bytes, and compare-and-swap outcomes."""

    requests: Counter[Request] = field(default_factory=Counter)
    bytes_read: int = 0
    bytes_written: int = 0
    cas_conflicts: int = 0
    cas_successes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, request: Request, *, nbytes: int = 0) -> None:
        with self._lock:
            self.requests[request] += 1
            if request in (Request.GET, Request.COPY):
                self.bytes_read += nbytes
            if request in (Request.PUT, Request.CAS_PUT):
                self.bytes_written += nbytes

    def total(self) -> int:
        with self._lock:
            return sum(self.requests.values())

    def snapshot(self) -> RequestLog:
        with self._lock:
            return RequestLog(
                requests=Counter(self.requests),
                bytes_read=self.bytes_read,
                bytes_written=self.bytes_written,
                cas_conflicts=self.cas_conflicts,
                cas_successes=self.cas_successes,
            )

    def delta(self, earlier: RequestLog) -> RequestLog:
        """Return the requests recorded since ``earlier``."""
        current = self.snapshot()
        return RequestLog(
            requests=current.requests - earlier.requests,
            bytes_read=current.bytes_read - earlier.bytes_read,
            bytes_written=current.bytes_written - earlier.bytes_written,
            cas_conflicts=current.cas_conflicts - earlier.cas_conflicts,
            cas_successes=current.cas_successes - earlier.cas_successes,
        )


class _BlockReader(io.RawIOBase):
    """A read-only file that fetches one ``block_size`` range per uncached read."""

    def __init__(self, fs: SimulatedS3, path: str, data: bytes, block_size: int) -> None:
        super().__init__()
        self._fs = fs
        self._path = path
        self._data = data
        self._block_size = block_size
        self._pos = 0
        self._cached: tuple[int, int] = (0, 0)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    @property
    def size(self) -> int:
        return len(self._data)

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = len(self._data) + offset
        return self._pos

    def read(self, size: int = -1) -> bytes:
        end = len(self._data) if size is None or size < 0 else min(len(self._data), self._pos + size)
        start = self._pos
        if start >= len(self._data):
            return b""
        cached_start, cached_end = self._cached
        if start < cached_start or end > cached_end:
            fetch_end = max(end, min(len(self._data), start + self._block_size))
            self._fs.charge(Request.GET, nbytes=fetch_end - start)
            self._cached = (start, fetch_end)
        self._pos = end
        return self._data[start:end]

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class _UploadFile(MemoryFile):
    """A buffered write that pays its PUT (or multipart PUTs) when closed."""

    def __init__(self, fs: SimulatedS3, path: str) -> None:
        super().__init__(fs, path)
        self._uploaded = False

    def close(self) -> None:
        if self._uploaded:
            return
        self._uploaded = True
        data = self.getvalue()
        parts = max(1, -(-len(data) // MULTIPART_THRESHOLD))
        if parts > 1:
            # Initiate, one PUT per part, then complete.
            self.fs.charge(Request.PUT, nbytes=0)
            for _ in range(parts):
                self.fs.charge(Request.PUT, nbytes=MULTIPART_THRESHOLD)
            self.fs.charge(Request.PUT, nbytes=0)
        else:
            self.fs.charge(Request.PUT, nbytes=len(data))
        self.fs.put_object(self.path, data)
        super().close()


class SimulatedS3(MemoryFileSystem):
    """A private in-memory bucket set that charges S3-like latency per request."""

    cachable = False

    def __init__(self, *, latency: float = DEFAULT_LATENCY, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        super().__init__()
        # fsspec shares one store across every MemoryFileSystem; each simulated bucket set is private.
        self.store: dict[str, Any] = {}  # pyrefly: ignore[bad-override]
        self.pseudo_dirs: list[str] = [""]
        self.latency = latency
        self.block_size = block_size
        self.log = RequestLog()
        self._store_lock = threading.RLock()
        self._in_flight_cas: dict[str, set[int]] = {}

    def charge(self, request: Request, *, nbytes: int = 0) -> None:
        self.log.record(request, nbytes=nbytes)
        if self.latency:
            _REAL_SLEEP(self.latency)

    def put_object(self, path: str, data: bytes) -> None:
        with self._store_lock:
            self.store[self._strip_protocol(path)] = MemoryFile(self, self._strip_protocol(path), data)

    def ls(self, path, detail=True, **kwargs):
        self.charge(Request.LIST)
        with self._store_lock:
            return super().ls(path, detail=detail, **kwargs)

    def info(self, path, **kwargs):
        self.charge(Request.HEAD)
        with self._store_lock:
            return super().info(path, **kwargs)

    def isfile(self, path):
        self.charge(Request.HEAD)
        with self._store_lock:
            return super().isfile(path)

    def makedirs(self, path, exist_ok=False):
        return None

    def mkdir(self, path, create_parents=True, **kwargs):
        return None

    def _open(self, path, mode="rb", block_size=None, autocommit=True, cache_options=None, **kwargs):
        path = self._strip_protocol(path)
        if mode in ("rb", "r"):
            with self._store_lock:
                if path not in self.store:
                    raise FileNotFoundError(path)
                data = self.store[path].getvalue()
            cache_type = kwargs.get("cache_type")
            if cache_type == "none":
                fetch_block = 1
            else:
                fetch_block = block_size or self.block_size
            return io.BufferedReader(_BlockReader(self, path, data, fetch_block))
        if mode in ("wb", "w", "xb", "ab"):
            return _UploadFile(self, path)
        raise ValueError(f"unsupported mode {mode!r}")

    def cp_file(self, path1, path2, **kwargs):
        with self._store_lock:
            size = self.store[self._strip_protocol(path1)].size
        self.charge(Request.COPY, nbytes=size)
        with self._store_lock:
            super().cp_file(path1, path2, **kwargs)

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        self.cp_file(path1, path2)
        self.rm(path1)

    def _rm(self, path):
        self.charge(Request.DELETE)
        with self._store_lock:
            super()._rm(path)

    def cat_file(self, path, start=None, end=None, **kwargs):
        with self._store_lock:
            data = super().cat_file(path, start=start, end=end, **kwargs)
        self.charge(Request.GET, nbytes=len(data))
        return data

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        self.charge(Request.PUT, nbytes=len(value))
        with self._store_lock:
            super().pipe_file(path, value, mode=mode, **kwargs)

    def modified(self, path):
        self.charge(Request.HEAD)
        with self._store_lock:
            return super().modified(path)

    def created(self, path):
        self.charge(Request.HEAD)
        with self._store_lock:
            return super().created(path)

    # Conditional-object support -------------------------------------------------

    def _etag(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def conditional_read(self, path: str) -> VersionedBytes | None:
        self.charge(Request.GET)
        with self._store_lock:
            entry = self.store.get(self._strip_protocol(path))
            if entry is None:
                return None
            data = entry.getvalue()
        return VersionedBytes(data=data, version=self._etag(data))

    def conditional_write(self, path: str, data: bytes, *, expected_version: str | None) -> str:
        """Compare-and-swap one object, failing like S3 when conditional writes overlap.

        S3 rejects a conditional PUT with ``409 ConditionalRequestConflict`` while another
        conditional PUT on the same key is in flight; both writers then re-read and retry.
        """
        key = self._strip_protocol(path)
        with self._store_lock:
            in_flight = self._in_flight_cas.setdefault(key, set())
            contended = bool(in_flight)
            in_flight.add(threading.get_ident())
        try:
            self.charge(Request.CAS_PUT, nbytes=len(data))
            with self._store_lock:
                entry = self.store.get(key)
                current = None if entry is None else self._etag(entry.getvalue())
                if contended or current != expected_version:
                    self.log.cas_conflicts += 1
                    reason = "ConditionalRequestConflict" if contended else "PreconditionFailed"
                    raise ConditionalWriteError(f"conditional write failed for {path}: {reason}")
                self.store[key] = MemoryFile(self, key, data)
                self.log.cas_successes += 1
                return self._etag(data)
        finally:
            with self._store_lock:
                in_flight.discard(threading.get_ident())


@dataclass(frozen=True)
class SimulatedConditionalObject:
    """The compare-and-swap commit pointer over a :class:`SimulatedS3` key."""

    path: str
    fs: SimulatedS3

    def version(self) -> str | None:
        current = self.fs.conditional_read(self.path)
        return None if current is None else current.version

    def read(self) -> VersionedBytes | None:
        return self.fs.conditional_read(self.path)

    def write(self, data: bytes, *, expected_version: str | None) -> str:
        return self.fs.conditional_write(self.path, data, expected_version=expected_version)


@contextlib.contextmanager
def simulated_s3(
    *, latency: float = DEFAULT_LATENCY, block_size: int = DEFAULT_BLOCK_SIZE, time_scale: float = 1.0
) -> Iterator[SimulatedS3]:
    """Route every ``s3://`` FineStore access in this process through one simulated bucket set.

    ``time_scale`` multiplies both the simulated latency and the retry sleeps FineStore
    performs, so a benchmark can run compressed without changing the ratio between
    round trips and backoff. Report elapsed times divided by ``time_scale``.
    """
    fs = SimulatedS3(latency=latency * time_scale, block_size=block_size)

    def scaled_sleep(seconds: float) -> None:
        _REAL_SLEEP(seconds * time_scale)

    def url_to_fs(url: str, **kwargs):
        if not url.startswith("s3://"):
            raise ValueError(f"simulated S3 received a non-s3 URL: {url!r}")
        return fs, url

    def open_url(url: str, mode: str = "rb", **kwargs):
        if not url.startswith("s3://"):
            raise ValueError(f"simulated S3 received a non-s3 URL: {url!r}")
        return OpenFile(fs, url, mode, **{k: v for k, v in kwargs.items() if k in ("compression", "encoding")})

    with (
        mock.patch.object(factory, "url_to_fs", url_to_fs),
        mock.patch.object(factory, "open_url", open_url),
        mock.patch.object(finestore.commit, "conditional_object", lambda path: SimulatedConditionalObject(path, fs)),
        mock.patch("rigging.timing.time.sleep", scaled_sleep),
    ):
        yield fs
