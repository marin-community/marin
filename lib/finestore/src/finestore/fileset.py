# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronize a consumer-owned local directory through FineStore file-set commits."""

from __future__ import annotations

import atexit
import logging
import pathlib
import stat
import threading
import time
from dataclasses import dataclass

from rigging.filesystem.atomic import atomic_rename
from rigging.timing import Deadline

from finestore.layout import BlobTables
from finestore.reader import BlobDescriptor, ReadView
from finestore.store import DataStore

logger = logging.getLogger(__name__)

DEFAULT_SYNC_INTERVAL = 120.0
DEFAULT_BATCH_DATA_BYTES = 100 * 1024 * 1024
DEFAULT_BATCH_FILES = 65_536
_EXIT_FLUSH_TIMEOUT = 10.0
DEFAULT_FETCH_LOG_INTERVAL = 10.0


class FileSetFetchLimitError(RuntimeError):
    """A committed file set exceeded a size cap or lacks size metadata needed to enforce one."""


class FileSetFetchDeadlineError(TimeoutError):
    """A committed file-set fetch passed its startup deadline."""


class FileSetPublishLimitError(RuntimeError):
    """New files would make the remote set exceed its fetch budget."""


@dataclass(frozen=True)
class FileSetFetchLimits:
    """Cap logical files and bytes; check the time limit between file operations."""

    max_files: int
    max_bytes: int
    timeout: float
    log_interval: float = DEFAULT_FETCH_LOG_INTERVAL


def _safe_relative(name: str) -> pathlib.PurePosixPath:
    relative = pathlib.PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe file-set path: {name!r}")
    return relative


def _local_files(directory: pathlib.Path) -> set[str]:
    return {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}


def _check_deadline(deadline: Deadline | None, root: str) -> None:
    if deadline is not None and deadline.expired():
        raise FileSetFetchDeadlineError(f"file-set fetch from {root} exceeded its deadline")


def _local_matches_blob(
    target: pathlib.Path, descriptor: BlobDescriptor, view: ReadView, deadline: Deadline | None, root: str
) -> bool:
    try:
        local_stat = target.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(local_stat.st_mode) or (descriptor.size is not None and local_stat.st_size != descriptor.size):
        return False
    try:
        current = target.open("rb")
    except OSError:
        return False
    with current:
        for part in view.blob_parts(descriptor):
            _check_deadline(deadline, root)
            if current.read(len(part)) != part:
                return False
        return not current.read(1)


def _write_blob(
    target: pathlib.Path, descriptor: BlobDescriptor, view: ReadView, deadline: Deadline | None, root: str
) -> None:
    with atomic_rename(str(target)) as staged:
        with pathlib.Path(staged).open("wb") as output:
            for part in view.blob_parts(descriptor):
                _check_deadline(deadline, root)
                output.write(part)


def fetch_file_set(root: str, local: str, *, limits: FileSetFetchLimits | None = None) -> set[str]:
    """Materialize a pinned set and return all relative paths, including matching local files."""
    destination = pathlib.Path(local)
    destination.mkdir(parents=True, exist_ok=True)
    fetched = set()
    started = time.monotonic()
    deadline = Deadline.from_seconds(limits.timeout) if limits is not None else None
    next_log = started + (limits.log_interval if limits else DEFAULT_FETCH_LOG_INTERVAL)
    files_written = 0
    files_skipped = 0
    total_bytes = 0

    view = ReadView(root)
    for row in view.iter_rows(
        BlobTables.DESCRIPTORS,
        columns=BlobDescriptor.COLUMNS,
    ):
        _check_deadline(deadline, root)
        descriptor = BlobDescriptor.from_row(row)
        if limits is not None:
            if len(fetched) + 1 > limits.max_files:
                raise FileSetFetchLimitError(f"file-set fetch from {root} exceeds {limits.max_files} files")
            if descriptor.size is None:
                raise FileSetFetchLimitError(f"file-set fetch from {root} cannot bound size of {descriptor.name!r}")
            if total_bytes + descriptor.size > limits.max_bytes:
                raise FileSetFetchLimitError(f"file-set fetch from {root} exceeds {limits.max_bytes} bytes")
        relative = _safe_relative(descriptor.name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if _local_matches_blob(target, descriptor, view, deadline, root):
            files_skipped += 1
        else:
            _write_blob(target, descriptor, view, deadline, root)
            files_written += 1
        fetched.add(relative.as_posix())
        total_bytes += descriptor.size or 0
        now = time.monotonic()
        if now >= next_log:
            elapsed = now - started
            logger.info(
                "File-set fetch progress: %d files (%d written, %d skipped), %d bytes in %.1fs (%.0f files/s)",
                len(fetched),
                files_written,
                files_skipped,
                total_bytes,
                elapsed,
                len(fetched) / elapsed,
                extra={
                    "files_processed": len(fetched),
                    "files_written": files_written,
                    "files_skipped": files_skipped,
                    "bytes_processed": total_bytes,
                },
            )
            next_log = now + (limits.log_interval if limits else DEFAULT_FETCH_LOG_INTERVAL)
    _check_deadline(deadline, root)
    logger.info(
        "File-set fetch complete: %d files (%d written, %d skipped), %d bytes in %.1fs",
        len(fetched),
        files_written,
        files_skipped,
        total_bytes,
        time.monotonic() - started,
    )
    return fetched


class FineStoreDirectory:
    """Mirror new files from a local directory as atomic FineStore file-set commits.

    Periodic synchronization is best-effort. Call :meth:`close` when the final
    files must be published; interpreter shutdown gives the final sync a bounded
    drain window.
    """

    def __init__(
        self,
        root: str,
        local: str,
        *,
        flush_interval: float = DEFAULT_SYNC_INTERVAL,
        max_batch_data_bytes: int = DEFAULT_BATCH_DATA_BYTES,
        max_batch_files: int = DEFAULT_BATCH_FILES,
        fetch_limits: FileSetFetchLimits | None = None,
    ) -> None:
        self._root = root
        self._local = pathlib.Path(local)
        self._local.mkdir(parents=True, exist_ok=True)
        fetched = fetch_file_set(self._root, local, limits=fetch_limits)
        self._known = _local_files(self._local)
        self._fetch_limits = fetch_limits
        self._published_files = len(fetched)
        self._published_bytes = sum((self._local / name).stat().st_size for name in fetched)
        self._store = DataStore.open(self._root)
        self._flush_interval = flush_interval
        self._max_batch_data_bytes = max_batch_data_bytes
        self._max_batch_files = max_batch_files
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._flush_at_stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="finestore-directory-sync", daemon=True)
        self._thread.start()
        atexit.register(self._close_at_exit)

    def flush(self) -> None:
        """Publish files added since the last sync in target-bounded multi-file transactions."""
        with self._flush_lock:
            present = _local_files(self._local)
            pending = sorted(present - self._known)
            if not pending:
                return
            if self._fetch_limits is not None:
                if self._published_files + len(pending) > self._fetch_limits.max_files:
                    raise FileSetPublishLimitError(
                        f"file-set publish to {self._root} exceeds {self._fetch_limits.max_files} files"
                    )
                pending_bytes = sum((self._local / name).stat().st_size for name in pending)
                self._check_publish_bytes(pending_bytes)
            batch = []
            batch_bytes = 0
            for relative in pending:
                data = (self._local / relative).read_bytes()
                self._check_publish_bytes(batch_bytes + len(data))
                if batch and (
                    batch_bytes + len(data) > self._max_batch_data_bytes or len(batch) >= self._max_batch_files
                ):
                    self._commit(batch)
                    batch = []
                    batch_bytes = 0
                batch.append((relative, data))
                batch_bytes += len(data)
            if batch:
                self._commit(batch)

    def close(self) -> None:
        """Stop synchronization and publish one final file-set commit."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join()
        try:
            self.flush()
        finally:
            self._store.close()
            atexit.unregister(self._close_at_exit)

    def _close_at_exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._flush_at_stop.set()
        self._stop.set()
        self._thread.join(timeout=_EXIT_FLUSH_TIMEOUT)
        if self._thread.is_alive():
            logger.warning(
                "FineStore directory synchronization did not finish within %.1f seconds; "
                "abandoning it at process exit",
                _EXIT_FLUSH_TIMEOUT,
            )

    def _run(self) -> None:
        while not self._stop.wait(self._flush_interval):
            try:
                self.flush()
            except FileSetPublishLimitError as exc:
                logger.warning("FineStore directory synchronization stopped: %s", exc)
                return
            except OSError as exc:
                logger.warning("FineStore directory synchronization failed: %s", exc)
        if self._flush_at_stop.is_set():
            try:
                self.flush()
            except Exception as exc:
                logger.warning("FineStore directory final synchronization failed: %s", exc)

    def _commit(self, files: list[tuple[str, bytes]]) -> None:
        with self._store.unbounded_transaction() as transaction:
            for relative, data in files:
                transaction.write_object(relative, data)
        self._known.update(name for name, _data in files)
        self._published_files += len(files)
        self._published_bytes += sum(len(data) for _name, data in files)

    def _check_publish_bytes(self, additional_bytes: int) -> None:
        if self._fetch_limits is not None and self._published_bytes + additional_bytes > self._fetch_limits.max_bytes:
            raise FileSetPublishLimitError(
                f"file-set publish to {self._root} exceeds {self._fetch_limits.max_bytes} bytes"
            )
