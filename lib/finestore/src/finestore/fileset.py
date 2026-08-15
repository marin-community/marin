# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronize a consumer-owned local directory through FineStore file-set commits."""

from __future__ import annotations

import atexit
import logging
import pathlib
import threading

from rigging.filesystem import atomic_rename

from finestore.layout import BLOB_DATA_COLUMN, BLOB_NAME_COLUMN, BLOBS_TABLE
from finestore.reader import ReadView
from finestore.store import DataStore

logger = logging.getLogger(__name__)

DEFAULT_SYNC_INTERVAL = 120.0
DEFAULT_TRANSACTION_BYTES = 100 * 1024 * 1024
DEFAULT_TRANSACTION_FILES = 65_536


def _safe_relative(name: str) -> pathlib.PurePosixPath:
    relative = pathlib.PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe file-set path: {name!r}")
    return relative


def fetch_file_set(root: str, local: str) -> set[str]:
    """Materialize one pinned file set and return the relative paths fetched."""
    destination = pathlib.Path(local)
    destination.mkdir(parents=True, exist_ok=True)
    fetched = set()
    for row in ReadView(root).iter_rows(BLOBS_TABLE, columns=[BLOB_NAME_COLUMN, BLOB_DATA_COLUMN]):
        relative = _safe_relative(row[BLOB_NAME_COLUMN])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_rename(str(target)) as staged:
            pathlib.Path(staged).write_bytes(bytes(row[BLOB_DATA_COLUMN]))
        fetched.add(relative.as_posix())
    return fetched


class FineStoreDirectory:
    """Mirror new files from a local directory as atomic FineStore file-set commits."""

    def __init__(
        self,
        root: str,
        local: str,
        *,
        flush_interval: float = DEFAULT_SYNC_INTERVAL,
        max_transaction_bytes: int = DEFAULT_TRANSACTION_BYTES,
        max_transaction_files: int = DEFAULT_TRANSACTION_FILES,
    ) -> None:
        self._root = root
        self._local = pathlib.Path(local)
        self._local.mkdir(parents=True, exist_ok=True)
        self._known = fetch_file_set(self._root, local)
        self._store = DataStore.open(self._root)
        self._flush_interval = flush_interval
        self._max_transaction_bytes = max_transaction_bytes
        self._max_transaction_files = max_transaction_files
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="finestore-directory-sync", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def flush(self) -> None:
        """Publish files added since the last successful sync in bounded transactions."""
        with self._flush_lock:
            present = {path.relative_to(self._local).as_posix() for path in self._local.rglob("*") if path.is_file()}
            pending = sorted(present - self._known)
            if not pending:
                return
            batch = []
            batch_bytes = 0
            for relative in pending:
                data = (self._local / relative).read_bytes()
                if batch and (
                    batch_bytes + len(data) > self._max_transaction_bytes or len(batch) >= self._max_transaction_files
                ):
                    self._commit(batch)
                    self._known.update(name for name, _data in batch)
                    batch = []
                    batch_bytes = 0
                batch.append((relative, data))
                batch_bytes += len(data)
            if batch:
                self._commit(batch)
                self._known.update(name for name, _data in batch)

    def close(self) -> None:
        """Stop synchronization and publish one final file-set commit."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join()
        self.flush()
        self._store.close()

    def _run(self) -> None:
        while not self._stop.wait(self._flush_interval):
            try:
                self.flush()
            except OSError as exc:
                logger.warning("FineStore directory synchronization failed: %s", exc)

    def _commit(self, files: list[tuple[str, bytes]]) -> None:
        with self._store.transaction(max_bytes=self._max_transaction_bytes) as transaction:
            for relative, data in files:
                transaction.write_object(relative, data)
