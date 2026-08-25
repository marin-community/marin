# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic distributed locking with lease-based semantics.

Provides lease-based distributed locks backed by a single lock file.
Four backend implementations are available:

- **GcsLease**: generation-based conditional writes for atomicity.
- **S3Lease**: conditional writes (``If-None-Match`` / ``If-Match``) for S3-compatible stores.
- **LocalFileLease**: ``fcntl`` file locking for mutual exclusion.
- **FsspecLease**: best-effort write-then-read-back (advisory only).

Use ``create_lock()`` to obtain the appropriate implementation for a given path.

The lock is lease-based: holders must periodically refresh the lease,
and stale leases (older than ``HEARTBEAT_TIMEOUT``) can be taken over
by other holders.
"""

import abc
import fcntl
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass

from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 30  # seconds between lease refreshes
HEARTBEAT_TIMEOUT = 90  # seconds before considering a lease stale


class LeaseLostError(Exception):
    """The lease is held by another worker.

    This is a fatal condition: the step must terminate immediately.
    """


@dataclass
class Lease:
    """Persisted lease state: who holds it and when it was last refreshed."""

    worker_id: str
    timestamp: float

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > HEARTBEAT_TIMEOUT


def default_worker_id() -> str:
    """Return a unique holder ID for the current host and thread."""
    return f"{os.uname()[1]}-{threading.get_ident()}"


def _is_local_path(path: str) -> bool:
    return not path.startswith("gs://") and "://" not in path


def _is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DistributedLease(abc.ABC):
    """Base class for lease-based distributed locks.

    Subclasses implement storage operations (read/write/delete);
    the locking protocol (acquire, refresh, release) is defined here.

    Args:
        lock_path: Path to the lock file.
        worker_id: Unique identifier for this lock holder.
    """

    def __init__(self, lock_path: str, worker_id: str | None = None):
        self.lock_path = lock_path
        self.worker_id = worker_id or default_worker_id()

    # -- abstract storage ops ------------------------------------------------

    @abc.abstractmethod
    def _read_with_generation(self) -> tuple[int, Lease | None]:
        """Read lock file.  Returns ``(generation, lease)`` or ``(0, None)`` if absent."""
        ...

    @abc.abstractmethod
    def _write(self, lease: Lease, if_generation_match: int) -> None:
        """Write lock file with generation/concurrency precondition."""
        ...

    @abc.abstractmethod
    def _delete(self) -> None:
        """Delete lock file.  Must not raise if already absent."""
        ...

    # -- public API ----------------------------------------------------------

    def try_acquire(self) -> bool:
        """Try to acquire the lock.  Returns True if acquired."""
        generation, lock_data = self._read_with_generation()

        if lock_data and not lock_data.is_stale():
            if lock_data.worker_id == self.worker_id:
                logger.debug("[%s] Already hold lock at %s", self.worker_id, self.lock_path)
                return True
            logger.debug("[%s] Lock %s held by %s (fresh)", self.worker_id, self.lock_path, lock_data.worker_id)
            return False

        if lock_data:
            logger.debug("[%s] Found stale lock at %s from %s", self.worker_id, self.lock_path, lock_data.worker_id)

        lease = Lease(worker_id=self.worker_id, timestamp=time.time())
        try:
            self._write(lease, if_generation_match=generation)
        except FileExistsError:
            logger.debug("[%s] Lost lock race for %s", self.worker_id, self.lock_path)
            return False
        except Exception as e:
            if "PreconditionFailed" in type(e).__name__:
                logger.debug("[%s] Lost lock race for %s (precondition)", self.worker_id, self.lock_path)
                return False
            raise

        return True

    def refresh(self) -> None:
        """Refresh a lease held by the current holder.

        Raises ``LeaseLostError`` if the lock is held by a different worker
        **or** if the lock file has disappeared.  A missing lock file means
        another worker deleted it (e.g. took over a stale lease and released
        it), so the current holder has irrecoverably lost ownership.
        """
        generation, lock_data = self._read_with_generation()
        if lock_data and lock_data.worker_id == self.worker_id:
            self._write(Lease(self.worker_id, time.time()), generation)
        elif lock_data is None:
            raise LeaseLostError(f"Lease lost: lock file {self.lock_path} disappeared — another worker likely took over")
        else:
            raise LeaseLostError(
                f"Lease lost: lock at {self.lock_path} held by {lock_data.worker_id}, expected {self.worker_id}"
            )

    def release(self) -> None:
        """Release the lock if held by this holder.  Idempotent."""
        try:
            _, lock_data = self._read_with_generation()
            if lock_data and lock_data.worker_id == self.worker_id:
                self._delete()
                logger.info("Released lock path=%s worker=%s", self.lock_path, self.worker_id)
        except FileNotFoundError:
            pass

    def has_active_holder(self) -> bool:
        """Check if any holder has an active (non-stale) lock."""
        return self.active_holder_id() is not None

    def active_holder_id(self) -> str | None:
        """Return the active lock-owner ID, or None if no active lock exists."""
        try:
            _, lock_data = self._read_with_generation()
        except FileNotFoundError:
            return None
        if lock_data is None or lock_data.is_stale():
            return None
        return lock_data.worker_id


# ---------------------------------------------------------------------------
# GCS backend
# ---------------------------------------------------------------------------


class GcsLease(DistributedLease):
    """GCS-backed lease using generation-based conditional writes."""

    def _read_with_generation(self) -> tuple[int, Lease | None]:
        found = conditional_object(self.lock_path).read()
        if found is None:
            return (0, None)
        return (int(found.version), Lease(**json.loads(found.data)))

    def _write(self, lease: Lease, if_generation_match: int) -> None:
        try:
            conditional_object(self.lock_path).write(
                json.dumps(asdict(lease)).encode(),
                expected_version=None if if_generation_match == 0 else str(if_generation_match),
            )
        except ConditionalWriteError as exc:
            raise FileExistsError(f"conditional write failed for {self.lock_path}") from exc

    def _delete(self) -> None:
        try:
            StoragePath(self.lock_path).rm()
        except FileNotFoundError:
            logger.debug("Lock blob %s already deleted", self.lock_path)


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------


class S3Lease(DistributedLease):
    """S3-backed lease using the conditional-object adapter."""

    def __init__(self, lock_path: str, worker_id: str | None = None):
        super().__init__(lock_path, worker_id)
        self._last_etag: str | None = None

    def _read_with_generation(self) -> tuple[int, Lease | None]:
        found = conditional_object(self.lock_path).read()
        if found is None:
            self._last_etag = None
            return (0, None)
        self._last_etag = found.version
        return (1, Lease(**json.loads(found.data)))

    def _write(self, lease: Lease, if_generation_match: int) -> None:
        try:
            conditional_object(self.lock_path).write(
                json.dumps(asdict(lease)).encode(),
                expected_version=None if if_generation_match == 0 else self._last_etag,
            )
        except ConditionalWriteError as exc:
            raise FileExistsError(f"conditional write failed for {self.lock_path}") from exc

    def _delete(self) -> None:
        try:
            StoragePath(self.lock_path).rm()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


class LocalFileLease(DistributedLease):
    """Local-filesystem lease using ``fcntl`` file locking."""

    def _read_with_generation(self) -> tuple[int, Lease | None]:
        try:
            with open(self.lock_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                content = f.read()
                if not content:
                    return (0, None)
                data = json.loads(content)
            return (1, Lease(**data))
        except FileNotFoundError:
            return (0, None)

    def _write(self, lease: Lease, if_generation_match: int) -> None:
        parent = os.path.dirname(self.lock_path)
        os.makedirs(parent, exist_ok=True)

        with open(self.lock_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            content = f.read()
            if content:
                current = Lease(**json.loads(content))
                if not current.is_stale() and current.worker_id != lease.worker_id:
                    raise FileExistsError(f"Lock held by {current.worker_id}")
            f.seek(0)
            f.truncate()
            f.write(json.dumps(asdict(lease)))

    def _delete(self) -> None:
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# fsspec best-effort backend
# ---------------------------------------------------------------------------


class FsspecLease(DistributedLease):
    """Best-effort lease for arbitrary fsspec filesystems."""

    @property
    def _path(self) -> StoragePath:
        return StoragePath.parse(self.lock_path)

    def _read_with_generation(self) -> tuple[int, Lease | None]:
        try:
            content = self._path.read_text()
        except FileNotFoundError:
            return (0, None)
        if not content:
            return (0, None)
        return (1, Lease(**json.loads(content)))

    def _write(self, lease: Lease, if_generation_match: int) -> None:
        path = self._path
        path.parent.mkdirs(exist_ok=True)
        path.write_text(json.dumps(asdict(lease)))
        # No conditional write on a generic fsspec store, so detect a lost race by
        # reading our own write back after a short settle.
        time.sleep(0.1)
        try:
            readback = json.loads(self._path.read_text())
        except FileNotFoundError as err:
            raise FileExistsError("Lock file disappeared after write") from err
        if readback.get("worker_id") != lease.worker_id:
            raise FileExistsError(f"Lock race lost to {readback.get('worker_id')}")

    def _delete(self) -> None:
        try:
            self._path.rm()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_lock(lock_path: str, worker_id: str | None = None) -> DistributedLease:
    """Create the appropriate lease implementation for *lock_path*."""
    if _is_gcs_path(lock_path):
        return GcsLease(lock_path, worker_id)
    elif _is_s3_path(lock_path):
        return S3Lease(lock_path, worker_id)
    elif _is_local_path(lock_path):
        return LocalFileLease(lock_path, worker_id)
    else:
        return FsspecLease(lock_path, worker_id)
