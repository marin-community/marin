# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic distributed locking with lease-based semantics.

Provides lease-based distributed locks backed by a single lock file.
Three backend implementations are available:

- **GcsLease**: generation-based conditional writes for atomicity.
- **LocalFileLease**: ``fcntl`` file locking for mutual exclusion.
- **FsspecLease**: best-effort write-then-read-back (advisory only).

Use ``create_lock()`` to obtain the appropriate implementation for a given path.

The lock is lease-based: holders must periodically refresh the lease,
and stale leases (older than ``HEARTBEAT_TIMEOUT``) can be taken over
by other holders.
"""

import abc
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass

import fsspec

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
                logger.debug("[%s] Released lock %s", self.worker_id, self.lock_path)
        except FileNotFoundError:
            pass

    def has_active_holder(self) -> bool:
        """Check if any holder has an active (non-stale) lock."""
        try:
            _, lock_data = self._read_with_generation()
        except FileNotFoundError:
            return False
        return lock_data is not None and not lock_data.is_stale()


# ---------------------------------------------------------------------------
# GCS backend
# ---------------------------------------------------------------------------


class GcsLease(DistributedLease):
    """GCS-backed lease using generation-based conditional writes."""

    @staticmethod
    def _parse_gcs_path(path: str) -> tuple[str, str]:
        """Parse ``gs://bucket/path`` into ``(bucket, blob_path)``."""
        path = path[5:]  # Remove gs://
        bucket, _, blob_path = path.partition("/")
        return (bucket, blob_path)

    def _read_with_generation(self) -> tuple[int, Lease | None]:
        from google.api_core.exceptions import NotFound
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.get_blob(blob_path)
        if blob is None:
            return (0, None)
        try:
            data = json.loads(blob.download_as_string())
        except NotFound:
            # Blob was deleted between get_blob and download_as_string
            logger.debug("[%s] Lock blob %s disappeared during read (race)", self.worker_id, self.lock_path)
            return (0, None)
        return (blob.generation, Lease(**data))

    def _write(self, lease: Lease, if_generation_match: int) -> None:
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(json.dumps(asdict(lease)), if_generation_match=if_generation_match)

    def _delete(self) -> None:
        from google.api_core.exceptions import NotFound
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        try:
            blob.delete()
        except NotFound:
            logger.debug("Lock blob %s already deleted", self.lock_path)


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


class LocalFileLease(DistributedLease):
    """Local-filesystem lease using ``fcntl`` file locking."""

    def _read_with_generation(self) -> tuple[int, Lease | None]:
        import fcntl

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
        import fcntl

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

    def _get_fs(self) -> tuple[fsspec.AbstractFileSystem, str]:
        """Return ``(fs, path)`` for the lock path via fsspec."""
        return fsspec.core.url_to_fs(self.lock_path)

    def _read_with_generation(self) -> tuple[int, Lease | None]:
        fs, path = self._get_fs()
        try:
            with fs.open(path, "r") as f:
                content = f.read()
            if not content:
                return (0, None)
            data = json.loads(content)
            return (1, Lease(**data))
        except FileNotFoundError:
            return (0, None)

    def _write(self, lease: Lease, if_generation_match: int) -> None:
        """Best-effort lock: write lease, then read back to check if we won."""
        fs, path = self._get_fs()
        data = json.dumps(asdict(lease))
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with fs.open(path, "w") as f:
            f.write(data)
        # Read back and check if our write stuck (best-effort race detection)
        time.sleep(0.1)
        try:
            with fs.open(path, "r") as f:
                readback = json.loads(f.read())
            if readback.get("worker_id") != lease.worker_id:
                raise FileExistsError(f"Lock race lost to {readback.get('worker_id')}")
        except FileNotFoundError as err:
            raise FileExistsError("Lock file disappeared after write") from err

    def _delete(self) -> None:
        fs, path = self._get_fs()
        try:
            fs.rm(path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_lock(lock_path: str, worker_id: str | None = None) -> DistributedLease:
    """Create the appropriate lease implementation for *lock_path*."""
    if _is_gcs_path(lock_path):
        return GcsLease(lock_path, worker_id)
    elif _is_local_path(lock_path):
        return LocalFileLease(lock_path, worker_id)
    else:
        return FsspecLease(lock_path, worker_id)


# Backward-compatible alias — call sites should migrate to create_lock().
DistributedLock = create_lock
