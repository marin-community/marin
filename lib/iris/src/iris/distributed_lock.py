# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic distributed locking with lease-based semantics.

Provides a ``DistributedLock`` that works on GCS, local filesystems, and any
fsspec-compatible filesystem as a best-effort fallback.

- **GCS**: generation-based conditional writes for atomicity.
- **Local**: ``fcntl`` file locking for mutual exclusion.
- **Other** (any fsspec filesystem): best-effort write-then-read-back.
  Not fully atomic, but sufficient for advisory locking where races are
  unlikely and the worst case is duplicate work.

The lock is lease-based: holders must periodically refresh the lease,
and stale leases (older than ``HEARTBEAT_TIMEOUT``) can be taken over
by other holders.
"""

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


@dataclass
class Lease:
    """A lease held by a lock holder."""

    worker_id: str
    timestamp: float

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > HEARTBEAT_TIMEOUT


def _lease_from_data(data: dict) -> Lease:
    """Parse lease JSON with support for legacy lock files.

    Supports both ``worker_id`` (current) and ``holder_id`` (legacy).
    """
    worker = data.get("worker_id")
    if worker is None:
        worker = data.get("holder_id")
    if worker is None:
        raise KeyError("Lock data missing holder_id/worker_id")

    timestamp = data.get("timestamp")
    if timestamp is None:
        raise KeyError("Lock data missing timestamp")

    return Lease(worker_id=str(worker), timestamp=float(timestamp))


def default_worker_id() -> str:
    """Return a unique holder ID for the current host and thread."""
    return f"{os.uname()[1]}-{threading.get_ident()}"


def _is_local_path(path: str) -> bool:
    return not path.startswith("gs://") and "://" not in path


def _is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


class DistributedLock:
    """Lease-based distributed lock backed by a single lock file.

    On GCS (paths starting with ``gs://``), uses generation-based conditional
    writes for atomicity.  On local filesystems, uses ``fcntl`` for mutual
    exclusion.  On any other fsspec filesystem, uses a best-effort
    write-then-read-back approach.

    Args:
        lock_path: Path to the lock file (``gs://...``, local, or any fsspec URL).
        worker_id: Unique identifier for this lock holder.
    """

    def __init__(self, lock_path: str, worker_id: str | None = None):
        self.lock_path = lock_path
        self.worker_id = worker_id or default_worker_id()

    def _read_lock_with_generation(self) -> tuple[int, Lease | None]:
        """Read lock file and its generation. Returns (0, None) if doesn't exist."""
        if _is_gcs_path(self.lock_path):
            return self._read_gcs()
        elif _is_local_path(self.lock_path):
            return self._read_local()
        else:
            return self._read_fsspec()

    def _write_lock(self, lease: Lease, if_generation_match: int) -> None:
        """Write lock file with generation/concurrency precondition."""
        if _is_gcs_path(self.lock_path):
            self._write_gcs(lease, if_generation_match)
        elif _is_local_path(self.lock_path):
            self._write_local(lease)
        else:
            self._write_fsspec(lease)

    # -- GCS backend ----------------------------------------------------------

    @staticmethod
    def _parse_gcs_path(path: str) -> tuple[str, str]:
        """Parse gs://bucket/path into (bucket, blob_path)."""
        path = path[5:]  # Remove gs://
        bucket, _, blob_path = path.partition("/")
        return (bucket, blob_path)

    def _read_gcs(self) -> tuple[int, Lease | None]:
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.get_blob(blob_path)
        if blob is None:
            return (0, None)
        data = json.loads(blob.download_as_string())
        return (blob.generation, _lease_from_data(data))

    def _write_gcs(self, lease: Lease, if_generation_match: int) -> None:
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(json.dumps(asdict(lease)), if_generation_match=if_generation_match)

    def _delete_gcs(self) -> None:
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.delete()

    # -- Local backend --------------------------------------------------------

    def _read_local(self) -> tuple[int, Lease | None]:
        import fcntl

        try:
            with open(self.lock_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                content = f.read()
                if not content:
                    return (0, None)
                data = json.loads(content)
            return (1, _lease_from_data(data))
        except FileNotFoundError:
            return (0, None)

    def _write_local(self, lease: Lease) -> None:
        import fcntl

        parent = os.path.dirname(self.lock_path)
        os.makedirs(parent, exist_ok=True)

        with open(self.lock_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            content = f.read()
            if content:
                current = _lease_from_data(json.loads(content))
                if not current.is_stale() and current.worker_id != lease.worker_id:
                    raise FileExistsError(f"Lock held by {current.worker_id}")
            f.seek(0)
            f.truncate()
            f.write(json.dumps(asdict(lease)))

    # -- fsspec best-effort backend -------------------------------------------

    def _get_fs(self) -> tuple[fsspec.AbstractFileSystem, str]:
        """Return (fs, path) for the lock path via fsspec."""
        return fsspec.core.url_to_fs(self.lock_path)

    def _read_fsspec(self) -> tuple[int, Lease | None]:
        fs, path = self._get_fs()
        try:
            with fs.open(path, "r") as f:
                content = f.read()
            if not content:
                return (0, None)
            data = json.loads(content)
            return (1, _lease_from_data(data))
        except FileNotFoundError:
            return (0, None)

    def _write_fsspec(self, lease: Lease) -> None:
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

    # -- public API -----------------------------------------------------------

    def try_acquire(self) -> bool:
        """Try to acquire the lock. Returns True if acquired."""
        generation, lock_data = self._read_lock_with_generation()

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
            self._write_lock(lease, if_generation_match=generation)
        except FileExistsError:
            logger.debug("[%s] Lost lock race for %s", self.worker_id, self.lock_path)
            return False
        except Exception as e:
            if _is_gcs_path(self.lock_path) and "PreconditionFailed" in type(e).__name__:
                logger.debug("[%s] Lost lock race for %s", self.worker_id, self.lock_path)
                return False
            raise

        return True

    def refresh(self) -> None:
        """Refresh a lock held by the current holder."""
        generation, lock_data = self._read_lock_with_generation()
        if lock_data and lock_data.worker_id == self.worker_id:
            self._write_lock(Lease(self.worker_id, time.time()), generation)
        else:
            current_holder = lock_data.worker_id if lock_data else "unknown"
            raise ValueError(
                f"Cannot refresh: lock at {self.lock_path} held by {current_holder}, expected {self.worker_id}"
            )

    def release(self) -> None:
        """Release the lock if held by this holder."""
        try:
            _, lock_data = self._read_lock_with_generation()
            if lock_data and lock_data.worker_id == self.worker_id:
                if _is_gcs_path(self.lock_path):
                    self._delete_gcs()
                elif _is_local_path(self.lock_path):
                    os.remove(self.lock_path)
                else:
                    fs, path = self._get_fs()
                    fs.rm(path)
                logger.debug("[%s] Released lock %s", self.worker_id, self.lock_path)
        except FileNotFoundError:
            pass

    def has_active_holder(self) -> bool:
        """Check if any holder has an active (non-stale) lock."""
        _, lock_data = self._read_lock_with_generation()
        return lock_data is not None and not lock_data.is_stale()
