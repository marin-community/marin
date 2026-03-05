# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic distributed locking with GCS generation-based atomicity.

Provides a ``DistributedLock`` that works on both GCS and local filesystems.
On GCS, uses generation-based conditional writes for atomicity.
On local filesystems, uses ``fcntl`` file locking.

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

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 30  # seconds between lease refreshes
HEARTBEAT_TIMEOUT = 90  # seconds before considering a lease stale


@dataclass
class Lease:
    """A lease held by a lock holder."""

    holder_id: str
    timestamp: float

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > HEARTBEAT_TIMEOUT


def default_holder_id() -> str:
    """Return a unique holder ID for the current host and thread."""
    return f"{os.uname()[1]}-{threading.get_ident()}"


class DistributedLock:
    """Lease-based distributed lock backed by a single lock file.

    On GCS (paths starting with ``gs://``), uses generation-based conditional
    writes for atomicity.  On local filesystems, uses ``fcntl`` for mutual
    exclusion.

    Args:
        lock_path: Path to the lock file (``gs://...`` or local path).
        holder_id: Unique identifier for this lock holder.
    """

    def __init__(self, lock_path: str, holder_id: str | None = None):
        self.lock_path = lock_path
        self.holder_id = holder_id or default_holder_id()

    @property
    def _is_gcs(self) -> bool:
        return self.lock_path.startswith("gs://")

    def _parse_gcs_path(self, path: str) -> tuple[str, str]:
        """Parse gs://bucket/path into (bucket, blob_path)."""
        path = path[5:]  # Remove gs:// prefix
        bucket, _, blob_path = path.partition("/")
        return (bucket, blob_path)

    def _read_lock_with_generation(self) -> tuple[int, Lease | None]:
        """Read lock file and its generation. Returns (0, None) if doesn't exist."""
        if self._is_gcs:
            from google.cloud import storage

            client = storage.Client()
            bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
            bucket = client.bucket(bucket_name)
            blob = bucket.get_blob(blob_path)
            if blob is None:
                return (0, None)
            data = json.loads(blob.download_as_string())
            return (blob.generation, Lease(**data))
        else:
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

    def _write_lock(self, lease: Lease, if_generation_match: int) -> None:
        """Write lock file with generation precondition.

        On GCS, uses generation-based conditional writes.
        On local, uses fcntl for mutual exclusion.
        """
        data = json.dumps(asdict(lease))

        if self._is_gcs:
            from google.cloud import storage

            client = storage.Client()
            bucket_name, blob_path = self._parse_gcs_path(self.lock_path)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(data, if_generation_match=if_generation_match)
        else:
            import fcntl

            parent = os.path.dirname(self.lock_path)
            os.makedirs(parent, exist_ok=True)

            with open(self.lock_path, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                content = f.read()
                if content:
                    current = Lease(**json.loads(content))
                    if not current.is_stale() and current.holder_id != lease.holder_id:
                        raise FileExistsError(f"Lock held by {current.holder_id}")
                f.seek(0)
                f.truncate()
                f.write(data)

    def try_acquire(self) -> bool:
        """Try to acquire the lock. Returns True if acquired."""
        generation, lock_data = self._read_lock_with_generation()

        if lock_data and not lock_data.is_stale():
            if lock_data.holder_id == self.holder_id:
                logger.debug("[%s] Already hold lock at %s", self.holder_id, self.lock_path)
                return True
            logger.debug("[%s] Lock %s held by %s (fresh)", self.holder_id, self.lock_path, lock_data.holder_id)
            return False

        if lock_data:
            logger.debug("[%s] Found stale lock at %s from %s", self.holder_id, self.lock_path, lock_data.holder_id)

        lease = Lease(holder_id=self.holder_id, timestamp=time.time())
        try:
            self._write_lock(lease, if_generation_match=generation)
        except FileExistsError:
            logger.debug("[%s] Lost lock race for %s", self.holder_id, self.lock_path)
            return False
        except Exception as e:
            if self._is_gcs and "PreconditionFailed" in type(e).__name__:
                logger.debug("[%s] Lost lock race for %s", self.holder_id, self.lock_path)
                return False
            raise

        return True

    def refresh(self) -> None:
        """Refresh a lock held by the current holder."""
        generation, lock_data = self._read_lock_with_generation()
        if lock_data and lock_data.holder_id == self.holder_id:
            self._write_lock(Lease(self.holder_id, time.time()), generation)
        else:
            current_holder = lock_data.holder_id if lock_data else "unknown"
            raise ValueError(
                f"Cannot refresh: lock at {self.lock_path} held by {current_holder}, expected {self.holder_id}"
            )

    def release(self) -> None:
        """Release the lock if held by this holder."""
        try:
            _, lock_data = self._read_lock_with_generation()
            if lock_data and lock_data.holder_id == self.holder_id:
                if self._is_gcs:
                    from iris.marin_fs import url_to_fs

                    fs = url_to_fs(self.lock_path, use_listings_cache=False)[0]
                    fs.rm(self.lock_path.replace("gs://", "", 1) if hasattr(fs, "_fs") else self.lock_path)
                else:
                    os.remove(self.lock_path)
                logger.debug("[%s] Released lock %s", self.holder_id, self.lock_path)
        except FileNotFoundError:
            pass

    def has_active_holder(self) -> bool:
        """Check if any holder has an active (non-stale) lock."""
        _, lock_data = self._read_lock_with_generation()
        return lock_data is not None and not lock_data.is_stale()
