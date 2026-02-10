# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Each `ExecutorStep` produces an `output_path`.
We associate each `output_path` with:
- A status file (`output_path/.executor_status`) containing simple text: SUCCESS, FAILURE, or RUNNING
- A LOCK file (`output_path/.executor_status.lock`) for distributed locking

The LOCK file contains JSON with {worker_id, timestamp} and is refreshed periodically.
On GCS, we use generation-based conditional writes for atomicity.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import fsspec
from google.cloud import storage

logger = logging.getLogger("ray")

HEARTBEAT_INTERVAL = 30  # seconds between lease refreshes
HEARTBEAT_TIMEOUT = 90  # seconds before considering a lease stale

STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_SUCCESS = "SUCCESS"
STATUS_DEP_FAILED = "DEP_FAILED"  # Dependency failed


def get_status_path(output_path: str) -> str:
    """Return the path of the status file associated with `output_path`."""
    return os.path.join(output_path, ".executor_status")


@dataclass
class Lease:
    """A lease held by a worker for a step."""

    worker_id: str
    timestamp: float

    def is_stale(self) -> bool:
        logger.debug(f"Is stale? {time.time()} {self.timestamp} {time.time() - self.timestamp}")
        return (time.time() - self.timestamp) > HEARTBEAT_TIMEOUT


class StatusFile:
    """Manages executor step status with distributed locking.

    Two types of files:
    - LOCK file (JSON): Single file for distributed lock acquisition.
      Contains {worker_id, timestamp}. Must be refreshed periodically.
    - Status file (simple text): Final state - SUCCESS, FAILURE, or RUNNING.

    Lock acquisition uses GCS generation-based conditional writes for atomicity.
    """

    def __init__(self, output_path: str, worker_id: str):
        self.output_path = output_path
        self.path = get_status_path(output_path)
        self.worker_id = worker_id
        self._lock_path = self.path + ".lock"
        self.fs = fsspec.core.url_to_fs(self.path, use_listings_cache=False)[0]

    @property
    def _is_gcs(self) -> bool:
        return self.path.startswith("gs://")

    def _parse_gcs_path(self, path: str) -> tuple[str, str]:
        """Parse gs://bucket/path into (bucket, blob_path)."""
        path = path[5:]  # Remove gs:// prefix
        bucket, _, blob_path = path.partition("/")
        return (bucket, blob_path)

    @property
    def status(self) -> str | None:
        """Read current status from status file.

        The modern representation stores a single status token, but older
        executors wrote JSON-line event logs. We support both until the legacy
        files are gone.
        """
        if not self.fs.exists(self.path):
            return None

        with self.fs.open(self.path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            return None

        # New format: a single status token such as SUCCESS/RUNNING/etc.
        if len(lines) == 1 and not lines[0].startswith("{"):
            return lines[0]

        # Legacy format: JSON event log, one object per line.
        legacy_status = self._parse_legacy_status(lines)
        if legacy_status:
            return legacy_status

        # Fall back to the last non-empty line if parsing fails.
        return lines[-1]

    @staticmethod
    def _parse_legacy_status(lines: list[str]) -> str | None:
        """Return the latest status from the deprecated JSON-lines format."""
        last_status: str | None = None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                status = event.get("status")
                if isinstance(status, str):
                    last_status = status
        return last_status

    def write_status(self, status: str) -> None:
        """Write status (SUCCESS/FAILURE/RUNNING).

        For terminal statuses (SUCCESS/FAILED), the lock is released.
        For RUNNING, the lock is maintained so heartbeat can continue refreshing it.
        """
        parent = os.path.dirname(self.path)
        if not self.fs.exists(parent):
            self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(self.path, "w") as f:
            f.write(status)

        if status != STATUS_RUNNING:
            self.release_lock()
        logger.debug("[%s] Wrote status %s to %s", self.worker_id, status, self.path)

    def _read_lock_with_generation(self) -> tuple[int, Lease | None]:
        """Read LOCK file and its generation. Returns (0, None) if doesn't exist."""
        if self._is_gcs:
            client = storage.Client()
            bucket_name, blob_path = self._parse_gcs_path(self._lock_path)
            bucket = client.bucket(bucket_name)
            blob = bucket.get_blob(blob_path)
            if blob is None:
                return (0, None)
            data = json.loads(blob.download_as_string())
            return (blob.generation, Lease(**data))
        else:
            import fcntl

            try:
                # Use flock to avoid reading partially written files
                with open(self._lock_path, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    content = f.read()
                    if not content:
                        return (0, None)
                    data = json.loads(content)
                return (1, Lease(**data))
            except FileNotFoundError:
                return (0, None)

    def _write_lock(self, lease: Lease, if_generation_match: int) -> None:
        """Write LOCK file with generation precondition.

        On GCS, uses generation-based conditional writes.
        On local, uses atomic rename then read-back to verify.
        """
        data = json.dumps(asdict(lease))

        if self._is_gcs:
            client = storage.Client()
            bucket_name, blob_path = self._parse_gcs_path(self._lock_path)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(data, if_generation_match=if_generation_match)
        else:
            import fcntl

            parent = os.path.dirname(self._lock_path)
            os.makedirs(parent, exist_ok=True)

            # Use flock on the lock file itself for mutual exclusion
            with open(self._lock_path, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                content = f.read()
                if content:
                    current = Lease(**json.loads(content))
                    if not current.is_stale() and current.worker_id != lease.worker_id:
                        raise FileExistsError(f"Lock held by {current.worker_id}")
                f.seek(0)
                f.truncate()
                f.write(data)

    def refresh_lock(self) -> None:
        """Refresh a lock held by the current worker."""
        generation, lock_data = self._read_lock_with_generation()
        if lock_data and lock_data.worker_id == self.worker_id:
            logger.debug("Refreshing lock for worker %s at generation %s", self.worker_id, generation)
            self._write_lock(Lease(self.worker_id, time.time()), generation)
        else:
            lock_worker = lock_data.worker_id if lock_data else "unknown"
            raise ValueError(
                f"Failed precondition: lock not held by current worker: found {lock_worker}, expected {self.worker_id}"
            )

    def try_acquire_lock(self) -> bool:
        """Try to acquire the lock using atomic LOCK file, or update the lock if held.

        On GCS, uses generation-based preconditions for atomicity.
        """
        generation, lock_data = self._read_lock_with_generation()

        if lock_data and not lock_data.is_stale():
            if lock_data.worker_id == self.worker_id:
                logger.info("[%s] Already hold lock", self.worker_id)
                return True
            logger.info("[%s] Lock held by %s (fresh)", self.worker_id, lock_data.worker_id)
            return False

        if lock_data:
            logger.info("[%s] Found stale lock from %s, attempting takeover", self.worker_id, lock_data.worker_id)

        lease = Lease(worker_id=self.worker_id, timestamp=time.time())
        try:
            self._write_lock(lease, if_generation_match=generation)
        except FileExistsError:
            logger.info("[%s] Lost lock race", self.worker_id)
            return False
        except Exception as e:
            if self._is_gcs and "PreconditionFailed" in type(e).__name__:
                logger.info("[%s] Lost lock race", self.worker_id)
                return False
            raise

        return True

    def release_lock(self) -> None:
        """Release the lock if we hold it."""
        try:
            _, lock_data = self._read_lock_with_generation()
            if lock_data and lock_data.worker_id == self.worker_id:
                self.fs.rm(self._lock_path)
                logger.debug("[%s] Released lock", self.worker_id)
        except FileNotFoundError:
            pass

    def has_active_lock(self) -> bool:
        """Check if any worker has an active (non-stale) lock."""
        _, lock_data = self._read_lock_with_generation()
        return lock_data is not None and not lock_data.is_stale()
