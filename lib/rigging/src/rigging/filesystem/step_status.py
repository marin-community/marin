# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Status and lock files of an executor step output.

Each step produces an `output_path`. We associate each `output_path` with:
- A status file (`output_path/.executor_status`) containing simple text: SUCCESS, FAILED, DEP_FAILED, or RUNNING
- A LOCK file (`output_path/.executor_status.lock`) for distributed locking

The LOCK file contains JSON with {worker_id, timestamp} and is refreshed periodically.
On GCS, we use generation-based conditional writes for atomicity.
"""

import json
import logging
import os

from rigging.filesystem.distributed_lock import create_lock
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)

STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_SUCCESS = "SUCCESS"
STATUS_DEP_FAILED = "DEP_FAILED"  # Dependency failed


def get_status_path(output_path: str) -> str:
    """Return the path of the status file associated with `output_path`."""
    return prefix_join(output_path, ".executor_status")


class StatusFile:
    """Manages executor step status with distributed locking.

    Two types of files:
    - LOCK file (JSON): Single file for distributed lock acquisition.
      Contains {worker_id, timestamp}. Must be refreshed periodically.
    - Status file (simple text): Step state - SUCCESS, FAILED, DEP_FAILED, or RUNNING.

    Lock acquisition and release delegate to ``rigging.filesystem.distributed_lock``.
    """

    def __init__(self, output_path: str, worker_id: str):
        self.output_path = output_path
        self.path = get_status_path(output_path)
        self.worker_id = worker_id
        self._lock_path = self.path + ".lock"
        self.fs = url_to_fs(self.path, use_listings_cache=False)[0]
        self._lock = create_lock(self._lock_path, worker_id=worker_id)

    @property
    def status(self) -> str | None:
        """Read current status from status file.

        The modern representation stores a single status token, but older
        executors wrote JSON-line event logs. We support both until the legacy
        files are gone.
        """
        if not self.fs.exists(self.path):
            return None

        # Status changes while peers wait; a ranged S3 read pins the ETag from
        # its metadata lookup and fails if another worker updates the object.
        content = self.fs.cat_file(self.path).decode("utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]

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
        """Write the status file without changing lock ownership.

        The caller that holds the lock owns its lifetime: it stops its heartbeat
        before it releases the lock. Keeping status writes separate from lock
        release prevents the heartbeat from observing the holder's own terminal
        cleanup as lease loss.
        """
        parent = os.path.dirname(self.path)
        if not self.fs.exists(parent):
            self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(self.path, "w") as f:
            f.write(status)

        logger.debug("[%s] Wrote status %s to %s", self.worker_id, status, self.path)

    def refresh_lock(self) -> None:
        """Refresh a lock held by the current worker.

        Raises ``LeaseLostError`` if another worker holds the lock, or if the
        lock disappeared before the heartbeat stopped.
        """
        self._lock.refresh()

    def try_acquire_lock(self) -> bool:
        """Try to acquire the lock using atomic LOCK file."""
        return self._lock.try_acquire()

    def release_lock(self) -> None:
        """Release the lock if we hold it."""
        self._lock.release()

    def active_lock_holder(self) -> str | None:
        """Return the active lock-owner ID, or None if no active lock exists."""
        return self._lock.active_holder_id()
