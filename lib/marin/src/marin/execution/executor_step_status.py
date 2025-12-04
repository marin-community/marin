# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Each `ExecutorStep` produces an `output_path`.
We associate each `output_path` with:
- A status file (`output_path/.executor_status`) containing simple text: SUCCESS, FAILURE, or RUNNING
- Lease files (`output_path/.executor_status.leases/*.json`) for distributed locking
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import fsspec

from marin.utils import fsspec_exists

logger = logging.getLogger("marin.executor.status")

# Lease configuration for distributed locking
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
    timestamp: float  # time.time() when lease was written/refreshed

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > HEARTBEAT_TIMEOUT


class StatusFile:
    """Manages executor step status with lease-based distributed locking.

    Two types of files:
    - Lease files (JSON): Workers write these to compete for/hold the lock.
      Contains {worker_id, timestamp}. Must be refreshed periodically.
    - Status file (simple text): Final state - SUCCESS, FAILURE, or RUNNING.
    """

    def __init__(self, output_path: str, worker_id: str):
        self.output_path = output_path
        self.path = get_status_path(output_path)
        self.worker_id = worker_id
        self._lease_dir = self.path + ".leases"
        self._lease_path = os.path.join(self._lease_dir, f"{self.worker_id}.json")

    @property
    def status(self) -> str | None:
        """Read current status from status file (simple text: SUCCESS/FAILURE/RUNNING)."""
        if not fsspec_exists(self.path):
            return None
        with fsspec.open(self.path, "r") as f:
            content = f.read().strip()
            return content or None

    def write_status(self, status: str) -> None:
        """Write final status (SUCCESS/FAILURE/RUNNING)."""
        fs, _ = fsspec.core.url_to_fs(self.path)
        parent = os.path.dirname(self.path)
        if not fs.exists(parent):
            fs.makedirs(parent, exist_ok=True)
        with fsspec.open(self.path, "w") as f:
            f.write(status)
        logger.debug("[%s] Wrote status %s to %s", self.worker_id, status, self.path)

    def write_lease(self) -> None:
        """Write/refresh our lease file."""
        fs, _ = fsspec.core.url_to_fs(self.path)
        if not fs.exists(self._lease_dir):
            fs.makedirs(self._lease_dir, exist_ok=True)

        lease = Lease(worker_id=self.worker_id, timestamp=time.time())
        with fsspec.open(self._lease_path, "w") as f:
            json.dump(asdict(lease), f)
        logger.debug("[%s] Wrote lease at %s", self.worker_id, self._lease_path)

    def release_lease(self) -> None:
        """Remove our lease file."""
        fs, _ = fsspec.core.url_to_fs(self.path)
        try:
            fs.rm(self._lease_path)
            logger.debug("[%s] Released lease", self.worker_id)
        except FileNotFoundError:
            pass

    def read_leases(self) -> list[Lease]:
        """Read all lease files."""
        fs, _ = fsspec.core.url_to_fs(self.path)
        try:
            lease_files = fs.ls(self._lease_dir, detail=False)
        except FileNotFoundError:
            return []

        leases = []
        for lease_path in lease_files:
            if not lease_path.endswith(".json"):
                continue
            try:
                with fsspec.open(lease_path, "r") as f:
                    data = json.load(f)
                    leases.append(Lease(**data))
            except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
                logger.debug("[%s] Skipping malformed lease %s: %s", self.worker_id, lease_path, e)
                continue
        return leases

    def try_acquire_lock(self) -> bool:
        """Try to acquire the lock using lease-based algorithm.

        Write our lease, then check all leases. Earliest non-stale timestamp wins.
        """
        self.write_lease()
        time.sleep(0.05)  # Brief delay for visibility

        leases = self.read_leases()
        # Filter out stale leases, sort by (timestamp, worker_id)
        active = [lease for lease in leases if not lease.is_stale()]
        if not active:
            logger.debug("[%s] No active leases found", self.worker_id)
            return False

        active.sort(key=lambda lease: (lease.timestamp, lease.worker_id))
        winner = active[0]

        logger.debug(
            "[%s] Leases: %s, winner: %s",
            self.worker_id,
            [(lease.worker_id, lease.timestamp) for lease in active],
            winner.worker_id,
        )
        return winner.worker_id == self.worker_id

    def has_active_lease(self) -> bool:
        """Check if any worker has an active (non-stale) lease."""
        return any(not lease.is_stale() for lease in self.read_leases())
