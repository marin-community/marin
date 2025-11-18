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

"""File-based queue implementation for cross-subprocess communication."""

import pickle
import time
import uuid
from typing import Any

import fsspec

from fray.cluster.queue import Lease


class FileQueue:
    """
    Fsspec-based queue implementation for cross-subprocess communication.

    Queue structure:
    - {queue_dir}/available/{uuid}.ts-{timestamp_us}.pkl - available tasks
    - {queue_dir}/leased/{uuid}.lease-{deadline_us}.pkl - currently leased tasks

    Files contain pickled user payloads. Metadata (timestamp/deadline) is encoded
    in filenames as microseconds since epoch.
    """

    def __init__(self, name: str, queue_dir: str | None = None):
        """
        Initialize an fsspec-based queue.

        Args:
            name: Unique queue name
            queue_dir: Directory to store queue files (must be set)
        """
        if not queue_dir:
            raise ValueError("queue_dir must be set explicitly")

        self._name = name
        self._base_dir = f"{queue_dir}/{name}"
        self._available_dir = f"{self._base_dir}/available"
        self._leased_dir = f"{self._base_dir}/leased"

        self._fs, _ = fsspec.core.url_to_fs(self._base_dir)
        self._fs.mkdirs(self._available_dir, exist_ok=True)
        self._fs.mkdirs(self._leased_dir, exist_ok=True)

    def _make_available_filename(self, item_id: str, timestamp: float) -> str:
        """Create filename for available item. Timestamp is stored in microseconds."""
        return f"{item_id}.ts-{int(timestamp * 1000000)}.pkl"

    def _make_leased_filename(self, item_id: str, deadline: float) -> str:
        """Create filename for leased item. Deadline is stored in microseconds."""
        return f"{item_id}.lease-{int(deadline * 1000000)}.pkl"

    def _parse_available_filename(self, filename: str) -> tuple[str, float]:
        """Parse available filename to extract item_id and timestamp. Timestamp is in microseconds."""
        name = filename.split("/")[-1]
        item_id, rest = name.split(".ts-")
        timestamp_us = int(rest.replace(".pkl", ""))
        return item_id, timestamp_us / 1000000.0

    def _parse_leased_filename(self, filename: str) -> tuple[str, float]:
        """Parse leased filename to extract item_id and deadline. Deadline is in microseconds."""
        name = filename.split("/")[-1]
        item_id = name.split(".lease-")[0]
        deadline_us = int(name.split(".lease-")[1].replace(".pkl", ""))
        return item_id, deadline_us / 1000000.0

    def _cleanup_expired_leases(self):
        """Move expired leases back to available."""
        current_time = time.time()
        leased_files = self._fs.glob(f"{self._leased_dir}/*.lease-*.pkl")
        for leased_file in leased_files:
            item_id, deadline = self._parse_leased_filename(leased_file)
            if current_time > deadline:
                timestamp = time.time()
                available_filename = self._make_available_filename(item_id, timestamp)
                available_path = f"{self._available_dir}/{available_filename}"
                self._fs.move(leased_file, available_path)

    def push(self, item: Any) -> None:
        """Add an item to the queue."""
        item_id = str(uuid.uuid4())
        timestamp = time.time()
        filename = self._make_available_filename(item_id, timestamp)
        file_path = f"{self._available_dir}/{filename}"
        with self._fs.open(file_path, "wb") as f:
            pickle.dump(item, f)

    def peek(self) -> Any | None:
        """View the next available item without acquiring a lease."""
        files = self._fs.glob(f"{self._available_dir}/*.ts-*.pkl")
        if not files:
            return None

        def get_timestamp(filepath: str) -> float:
            try:
                _, timestamp = self._parse_available_filename(filepath)
                return timestamp
            except Exception:
                return 0

        files_sorted = sorted(files, key=get_timestamp)
        with self._fs.open(files_sorted[0], "rb") as f:
            return pickle.load(f)

    def pop(self, lease_timeout: float = 60.0) -> Lease[Any] | None:
        """Acquire a lease on the next available item. Lease deadline is encoded in the filename."""
        self._cleanup_expired_leases()

        # Keep trying until we successfully lease a file or run out
        while files := self._fs.glob(f"{self._available_dir}/*.ts-*.pkl"):
            files_sorted = sorted(files, key=lambda f: self._parse_available_filename(f)[1])
            if not files_sorted:
                break

            file_path = files_sorted[0]
            item_id, _ = self._parse_available_filename(file_path)
            now = time.time()
            lease_deadline = now + lease_timeout
            leased_filename = self._make_leased_filename(item_id, lease_deadline)
            leased_path = f"{self._leased_dir}/{leased_filename}"

            try:
                with self._fs.open(file_path, "rb") as f:
                    item = pickle.load(f)
                self._fs.move(file_path, leased_path)
                return Lease(item=item, lease_id=item_id, timestamp=now)
            except Exception:
                # File might be gone (race) or corrupt, re-glob and try next
                continue

        return None

    def done(self, lease: Lease[Any]) -> None:
        """Mark a leased task as successfully completed."""
        # Find the leased file with the lease_id prefix
        leased_files = self._fs.glob(f"{self._leased_dir}/{lease.lease_id}.lease-*.pkl")
        if not leased_files:
            raise ValueError(f"Invalid lease: {lease.lease_id}")
        self._fs.rm(leased_files[0])

    def release(self, lease: Lease[Any]) -> None:
        """Release a lease and requeue the item."""
        leased_files = self._fs.glob(f"{self._leased_dir}/{lease.lease_id}.lease-*.pkl")
        if not leased_files:
            raise ValueError(f"Invalid lease: {lease.lease_id}")
        leased_path = leased_files[0]
        timestamp = time.time()
        available_filename = self._make_available_filename(lease.lease_id, timestamp)
        available_path = f"{self._available_dir}/{available_filename}"
        self._fs.move(leased_path, available_path)

    def size(self) -> int:
        """Return the total number of items in the queue."""
        available = len(self._fs.glob(f"{self._available_dir}/*.ts-*.pkl"))
        leased = len(self._fs.glob(f"{self._leased_dir}/*.lease-*.pkl"))
        return available + leased

    def pending(self) -> int:
        """Return the number of items available for leasing."""
        return len(self._fs.glob(f"{self._available_dir}/*.ts-*.pkl"))
