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

"""File-based queue implementation for cross-subprocess communication.

This queue implementation uses the filesystem for state management, enabling
reliable cross-process communication even when using subprocess.Popen (which
doesn't share memory like multiprocessing.Process does).
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fray.cluster.queue import Lease


class FileQueue:
    """File-based queue implementation for cross-subprocess communication.

    Uses the filesystem to store queue state, allowing multiple independent
    processes (even those started via subprocess.Popen) to share a queue.

    Queue structure:
    - {queue_dir}/available/{uuid}.json - available tasks
    - {queue_dir}/leased/{uuid}.json - currently leased tasks
    - {queue_dir}/.lock - simple file-based lock (not perfect but good enough)
    """

    def __init__(self, name: str, queue_dir: Path | None = None):
        """Initialize a file-based queue.

        Args:
            name: Unique queue name
            queue_dir: Directory to store queue files (default: /tmp/fray_queues)
        """
        if queue_dir is None:
            queue_dir = Path("/tmp/fray_queues")

        self._name = name
        self._base_dir = queue_dir / name
        self._available_dir = self._base_dir / "available"
        self._leased_dir = self._base_dir / "leased"

        # Create directories
        self._available_dir.mkdir(parents=True, exist_ok=True)
        self._leased_dir.mkdir(parents=True, exist_ok=True)

    def push(self, item: Any) -> None:
        """Add an item to the queue."""
        item_id = str(uuid.uuid4())
        file_path = self._available_dir / f"{item_id}.json"

        with open(file_path, "w") as f:
            json.dump({"item": item, "timestamp": time.time()}, f)

    def peek(self) -> Any | None:
        """View the next available item without acquiring a lease."""
        files = list(self._available_dir.glob("*.json"))
        if not files:
            return None

        # Sort by timestamp to maintain FIFO order
        def get_timestamp(filepath: Path) -> float:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    return data.get("timestamp", 0)
            except (FileNotFoundError, json.JSONDecodeError):
                return 0

        files_sorted = sorted(files, key=get_timestamp)
        with open(files_sorted[0]) as f:
            data = json.load(f)
            return data["item"]

    def pop(self) -> Lease[Any] | None:
        """Acquire a lease on the next available item.

        Also checks for expired leases (older than 60 seconds) and requeues them.
        """
        # Check for expired leases and requeue them
        current_time = time.time()
        lease_timeout = 60.0  # seconds

        for leased_file in self._leased_dir.glob("*.json"):
            try:
                with open(leased_file) as f:
                    data = json.load(f)

                # Check if lease has expired
                lease_time = data.get("lease_time", 0)
                if current_time - lease_time > lease_timeout:
                    # Requeue expired lease
                    available_path = self._available_dir / leased_file.name
                    leased_file.rename(available_path)
            except (FileNotFoundError, json.JSONDecodeError):
                # File was removed or corrupted, skip
                continue

        # Find oldest available item (sort by timestamp for FIFO)
        files = list(self._available_dir.glob("*.json"))
        if not files:
            return None

        # Sort by timestamp to maintain FIFO order
        def get_timestamp(filepath: Path) -> float:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    return data.get("timestamp", 0)
            except (FileNotFoundError, json.JSONDecodeError):
                return 0

        files_sorted = sorted(files, key=get_timestamp)

        # Try to move file to leased directory (atomic operation)
        for file_path in files_sorted:
            lease_id = file_path.stem
            leased_path = self._leased_dir / f"{lease_id}.json"

            try:
                # Read the item first
                with open(file_path) as f:
                    data = json.load(f)

                # Add lease time
                data["lease_time"] = time.time()

                # Write with lease time, then rename (atomic)
                temp_path = file_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f)

                temp_path.rename(file_path)
                file_path.rename(leased_path)

                return Lease(
                    item=data["item"],
                    lease_id=lease_id,
                    timestamp=data["lease_time"],
                )
            except FileNotFoundError:
                # Another process got it first, try next file
                continue

        return None

    def done(self, lease: Lease[Any]) -> None:
        """Mark a leased task as successfully completed."""
        leased_path = self._leased_dir / f"{lease.lease_id}.json"

        if not leased_path.exists():
            raise ValueError(f"Invalid lease: {lease.lease_id}")

        leased_path.unlink()

    def release(self, lease: Lease[Any]) -> None:
        """Release a lease and requeue the item."""
        leased_path = self._leased_dir / f"{lease.lease_id}.json"

        if not leased_path.exists():
            raise ValueError(f"Invalid lease: {lease.lease_id}")

        # Read the item and update timestamp to put it at the back of the queue
        with open(leased_path) as f:
            data = json.load(f)

        data["timestamp"] = time.time()  # Update to current time

        # Write back with new timestamp
        available_path = self._available_dir / f"{lease.lease_id}.json"
        with open(available_path, "w") as f:
            json.dump(data, f)

        # Remove the leased file
        leased_path.unlink()

    def size(self) -> int:
        """Return the total number of items in the queue."""
        available = len(list(self._available_dir.glob("*.json")))
        leased = len(list(self._leased_dir.glob("*.json")))
        return available + leased

    def pending(self) -> int:
        """Return the number of items available for leasing."""
        return len(list(self._available_dir.glob("*.json")))
