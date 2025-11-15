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

"""Local queue implementation using multiprocessing for cross-process communication.

This queue implementation uses Python's multiprocessing.Manager to provide
shared state across multiple processes, enabling distributed task processing
within a single machine.
"""

import atexit
import time
import uuid
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from typing import Any, TypeVar

from fray.cluster.queue import Lease

T = TypeVar("T")

# Global shared manager and queue registry for cross-process communication
_global_manager: SyncManager | None = None
_queue_registry: dict[str, tuple[Any, Any, Any]] | None = None  # name -> (available, leased, lock)


def _get_global_manager() -> SyncManager:
    """Get or create the global multiprocessing Manager.

    The manager is created once per process and reused. This allows sharing
    state within a process tree (when using multiprocessing.Process with fork).
    For subprocess.Popen-style workers, each process will have its own manager,
    so cross-process communication requires additional coordination.

    Returns:
        Shared Manager instance
    """
    global _global_manager, _queue_registry
    if _global_manager is None:
        _global_manager = Manager()
        _queue_registry = _global_manager.dict()

        # Ensure manager is shut down cleanly
        def cleanup():
            global _global_manager
            if _global_manager is not None:
                _global_manager.shutdown()
                _global_manager = None

        atexit.register(cleanup)

    return _global_manager


def _get_or_create_queue_state(name: str) -> tuple[Any, Any, Any]:
    """Get or create queue state for a named queue.

    Args:
        name: Queue name

    Returns:
        Tuple of (available_list, leased_dict, lock)
    """
    manager = _get_global_manager()

    assert _queue_registry is not None

    if name not in _queue_registry:
        # Create new queue state
        available = manager.list()
        leased = manager.dict()
        lock = manager.Lock()
        _queue_registry[name] = (available, leased, lock)

    return _queue_registry[name]


class LocalQueue:
    """Thread-safe queue implementation using multiprocessing for cross-process support.

    Uses multiprocessing.Manager() to maintain shared state across worker processes,
    with proper locking to ensure thread-safe operations. Implements the Queue protocol
    with lease semantics for reliable task processing.

    The queue maintains two data structures:
    - _available: List of items ready to be leased
    - _leased: Dict mapping lease_id -> (item, timestamp) for items currently leased

    All operations are protected by a lock to ensure atomicity across processes.
    """

    def __init__(self, name: str):
        """Initialize a new LocalQueue.

        Args:
            name: Unique identifier for this queue instance.
        """
        self._name = name
        self._available, self._leased, self._lock = _get_or_create_queue_state(name)

    def push(self, item: Any) -> None:
        """Add an item to the queue.

        The item will be available for leasing via pop().

        Args:
            item: The task data to add to the queue.
        """
        with self._lock:
            self._available.append(item)

    def peek(self) -> Any | None:
        """View the next available item without acquiring a lease.

        Returns:
            The next item that would be returned by pop(), or None if
            no items are available for leasing.
        """
        with self._lock:
            if not self._available:
                return None
            return self._available[0]

    def pop(self) -> Lease[Any] | None:
        """Acquire a lease on the next available item.

        Returns:
            A Lease containing the item and lease metadata, or None if the
            queue is empty or all items are currently leased.
        """
        with self._lock:
            if not self._available:
                return None

            item = self._available.pop(0)
            lease_id = str(uuid.uuid4())
            timestamp = time.time()

            self._leased[lease_id] = (item, timestamp)

            return Lease(item=item, lease_id=lease_id, timestamp=timestamp)

    def done(self, lease: Lease[Any]) -> None:
        """Mark a leased task as successfully completed and remove it from the queue.

        Args:
            lease: The lease to complete. Must be a valid lease obtained from pop().

        Raises:
            ValueError: If the lease is invalid or has already been completed/released.
        """
        with self._lock:
            if lease.lease_id not in self._leased:
                raise ValueError(f"Invalid lease: {lease.lease_id}")
            del self._leased[lease.lease_id]

    def release(self, lease: Lease[Any]) -> None:
        """Release a lease and requeue the item for reprocessing.

        The item will become available for leasing again via pop().

        Args:
            lease: The lease to release. Must be a valid lease obtained from pop().

        Raises:
            ValueError: If the lease is invalid or has already been completed/released.
        """
        with self._lock:
            if lease.lease_id not in self._leased:
                raise ValueError(f"Invalid lease: {lease.lease_id}")
            item, _ = self._leased[lease.lease_id]
            del self._leased[lease.lease_id]
            self._available.append(item)

    def size(self) -> int:
        """Return the total number of items in the queue.

        This includes both items available for leasing and items currently leased.

        Returns:
            Total count of items in the queue, including leased items.
        """
        with self._lock:
            return len(self._available) + len(self._leased)

    def pending(self) -> int:
        """Return the number of items available for leasing.

        This excludes items that are currently leased out to workers.

        Returns:
            Count of items not yet leased and available via pop().
        """
        with self._lock:
            return len(self._available)
