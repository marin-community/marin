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

"""Ray-based distributed queue implementation with lease semantics."""

import time
import uuid
from typing import Any, TypeVar

import ray

from fray.cluster.queue import Lease, Queue

T = TypeVar("T")


@ray.remote
class RayQueueActor:
    """Ray actor for distributed queue state management.

    Maintains queue state in the actor's memory, accessible across
    the Ray cluster. Uses lease semantics for reliable distributed
    task processing.
    """

    def __init__(self):
        """Initialize empty queue state."""
        self._available: list[Any] = []
        self._leased: dict[str, tuple[Any, float]] = {}

    def push(self, item: Any) -> None:
        """Add an item to the available queue."""
        self._available.append(item)

    def peek(self) -> Any | None:
        """Return the next available item without leasing it."""
        if not self._available:
            return None
        return self._available[0]

    def pop(self) -> dict[str, Any] | None:
        """Lease the next available item, returning lease data or None.

        Returns:
            Dict with keys 'item', 'lease_id', 'timestamp', or None if empty.
        """
        if not self._available:
            return None

        item = self._available.pop(0)
        lease_id = str(uuid.uuid4())
        timestamp = time.time()
        self._leased[lease_id] = (item, timestamp)

        return {
            "item": item,
            "lease_id": lease_id,
            "timestamp": timestamp,
        }

    def done(self, lease_id: str) -> None:
        """Remove a completed lease from tracking.

        Args:
            lease_id: The lease identifier to complete.

        Raises:
            ValueError: If the lease_id is not found in leased items.
        """
        if lease_id not in self._leased:
            raise ValueError(f"Invalid lease: {lease_id}")
        del self._leased[lease_id]

    def release(self, lease_id: str) -> None:
        """Release a lease and requeue the item.

        Args:
            lease_id: The lease identifier to release.

        Raises:
            ValueError: If the lease_id is not found in leased items.
        """
        if lease_id not in self._leased:
            raise ValueError(f"Invalid lease: {lease_id}")

        item, _ = self._leased[lease_id]
        del self._leased[lease_id]
        self._available.append(item)

    def size(self) -> int:
        """Return total number of items (available + leased)."""
        return len(self._available) + len(self._leased)

    def pending(self) -> int:
        """Return number of available (unleased) items."""
        return len(self._available)


class RayQueue(Queue[T]):
    """Distributed queue implementation using Ray actors.

    Provides a Queue interface backed by a Ray actor for distributed
    state management. All operations are synchronous from the caller's
    perspective, using ray.get() to wait for actor responses.

    The actor handle is stored and all queue operations are delegated
    to the actor via remote method calls.
    """

    def __init__(self, name: str):
        """Initialize queue with a Ray actor backend.

        Args:
            name: Name identifier for the queue (currently not used for
                  actor registration, but available for future routing).
        """
        self._name = name
        self._actor = RayQueueActor.remote()

    def push(self, item: T) -> None:
        """Add an item to the queue."""
        ray.get(self._actor.push.remote(item))

    def peek(self) -> T | None:
        """View the next available item without acquiring a lease."""
        return ray.get(self._actor.peek.remote())

    def pop(self) -> Lease[T] | None:
        """Acquire a lease on the next available item."""
        result = ray.get(self._actor.pop.remote())
        if result is None:
            return None

        return Lease(
            item=result["item"],
            lease_id=result["lease_id"],
            timestamp=result["timestamp"],
        )

    def done(self, lease: Lease[T]) -> None:
        """Mark a leased task as completed and remove it from the queue."""
        ray.get(self._actor.done.remote(lease.lease_id))

    def release(self, lease: Lease[T]) -> None:
        """Release a lease and requeue the item for reprocessing."""
        ray.get(self._actor.release.remote(lease.lease_id))

    def size(self) -> int:
        """Return the total number of items in the queue."""
        return ray.get(self._actor.size.remote())

    def pending(self) -> int:
        """Return the number of items available for leasing."""
        return ray.get(self._actor.pending.remote())
