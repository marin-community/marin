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

"""Queue abstraction with lease semantics for distributed task management.

The lease pattern enables failure recovery in distributed systems. When a worker
acquires a task via `pop()`, it receives a Lease that must be either completed
via `done()` or released via `release()`. If a worker dies while holding a lease,
the lease can timeout and the task can be automatically requeued.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@dataclass
class Lease(Generic[T]):
    """A lease on a queue item, representing exclusive temporary ownership.

    Attributes:
        item: The task data that was leased from the queue.
        lease_id: Unique identifier for this lease, used to track and manage the lease.
        timestamp: Unix timestamp (from time.time()) when the lease was acquired.
    """

    item: T
    lease_id: str
    timestamp: float


class Queue(Protocol[T_co]):
    """Distributed queue interface with lease-based task acquisition.

    This protocol defines a queue that supports lease semantics for reliable
    distributed task processing. Tasks are acquired via leases that must be
    explicitly completed or released.

    The lease pattern provides:
    - Exclusive access: Only one worker can hold a lease on a task at a time
    - Failure recovery: Leases can timeout if workers die
    - Explicit completion: Tasks are only removed after successful processing
    - Requeuing: Failed tasks can be returned to the queue

    Type parameter T_co is covariant, allowing Queue[Derived] to be used where
    Queue[Base] is expected.
    """

    def push(self, item: T_co) -> None:
        """Add an item to the queue.

        The item will be available for leasing via `pop()`.

        Args:
            item: The task data to add to the queue.
        """
        ...

    def peek(self) -> T_co | None:
        """View the next available item without acquiring a lease.

        This is a non-blocking operation that returns immediately.

        Returns:
            The next item that would be returned by `pop()`, or None if
            no items are available for leasing.
        """
        ...

    def pop(self) -> Lease[T_co] | None:
        """Acquire a lease on the next available item.

        This is a non-blocking operation that returns immediately. The returned
        lease represents exclusive ownership of the item until it is either
        completed via `done()` or released via `release()`.

        Returns:
            A Lease containing the item and lease metadata, or None if the
            queue is empty or all items are currently leased.
        """
        ...

    def done(self, lease: Lease[T_co]) -> None:
        """Mark a leased task as successfully completed and remove it from the queue.

        This should be called when a worker successfully processes a task.
        The item will be permanently removed from the queue.

        Args:
            lease: The lease to complete. Must be a valid lease obtained from `pop()`.

        Raises:
            ValueError: If the lease is invalid or has already been completed/released.
        """
        ...

    def release(self, lease: Lease[T_co]) -> None:
        """Release a lease and requeue the item for reprocessing.

        This should be called when a worker fails to process a task or needs to
        abandon it. The item will become available for leasing again via `pop()`.

        Args:
            lease: The lease to release. Must be a valid lease obtained from `pop()`.

        Raises:
            ValueError: If the lease is invalid or has already been completed/released.
        """
        ...

    def size(self) -> int:
        """Return the total number of items in the queue.

        This includes both items available for leasing and items currently leased.

        Returns:
            Total count of items in the queue, including leased items.
        """
        ...

    def pending(self) -> int:
        """Return the number of items available for leasing.

        This excludes items that are currently leased out to workers.

        Returns:
            Count of items not yet leased and available via `pop()`.
        """
        ...
