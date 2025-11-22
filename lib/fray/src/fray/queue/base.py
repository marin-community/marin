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

"""Queue abstraction with lease semantics for distributed task management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@dataclass
class Lease(Generic[T]):
    """A lease on a queue item, representing exclusive temporary ownership.

    Attributes:
        item: The task data that was leased from the queue.
        lease_id: Unique identifier for this lease.
        timestamp: Unix timestamp (from time.time()) when the lease was acquired.
    """

    item: T
    lease_id: str
    timestamp: float


class Queue(Protocol[T_co]):
    """Distributed queue interface with lease-based task acquisition."""

    def push(self, item: T_co) -> None:
        """Add an item to the queue."""
        ...

    def peek(self) -> T_co | None:
        """View the next available item without acquiring a lease."""
        ...

    def pop(self, lease_timeout: float = 60.0) -> Lease[T_co] | None:
        """Acquire a lease on the next available item."""
        ...

    def done(self, lease: Lease[T_co]) -> None:
        """Mark a leased task as successfully completed."""
        ...

    def release(self, lease: Lease[T_co]) -> None:
        """Release a lease and requeue the item for reprocessing."""
        ...

    def size(self) -> int:
        """Return the total number of items in the queue."""
        ...

    def pending(self) -> int:
        """Return the number of items available for leasing."""
        ...
