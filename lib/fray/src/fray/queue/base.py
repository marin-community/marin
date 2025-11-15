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

import time
import uuid
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


class MemoryQueue(Queue[T_co]):
    def __init__(self):
        self.queue = []
        self.leases = {}  # lease_id -> (item, timestamp, timeout)

    def push(self, item: T_co) -> None:
        self.queue.append(item)

    def peek(self) -> T_co | None:
        self._recover_expired_leases()
        if self.queue:
            return self.queue[0]
        return None

    def _recover_expired_leases(self) -> None:
        """Move expired leases back to the front of the queue."""
        current_time = time.time()
        expired = []
        for lease_id, (_item, timestamp, timeout) in self.leases.items():
            if current_time - timestamp >= timeout:
                expired.append(lease_id)

        for lease_id in expired:
            item, _, _ = self.leases[lease_id]
            self.queue.insert(0, item)
            del self.leases[lease_id]

    def pop(self, lease_timeout: float = 60.0) -> Lease[T_co] | None:
        self._recover_expired_leases()
        if self.queue:
            item = self.queue.pop(0)
            lease_id = str(uuid.uuid4())
            timestamp = time.time()
            lease = Lease(item, lease_id, timestamp)
            self.leases[lease_id] = (item, timestamp, lease_timeout)
            return lease
        return None

    def done(self, lease: Lease[T_co]) -> None:
        if lease.lease_id in self.leases:
            del self.leases[lease.lease_id]

    def release(self, lease: Lease[T_co]) -> None:
        if lease.lease_id in self.leases:
            item, _, _ = self.leases[lease.lease_id]
            self.queue.insert(0, item)
            del self.leases[lease.lease_id]

    def size(self) -> int:
        return len(self.queue)

    def pending(self) -> int:
        return len(self.leases)
