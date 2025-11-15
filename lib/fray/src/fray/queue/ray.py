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

"""Ray-based distributed queue implementation."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from typing import Any, TypeVar

import ray
from fray.queue.base import Lease, Queue

logger = logging.getLogger(__name__)

T = TypeVar("T")


@ray.remote
class RayQueueActor:
    """Actor managing queue state and lease monitoring."""

    def __init__(self):
        self.queue = deque()
        # Map lease_id -> (Lease, timeout)
        self.leases: dict[str, tuple[Lease, float]] = {}
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_leases, daemon=True)
        self._monitor_thread.start()

    def push(self, item: Any) -> None:
        self.queue.append(item)

    def peek(self) -> Any | None:
        if not self.queue:
            return None
        return self.queue[0]

    def pop(self, lease_timeout: float) -> Lease | None:
        if not self.queue:
            return None

        item = self.queue.popleft()
        lease_id = str(uuid.uuid4())
        lease = Lease(item=item, lease_id=lease_id, timestamp=time.time())
        self.leases[lease_id] = (lease, lease_timeout)
        return lease

    def done(self, lease_id: str) -> None:
        if lease_id not in self.leases:
            raise ValueError(f"Invalid lease: {lease_id} not found")
        del self.leases[lease_id]

    def release(self, lease_id: str) -> None:
        if lease_id not in self.leases:
            raise ValueError(f"Invalid lease: {lease_id} not found")
        lease, _ = self.leases.pop(lease_id)
        # Requeue at the front for immediate retry
        self.queue.appendleft(lease.item)

    def _monitor_leases(self) -> None:
        """Background thread to check for expired leases."""
        while not self._stop_event.is_set():
            time.sleep(0.1)  # Check frequently to catch short timeouts
            # Make a copy to avoid "dictionary changed size during iteration"
            leases_snapshot = list(self.leases.items())
            for lease_id, (lease, timeout) in leases_snapshot:
                if time.time() - lease.timestamp > timeout:
                    try:
                        self.release(lease_id)
                    except ValueError:
                        # Already released/done by another thread
                        pass

    def shutdown(self) -> None:
        self._stop_event.set()
        self._monitor_thread.join(timeout=1.0)


class RayQueue(Queue[T]):
    """Client for Ray-based queue."""

    def __init__(self, name: str, actor: ray.actor.ActorHandle | None = None):
        self.name = name
        if actor:
            self._actor = actor
        else:
            # Try to get existing actor or create new one
            try:
                self._actor = ray.get_actor(name)
            except ValueError:
                self._actor = RayQueueActor.options(name=name, lifetime="detached").remote()

    def push(self, item: T) -> None:
        ray.get(self._actor.push.remote(item))

    def peek(self) -> T | None:
        return ray.get(self._actor.peek.remote())

    def pop(self, lease_timeout: float = 60.0) -> Lease[T] | None:
        return ray.get(self._actor.pop.remote(lease_timeout))

    def done(self, lease: Lease[T]) -> None:
        ray.get(self._actor.done.remote(lease.lease_id))

    def release(self, lease: Lease[T]) -> None:
        ray.get(self._actor.release.remote(lease.lease_id))
