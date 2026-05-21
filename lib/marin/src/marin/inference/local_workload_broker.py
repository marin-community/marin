# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from marin.inference.workload_broker import (
    WorkloadRequest,
    WorkloadResponse,
)

T = TypeVar("T")


@dataclass(frozen=True)
class Lease(Generic[T]):
    item: T
    lease_id: str
    timestamp: float


class MemoryQueue(Generic[T]):
    """In-memory leased FIFO queue."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._queue: list[T] = []
        self._leases: dict[str, tuple[T, float, float]] = {}

    def push(self, item: T) -> None:
        self._queue.append(item)

    def pop(self, lease_timeout: float) -> Lease[T] | None:
        self._recover_expired_leases()
        if not self._queue:
            return None

        item = self._queue.pop(0)
        lease = Lease(item=item, lease_id=str(uuid.uuid4()), timestamp=self._clock())
        self._leases[lease.lease_id] = (item, lease.timestamp, lease_timeout)
        return lease

    def done(self, lease: Lease[T]) -> None:
        self._leases.pop(lease.lease_id, None)

    def size(self) -> int:
        return len(self._queue)

    def pending(self) -> int:
        return len(self._leases)

    def _recover_expired_leases(self) -> None:
        now = self._clock()
        expired = [lease_id for lease_id, (_, timestamp, timeout) in self._leases.items() if now - timestamp >= timeout]
        for lease_id in expired:
            item, _, _ = self._leases.pop(lease_id)
            self._queue.insert(0, item)


class LocalWorkloadBroker:
    """Thread-safe in-process WorkloadBroker implementation."""

    def __init__(
        self,
        *,
        request_lease_timeout_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # The proxy and worker can run in separate event-loop threads, so this
        # broker needs a thread lock rather than an asyncio-loop-bound lock.
        self._lock = threading.Lock()
        # Request leases make fetched-but-unanswered work visible again if the
        # local worker dies while holding it.
        self._request_lease_timeout_seconds = request_lease_timeout_seconds
        self._requests: MemoryQueue[WorkloadRequest] = MemoryQueue(clock=clock)
        self._responses: deque[WorkloadResponse] = deque()
        self._request_leases: dict[str, Lease[WorkloadRequest]] = {}
        # Insertion-ordered set of request ids for diagnostics.
        self._pending: dict[str, None] = {}

    def submit_request(self, request: WorkloadRequest) -> None:
        with self._lock:
            self._requests.push(request)
            self._pending[request.request_id] = None

    def fetch_requests(self, *, max_items: int) -> list[WorkloadRequest]:
        fetched: list[WorkloadRequest] = []
        with self._lock:
            while len(fetched) < max_items:
                lease = self._requests.pop(self._request_lease_timeout_seconds)
                if lease is None:
                    break
                fetched.append(lease.item)
                self._request_leases[lease.item.request_id] = lease
        return fetched

    def submit_responses(self, responses: Iterable[WorkloadResponse]) -> None:
        with self._lock:
            for response in responses:
                self._responses.append(response)
                if lease := self._request_leases.pop(response.request_id, None):
                    self._requests.done(lease)
                self._pending.pop(response.request_id, None)

    def fetch_responses(self, *, max_items: int) -> list[WorkloadResponse]:
        fetched: list[WorkloadResponse] = []
        with self._lock:
            while len(fetched) < max_items and self._responses:
                fetched.append(self._responses.popleft())
        return fetched

    def size(self) -> int:
        with self._lock:
            return self._requests.size() + self._requests.pending() + len(self._responses)

    def pending(self) -> list[str]:
        with self._lock:
            return list(self._pending)
