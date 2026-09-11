# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from marin.inference.types import (
    InferenceRequest,
    InferenceResponse,
    InferenceWorkerMetadata,
    LeasedInferenceRequest,
    LeasedInferenceResponse,
    format_request_ids,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Lease(Generic[T]):
    item: T
    lease_id: str
    timestamp: float
    timeout: float


class LeaseQueue(Generic[T]):
    """In-memory leased FIFO queue."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._queue: list[T] = []
        self._leases: dict[str, Lease[T]] = {}

    def push(self, item: T) -> None:
        self._queue.append(item)

    def pop(self, lease_timeout: float) -> Lease[T] | None:
        self._recover_expired_leases()
        if not self._queue:
            return None

        item = self._queue.pop(0)
        lease = Lease(item=item, lease_id=str(uuid.uuid4()), timestamp=self._clock(), timeout=lease_timeout)
        self._leases[lease.lease_id] = lease
        return lease

    def done(self, lease: Lease[T]) -> bool:
        self._recover_expired_leases()
        return self._leases.pop(lease.lease_id, None) is not None

    def size(self) -> int:
        return len(self._queue)

    def pending(self) -> int:
        return len(self._leases)

    def _recover_expired_leases(self) -> None:
        now = self._clock()
        expired = [lease_id for lease_id, lease in self._leases.items() if now - lease.timestamp >= lease.timeout]
        for lease_id in expired:
            lease = self._leases.pop(lease_id)
            self._queue.insert(0, lease.item)


class InferenceBroker:
    """Thread-safe leased request/response broker."""

    def __init__(
        self,
        *,
        request_lease_timeout_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.Lock()
        # Request leases make fetched-but-unanswered work visible again if a
        # worker dies while holding it.
        self._request_lease_timeout_seconds = request_lease_timeout_seconds
        self._requests: LeaseQueue[InferenceRequest] = LeaseQueue(clock=clock)
        self._responses: deque[InferenceResponse] = deque()
        self._request_leases: dict[str, Lease[InferenceRequest]] = {}
        # Insertion-ordered set of request ids for diagnostics.
        self._pending: dict[str, None] = {}
        self._worker_metadata: dict[str, InferenceWorkerMetadata] = {}

    def register_worker(self, worker_id: str, metadata: InferenceWorkerMetadata) -> None:
        with self._lock:
            self._worker_metadata[worker_id] = metadata

    def worker_metadata(self) -> dict[str, InferenceWorkerMetadata]:
        with self._lock:
            return dict(self._worker_metadata)

    def submit_request(self, request: InferenceRequest) -> None:
        with self._lock:
            self._requests.push(request)
            self._pending[request.request_id] = None

    def fetch_requests(self, *, max_items: int) -> list[LeasedInferenceRequest]:
        fetched: list[LeasedInferenceRequest] = []
        with self._lock:
            while len(fetched) < max_items:
                lease = self._requests.pop(self._request_lease_timeout_seconds)
                if lease is None:
                    break
                fetched.append(LeasedInferenceRequest(lease_id=lease.lease_id, request=lease.item))
                self._request_leases[lease.item.request_id] = lease
        return fetched

    def submit_responses(self, responses: Iterable[LeasedInferenceResponse]) -> None:
        dropped_ids: list[str] = []
        with self._lock:
            for leased_response in responses:
                response = leased_response.response
                lease = self._request_leases.get(response.request_id)
                if lease is None or lease.lease_id != leased_response.lease_id or not self._requests.done(lease):
                    dropped_ids.append(response.request_id)
                    if lease is not None and lease.lease_id == leased_response.lease_id:
                        self._request_leases.pop(response.request_id, None)
                    continue
                self._request_leases.pop(response.request_id, None)
                self._responses.append(response)
                self._pending.pop(response.request_id, None)
        if dropped_ids:
            logger.warning(
                "InferenceBroker dropped responses for inactive request leases count=%d request_ids=%s",
                len(dropped_ids),
                format_request_ids(dropped_ids),
            )

    def fetch_responses(self, *, max_items: int) -> list[InferenceResponse]:
        fetched: list[InferenceResponse] = []
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
