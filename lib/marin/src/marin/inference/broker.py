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
    STREAM_BUFFER_BYTES,
    WORKER_LEASE_SECONDS,
    BrokerResponse,
    BrokerStatus,
    InferenceBackendState,
    InferenceRequest,
    InferenceResponse,
    InferenceWorkerMetadata,
    LeasedInferenceRequest,
    LeasedInferenceResponse,
    StreamChunk,
    StreamEnd,
    StreamError,
    StreamEvent,
    StreamStart,
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
        if not self._queue:
            return None

        item = self._queue.pop(0)
        lease = Lease(item=item, lease_id=str(uuid.uuid4()), timestamp=self._clock(), timeout=lease_timeout)
        self._leases[lease.lease_id] = lease
        return lease

    def done(self, lease: Lease[T]) -> bool:
        return self._leases.pop(lease.lease_id, None) is not None

    def size(self) -> int:
        return len(self._queue)

    def pending(self) -> int:
        return len(self._leases)

    def discard(self, item: T) -> None:
        self._queue = [queued for queued in self._queue if queued != item]
        self._leases = {key: lease for key, lease in self._leases.items() if lease.item != item}

    def expired(self) -> list[Lease[T]]:
        now = self._clock()
        expired = [lease for lease in self._leases.values() if now - lease.timestamp >= lease.timeout]
        for lease in expired:
            del self._leases[lease.lease_id]
        return expired


@dataclass
class StreamProgress:
    next_sequence: int
    buffered_bytes: int = 0


class InferenceBroker:
    """Thread-safe leased request/response broker."""

    def __init__(
        self,
        *,
        request_lease_timeout_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        # Request leases make fetched-but-unanswered work visible again if a
        # worker dies while holding it.
        self._request_lease_timeout_seconds = request_lease_timeout_seconds
        self._requests: LeaseQueue[InferenceRequest] = LeaseQueue(clock=clock)
        self._responses: deque[BrokerResponse] = deque()
        self._request_leases: dict[str, Lease[InferenceRequest]] = {}
        # Retain pending requests for cancellation and diagnostics.
        self._pending: dict[str, InferenceRequest] = {}
        self._worker_metadata: dict[str, InferenceWorkerMetadata] = {}
        self._worker_heartbeats: dict[str, float] = {}
        self._readiness = InferenceBackendState.RECOVERING
        self._recovery_count = 0
        self._finished = False
        self._streams: dict[str, StreamProgress] = {}
        self._queued_at: dict[str, float] = {}
        self._submitted_requests = 0
        self._dispatched_requests = 0
        self._completed_requests = 0
        self._failed_responses = 0
        self._dropped_responses = 0
        self._queue_wait_total = 0.0
        self._queue_wait_max = 0.0

    def register_worker(self, worker_id: str, metadata: InferenceWorkerMetadata) -> None:
        with self._lock:
            if worker_id in self._worker_metadata:
                self._recovery_count += 1
            self._worker_metadata[worker_id] = metadata
            self._worker_heartbeats[worker_id] = self._clock()

    def heartbeat(self, worker_id: str) -> None:
        with self._lock:
            self._update_readiness()
            self._worker_heartbeats[worker_id] = self._clock()

    def finish(self) -> None:
        """Mark serving permanently finished and fail outstanding requests."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            for request_id, lease in list(self._request_leases.items()):
                progress = self._streams.get(request_id)
                response: BrokerResponse = (
                    StreamError(request_id, lease.lease_id, progress.next_sequence, "inference finished")
                    if progress is not None
                    else InferenceResponse(request_id, 503, b"inference finished")
                )
                self._responses.append(response)
                self._requests.discard(lease.item)
            leased = set(self._request_leases)
            for request_id in self._pending.keys() - leased:
                self._responses.append(InferenceResponse(request_id, 503, b"inference finished"))
                self._requests.discard(self._pending[request_id])
            self._pending.clear()
            self._request_leases.clear()

    def _update_readiness(self) -> InferenceBackendState:
        ready = any(self._clock() - timestamp < WORKER_LEASE_SECONDS for timestamp in self._worker_heartbeats.values())
        state = (
            InferenceBackendState.FINISHED
            if self._finished
            else InferenceBackendState.READY if ready else InferenceBackendState.RECOVERING
        )
        if self._readiness is InferenceBackendState.READY and state is InferenceBackendState.RECOVERING:
            self._recovery_count += 1
        self._readiness = state
        return state

    def worker_metadata(self) -> dict[str, InferenceWorkerMetadata]:
        with self._lock:
            return dict(self._worker_metadata)

    def submit_request(self, request: InferenceRequest) -> None:
        with self._lock:
            if self._finished:
                self._responses.append(InferenceResponse(request.request_id, 503, b"inference finished"))
                return
            self._requests.push(request)
            self._pending[request.request_id] = request
            self._queued_at[request.request_id] = self._clock()
            self._submitted_requests += 1

    def fetch_requests(self, *, max_items: int) -> list[LeasedInferenceRequest]:
        fetched: list[LeasedInferenceRequest] = []
        with self._lock:
            self._expire_requests()
            if self._finished:
                return []
            while len(fetched) < max_items:
                lease = self._requests.pop(self._request_lease_timeout_seconds)
                if lease is None:
                    break
                fetched.append(LeasedInferenceRequest(lease_id=lease.lease_id, request=lease.item))
                self._request_leases[lease.item.request_id] = lease
                queued_at = self._queued_at.pop(lease.item.request_id, None)
                if queued_at is not None:
                    wait = self._clock() - queued_at
                    self._queue_wait_total += wait
                    self._queue_wait_max = max(self._queue_wait_max, wait)
                    self._dispatched_requests += 1
        return fetched

    def submit_responses(self, responses: Iterable[LeasedInferenceResponse]) -> None:
        dropped_ids: list[str] = []
        with self._lock:
            self._expire_requests()
            for leased_response in responses:
                response = leased_response.response
                lease = self._request_leases.get(response.request_id)
                if lease is None or lease.lease_id != leased_response.lease_id or not self._requests.done(lease):
                    dropped_ids.append(response.request_id)
                    self._dropped_responses += 1
                    if lease is not None and lease.lease_id == leased_response.lease_id:
                        self._request_leases.pop(response.request_id, None)
                    continue
                self._request_leases.pop(response.request_id, None)
                self._responses.append(response)
                self._pending.pop(response.request_id, None)
                self._completed_requests += 1
                self._failed_responses += int(response.status_code >= 400)
        if dropped_ids:
            logger.warning(
                "InferenceBroker dropped responses for inactive request leases count=%d request_ids=%s",
                len(dropped_ids),
                format_request_ids(dropped_ids),
            )

    def submit_stream_event(self, event: StreamEvent) -> bool:
        """Accept ordered events only from the current request lease."""
        with self._lock:
            self._expire_requests()
            lease = self._request_leases.get(event.request_id)
            if lease is None or lease.lease_id != event.lease_id:
                return False
            progress = self._streams.get(event.request_id)
            if isinstance(event, StreamStart):
                if progress is not None or event.sequence != 0:
                    return False
                progress = StreamProgress(next_sequence=0)
                self._streams[event.request_id] = progress
            if progress is None or event.sequence != progress.next_sequence:
                return False
            if isinstance(event, StreamChunk):
                progress.buffered_bytes += len(event.payload)
                if progress.buffered_bytes > STREAM_BUFFER_BYTES:
                    self._responses.append(
                        StreamError(event.request_id, event.lease_id, event.sequence, "stream buffer full")
                    )
                    self._complete_stream(lease)
                    return False
            self._responses.append(event)
            progress.next_sequence += 1
            if isinstance(event, (StreamEnd, StreamError)):
                self._complete_stream(lease)
            return True

    def _complete_stream(self, lease: Lease[InferenceRequest]) -> None:
        self._requests.discard(lease.item)
        self._request_leases.pop(lease.item.request_id, None)
        self._pending.pop(lease.item.request_id, None)
        self._completed_requests += 1

    def _expire_requests(self) -> None:
        for lease in self._requests.expired():
            request_id = lease.item.request_id
            self._request_leases.pop(request_id, None)
            progress = self._streams.get(request_id)
            if progress is None:
                self._requests.push(lease.item)
            else:
                self._responses.append(
                    StreamError(request_id, lease.lease_id, progress.next_sequence, "inference lease expired")
                )
                self._pending.pop(request_id, None)

    def cancel_request(self, request_id: str) -> None:
        with self._lock:
            lease = self._request_leases.pop(request_id, None)
            if lease is not None:
                self._requests.discard(lease.item)
            request = self._pending.pop(request_id, None)
            if request is not None:
                self._requests.discard(request)
            self._streams.pop(request_id, None)
            self._queued_at.pop(request_id, None)
            self._responses = deque(response for response in self._responses if response.request_id != request_id)

    def fetch_responses(self, *, max_items: int) -> list[BrokerResponse]:
        fetched: list[BrokerResponse] = []
        with self._lock:
            self._expire_requests()
            while len(fetched) < max_items and self._responses:
                response = self._responses.popleft()
                if isinstance(response, StreamChunk):
                    self._streams[response.request_id].buffered_bytes -= len(response.payload)
                if isinstance(response, (StreamEnd, StreamError)):
                    self._streams.pop(response.request_id, None)
                fetched.append(response)
        return fetched

    def size(self) -> int:
        with self._lock:
            return self._requests.size() + self._requests.pending() + len(self._responses)

    def pending(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def status(self) -> BrokerStatus:
        """Snapshot broker queues and lifetime counters without exposing request contents."""
        with self._lock:
            return BrokerStatus(
                queued_requests=self._requests.size(),
                leased_requests=self._requests.pending(),
                buffered_responses=len(self._responses),
                submitted_requests=self._submitted_requests,
                dispatched_requests=self._dispatched_requests,
                completed_requests=self._completed_requests,
                failed_responses=self._failed_responses,
                dropped_responses=self._dropped_responses,
                mean_queue_wait_seconds=(
                    self._queue_wait_total / self._dispatched_requests if self._dispatched_requests else 0.0
                ),
                max_queue_wait_seconds=self._queue_wait_max,
                workers=dict(self._worker_metadata),
                readiness=self._update_readiness(),
                recovery_count=self._recovery_count,
            )
