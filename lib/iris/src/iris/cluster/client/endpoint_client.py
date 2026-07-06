# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client for the leased service-endpoint registry.

:class:`EndpointClient` owns both the RPC stub and the background lease renewal,
so callers never register an endpoint without keeping it alive: ``register``
registers and starts renewing; ``unregister`` (or ``close``) stops renewing and
deletes. :class:`EndpointLeaseRenewer` is the renewal engine ``EndpointClient``
drives; a crashed task simply stops renewing and its lease expires.
"""

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rigging.timing import Duration, ExponentialBackoff, Timestamp

from iris.cluster.types import EndpointAccess, TaskAttempt
from iris.rpc import controller_pb2
from iris.rpc.errors import call_with_retry
from iris.time_proto import duration_from_proto, duration_to_proto

logger = logging.getLogger(__name__)

# Lease this client requests at registration; the server clamps it to its own
# bounds and the renewer paces off whatever it grants. Short so a crashed task's
# endpoint expires promptly.
DEFAULT_REQUESTED_LEASE = Duration.from_minutes(10)
# Renew a third of the way into the lease, leaving margin for retries; the floor
# keeps short (e.g. test) leases from busy-renewing.
RENEW_FRACTION = 1 / 3
MIN_RENEW_INTERVAL = Duration.from_seconds(30)
# Exponential backoff bounds for failed renewals, sized to land several retries
# inside the requested-lease margin rather than the multi-day legacy default.
RETRY_INITIAL = Duration.from_seconds(10)
RETRY_MAXIMUM = Duration.from_minutes(1)
# Cap on a single sleep so the loop revisits its state; track/untrack/close wake
# it sooner.
_MAX_WAIT = Duration.from_minutes(5)
# Per-call deadline for ListEndpoints.
_LIST_TIMEOUT_MS = 10_000
# Short deadline for the best-effort unregisters on close() so an unreachable
# controller can't stall shutdown; the lease expires on its own regardless.
_CLOSE_TIMEOUT_MS = 5_000

RegisterFn = Callable[
    [controller_pb2.Controller.RegisterEndpointRequest],
    controller_pb2.Controller.RegisterEndpointResponse,
]


class EndpointStub(Protocol):
    """The subset of ``EndpointServiceClientSync`` that :class:`EndpointClient` drives."""

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        *,
        timeout_ms: int | None = ...,
    ) -> controller_pb2.Controller.RegisterEndpointResponse: ...

    def unregister_endpoint(
        self,
        request: controller_pb2.Controller.UnregisterEndpointRequest,
        *,
        timeout_ms: int | None = ...,
    ) -> Any: ...

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        *,
        timeout_ms: int | None = ...,
    ) -> controller_pb2.Controller.ListEndpointsResponse: ...

    def close(self) -> None: ...


class EndpointClient:
    """Registers service endpoints and keeps their leases renewed.

    ``register`` registers an endpoint and starts renewing its lease on a daemon
    thread; the lease is what keeps the controller serving the endpoint, so
    registration without renewal is never exposed. ``close`` stops renewing and
    best-effort unregisters everything still registered before disconnecting.
    """

    def __init__(self, stub: EndpointStub, *, requested_lease: Duration = DEFAULT_REQUESTED_LEASE) -> None:
        self._stub = stub
        self._renewer = EndpointLeaseRenewer(stub.register_endpoint)
        self._registered: set[str] = set()
        self._requested_lease = requested_lease

    def register(
        self,
        name: str,
        address: str,
        task_attempt: TaskAttempt,
        metadata: dict[str, str] | None = None,
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> str:
        """Register an endpoint and renew its lease until ``unregister`` or ``close``.

        ``access`` sets the proxy access mode: PRIVATE (cluster identity required,
        the default), PUBLIC, or BEARER (a scoped endpoint token). It is carried on
        every renewal so it survives re-registration.
        """
        endpoint_id = str(uuid.uuid4())
        request = controller_pb2.Controller.RegisterEndpointRequest(
            name=name,
            address=address,
            task_id=task_attempt.task_id.to_wire(),
            attempt_id=task_attempt.attempt_id if task_attempt.attempt_id is not None else 0,
            metadata=metadata or {},
            endpoint_id=endpoint_id,
            lease_duration=duration_to_proto(self._requested_lease),
            access=int(access),
        )
        response = call_with_retry("register_endpoint", lambda: self._stub.register_endpoint(request))
        self._renewer.track(request, duration_from_proto(response.lease_duration))
        self._renewer.start()
        self._registered.add(response.endpoint_id)
        return response.endpoint_id

    def unregister(self, endpoint_id: str) -> None:
        """Stop renewing ``endpoint_id`` and delete it. Idempotent."""
        self._renewer.untrack(endpoint_id)
        self._registered.discard(endpoint_id)
        self._stub.unregister_endpoint(controller_pb2.Controller.UnregisterEndpointRequest(endpoint_id=endpoint_id))

    def list_endpoints(self, prefix: str, *, exact: bool = False) -> list[controller_pb2.Controller.Endpoint]:
        """List endpoints by name prefix (or exact name when ``exact`` is set)."""

        def _call() -> list[controller_pb2.Controller.Endpoint]:
            request = controller_pb2.Controller.ListEndpointsRequest(prefix=prefix, exact=exact)
            return list(self._stub.list_endpoints(request, timeout_ms=_LIST_TIMEOUT_MS).endpoints)

        return call_with_retry("list_endpoints", _call)

    def close(self) -> None:
        """Stop renewing and best-effort unregister everything still registered.

        Each delete uses a short deadline so an unreachable controller can't
        stall shutdown; anything left registered expires once its lease lapses.
        ``untrack`` waits out any in-flight renewal so the delete cannot be
        undone by a renewal that was mid-RPC.
        """
        self._renewer.close()
        for endpoint_id in list(self._registered):
            self._renewer.untrack(endpoint_id)
            request = controller_pb2.Controller.UnregisterEndpointRequest(endpoint_id=endpoint_id)
            try:
                self._stub.unregister_endpoint(request, timeout_ms=_CLOSE_TIMEOUT_MS)
            except Exception as e:
                logger.warning("Best-effort unregister of endpoint %s on close failed: %s", endpoint_id, e)
        self._registered.clear()
        self._stub.close()


def renew_interval(lease: Duration) -> Duration:
    """Pace renewals at ``RENEW_FRACTION`` of the lease, floored at ``MIN_RENEW_INTERVAL``."""
    interval = lease * RENEW_FRACTION
    return interval if interval >= MIN_RENEW_INTERVAL else MIN_RENEW_INTERVAL


def _retry_backoff() -> ExponentialBackoff:
    return ExponentialBackoff(initial=RETRY_INITIAL.to_seconds(), maximum=RETRY_MAXIMUM.to_seconds(), factor=2.0)


@dataclass
class _Lease:
    request: controller_pb2.Controller.RegisterEndpointRequest
    interval: Duration
    next_renew: Timestamp
    # Per-lease retry schedule; reset on every successful renewal.
    retry: ExponentialBackoff = field(default_factory=_retry_backoff)


class EndpointLeaseRenewer:
    """Re-registers leased endpoints before they expire, on a daemon thread."""

    def __init__(self, register: RegisterFn) -> None:
        self._register = register
        self._cond = threading.Condition(threading.Lock())
        self._leases: dict[str, _Lease] = {}
        # endpoint_ids whose register RPC is currently in flight on the loop
        # thread. untrack() waits these out so an unregister issued right after
        # cannot be undone by a renewal that is mid-flight (see untrack()).
        self._inflight: set[str] = set()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def track(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        lease: Duration,
        *,
        now: Timestamp | None = None,
    ) -> None:
        """Begin renewing ``request``'s endpoint, pacing off the granted ``lease``."""
        now = now or Timestamp.now()
        interval = renew_interval(lease)
        with self._cond:
            self._leases[request.endpoint_id] = _Lease(
                request=request,
                interval=interval,
                next_renew=now.add(interval),
            )
        self._wake.set()

    def untrack(self, endpoint_id: str) -> None:
        """Stop renewing an endpoint; it expires once its current lease elapses.

        Blocks until any in-flight renewal of this endpoint completes, so a
        caller that unregisters immediately after cannot have its delete undone
        by a renewal that was already mid-RPC.
        """
        with self._cond:
            self._leases.pop(endpoint_id, None)
            while endpoint_id in self._inflight:
                self._cond.wait()
        self._wake.set()

    def start(self) -> None:
        """Start the renewal daemon thread (idempotent)."""
        with self._cond:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="endpoint-lease-renewer", daemon=True)
                self._thread.start()

    @property
    def is_running(self) -> bool:
        """Whether the renewal daemon thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def close(self) -> None:
        """Stop the renewal thread. Tracked endpoints expire on their own."""
        self._stop.set()
        self._wake.set()

    def tick(self, now: Timestamp | None = None) -> float:
        """Renew every lease whose deadline has passed; return seconds until the next is due.

        The renewal RPCs run outside the lock so a slow controller cannot stall
        track()/untrack(). Public so the loop and tests share one entry point.
        """
        now = now or Timestamp.now()
        with self._cond:
            due = [lease for lease in self._leases.values() if lease.next_renew <= now]
            self._inflight.update(lease.request.endpoint_id for lease in due)
        for lease in due:
            try:
                self._renew_one(lease, now)
            finally:
                with self._cond:
                    self._inflight.discard(lease.request.endpoint_id)
                    self._cond.notify_all()

        now = Timestamp.now()
        with self._cond:
            if not self._leases:
                return _MAX_WAIT.to_seconds()
            soonest = min(lease.next_renew.epoch_ms() for lease in self._leases.values())
        wait = (soonest - now.epoch_ms()) / 1000.0
        return max(0.0, min(wait, _MAX_WAIT.to_seconds()))

    # -- internals ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            wait = self.tick()
            self._wake.wait(wait)
            self._wake.clear()

    def _renew_one(self, lease: _Lease, now: Timestamp) -> None:
        endpoint_id = lease.request.endpoint_id
        try:
            response = self._register(lease.request)
        except Exception as e:
            delay = Duration.from_seconds(lease.retry.next_interval())
            logger.warning(
                "Failed to renew endpoint lease %s (%s): %s; retrying in %.0fs",
                lease.request.name,
                endpoint_id,
                e,
                delay.to_seconds(),
            )
            self._reschedule(endpoint_id, now.add(delay))
            return
        lease.retry.reset()
        granted = duration_from_proto(response.lease_duration)
        interval = renew_interval(granted) if granted.to_ms() > 0 else lease.interval
        with self._cond:
            current = self._leases.get(endpoint_id)
            if current is not None:
                current.interval = interval
                current.next_renew = now.add(interval)

    def _reschedule(self, endpoint_id: str, next_renew: Timestamp) -> None:
        with self._cond:
            current = self._leases.get(endpoint_id)
            if current is not None:
                current.next_renew = next_renew
