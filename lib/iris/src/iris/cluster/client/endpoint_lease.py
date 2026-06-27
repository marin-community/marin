# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Background renewal of leased service endpoints.

The controller grants each registered endpoint a lease and drops it once the
lease elapses without a renewal. :class:`EndpointLeaseRenewer` re-registers every
tracked endpoint with the same ``endpoint_id`` — an idempotent upsert/renew on
the controller — at a fraction of the granted lease, on one daemon thread.
Renewal stops when the endpoint is untracked (on unregister) or the process
dies, so a crashed task's endpoint expires on its own.

The renewer is driven by :meth:`tick`, which renews every due lease and returns
how long to wait before the next one. :meth:`start` runs that on a daemon thread;
tests call :meth:`tick` directly with an injected ``now`` to step the schedule
deterministically.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from rigging.timing import Duration, ExponentialBackoff, Timestamp

from iris.rpc import controller_pb2
from iris.time_proto import duration_from_proto

logger = logging.getLogger(__name__)

# Renew this far into the lease, so a renewal plus a couple of retries still land
# before expiry. The floor keeps short (e.g. test) leases from busy-renewing.
RENEW_FRACTION = 1 / 3
MIN_RENEW_INTERVAL = Duration.from_seconds(30)
# After a failed renewal, back off from this initial delay up to this cap. Both
# stay short relative to any real lease, so a transient controller blip costs
# little lease margin while a sustained outage stops hammering the controller.
RETRY_INITIAL = Duration.from_seconds(30)
RETRY_MAXIMUM = Duration.from_minutes(5)
# Cap on a single sleep so the loop revisits its state even if every lease is far
# out; track()/untrack()/close() wake it sooner regardless.
_MAX_WAIT = Duration.from_minutes(5)

RegisterFn = Callable[
    [controller_pb2.Controller.RegisterEndpointRequest],
    controller_pb2.Controller.RegisterEndpointResponse,
]


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
