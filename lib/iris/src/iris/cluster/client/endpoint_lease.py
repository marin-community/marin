# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Background renewal of leased service endpoints.

The controller grants each registered endpoint a lease and drops it if the
lease is not renewed. :class:`EndpointLeaseRenewer` re-registers every tracked
endpoint with the same ``endpoint_id`` (an idempotent upsert/renew on the
controller) at a fraction of the granted lease, on one daemon thread. Untracking
on unregister — or the process simply dying — stops renewals, so a crashed
task's endpoint expires on its own.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from rigging.timing import Duration, Timestamp

from iris.rpc import controller_pb2
from iris.time_proto import duration_from_proto

logger = logging.getLogger(__name__)

# Renew this far into the lease, so a renewal plus a couple of retries still
# land before expiry. The floor keeps short (e.g. test) leases from busy-renewing.
RENEW_FRACTION = 1 / 3
MIN_RENEW_INTERVAL = Duration.from_seconds(30)
# Re-attempt this soon after a failed renewal — short relative to any lease, so a
# transient controller blip costs little lease margin.
RENEW_RETRY_INTERVAL = Duration.from_seconds(30)
# Cap on a single sleep so the loop revisits its state even if every lease is
# far out; track()/untrack()/close() wake it sooner regardless.
_MAX_WAIT = Duration.from_minutes(5)

RegisterFn = Callable[
    [controller_pb2.Controller.RegisterEndpointRequest],
    controller_pb2.Controller.RegisterEndpointResponse,
]


def renew_interval(lease: Duration) -> Duration:
    """Pace renewals at ``RENEW_FRACTION`` of the lease, floored at ``MIN_RENEW_INTERVAL``."""
    interval = lease * RENEW_FRACTION
    return interval if interval >= MIN_RENEW_INTERVAL else MIN_RENEW_INTERVAL


@dataclass
class _Lease:
    request: controller_pb2.Controller.RegisterEndpointRequest
    interval: Duration
    next_renew: Timestamp


class EndpointLeaseRenewer:
    """Re-registers leased endpoints before they expire, on a daemon thread."""

    def __init__(self, register: RegisterFn) -> None:
        self._register = register
        self._lock = threading.Lock()
        self._leases: dict[str, _Lease] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def track(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        lease: Duration,
    ) -> None:
        """Begin renewing ``request``'s endpoint, pacing off the granted ``lease``."""
        interval = renew_interval(lease)
        with self._lock:
            self._leases[request.endpoint_id] = _Lease(
                request=request,
                interval=interval,
                next_renew=Timestamp.now().add(interval),
            )
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="endpoint-lease-renewer", daemon=True)
                self._thread.start()
        self._wake.set()

    def untrack(self, endpoint_id: str) -> None:
        """Stop renewing an endpoint; it expires once its current lease elapses."""
        with self._lock:
            self._leases.pop(endpoint_id, None)
        self._wake.set()

    def close(self) -> None:
        """Stop the renewal thread. Tracked endpoints expire on their own."""
        self._stop.set()
        self._wake.set()

    # -- internals ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            wait = self._renew_due()
            self._wake.wait(wait)
            self._wake.clear()

    def _renew_due(self) -> float:
        """Renew leases whose deadline has passed; return seconds until the next is due."""
        now = Timestamp.now()
        with self._lock:
            due = [lease for lease in self._leases.values() if lease.next_renew <= now]
        for lease in due:
            self._renew_one(lease)

        now = Timestamp.now()
        with self._lock:
            if not self._leases:
                return _MAX_WAIT.to_seconds()
            soonest = min(lease.next_renew.epoch_ms() for lease in self._leases.values())
        wait = (soonest - now.epoch_ms()) / 1000.0
        return max(0.0, min(wait, _MAX_WAIT.to_seconds()))

    def _renew_one(self, lease: _Lease) -> None:
        endpoint_id = lease.request.endpoint_id
        try:
            response = self._register(lease.request)
        except Exception as e:
            logger.warning(
                "Failed to renew endpoint lease %s (%s): %s; retrying in %.0fs",
                lease.request.name,
                endpoint_id,
                e,
                RENEW_RETRY_INTERVAL.to_seconds(),
            )
            self._reschedule(endpoint_id, RENEW_RETRY_INTERVAL)
            return
        granted = duration_from_proto(response.lease_duration)
        interval = renew_interval(granted) if granted.to_ms() > 0 else lease.interval
        with self._lock:
            current = self._leases.get(endpoint_id)
            if current is not None:
                current.interval = interval
                current.next_renew = Timestamp.now().add(interval)

    def _reschedule(self, endpoint_id: str, delay: Duration) -> None:
        with self._lock:
            current = self._leases.get(endpoint_id)
            if current is not None:
                current.next_renew = Timestamp.now().add(delay)
