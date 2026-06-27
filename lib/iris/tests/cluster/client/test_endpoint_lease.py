# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EndpointLeaseRenewer: re-registers leased endpoints before they expire."""

from iris.cluster.client.endpoint_lease import (
    RENEW_RETRY_INTERVAL,
    EndpointLeaseRenewer,
    _Lease,
    renew_interval,
)
from iris.rpc import controller_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, Timestamp


class _FakeRegister:
    """Records register calls and returns a configurable granted lease."""

    def __init__(self, granted: Duration | None = None, raises: bool = False):
        self.granted = granted
        self.raises = raises
        self.requests: list[controller_pb2.Controller.RegisterEndpointRequest] = []

    def __call__(self, request):
        self.requests.append(request)
        if self.raises:
            raise RuntimeError("controller unavailable")
        lease = self.granted if self.granted is not None else Duration.from_hours(72)
        return controller_pb2.Controller.RegisterEndpointResponse(
            endpoint_id=request.endpoint_id,
            lease_duration=duration_to_proto(lease),
        )


def _request(endpoint_id: str = "e1") -> controller_pb2.Controller.RegisterEndpointRequest:
    return controller_pb2.Controller.RegisterEndpointRequest(
        name="svc", address="h:1", task_id="/u/j/0", attempt_id=0, endpoint_id=endpoint_id
    )


def _due_lease(request, interval: Duration) -> _Lease:
    # next_renew in the past so a single _renew_due() tick fires the renewal.
    return _Lease(request=request, interval=interval, next_renew=Timestamp.now().add(Duration.from_ms(-1)))


def test_due_lease_is_reregistered_with_same_request():
    fake = _FakeRegister()
    renewer = EndpointLeaseRenewer(fake)
    request = _request()
    renewer._leases["e1"] = _due_lease(request, Duration.from_hours(24))

    renewer._renew_due()

    assert fake.requests == [request]  # same endpoint_id re-sent → server-side renew
    assert renewer._leases["e1"].next_renew > Timestamp.now()  # rescheduled into the future


def test_renewal_paces_off_the_granted_lease():
    fake = _FakeRegister(granted=Duration.from_minutes(30))
    renewer = EndpointLeaseRenewer(fake)
    renewer._leases["e1"] = _due_lease(_request(), Duration.from_hours(24))

    renewer._renew_due()

    assert renewer._leases["e1"].interval == renew_interval(Duration.from_minutes(30))


def test_failed_renewal_reschedules_soon_and_keeps_lease():
    fake = _FakeRegister(raises=True)
    renewer = EndpointLeaseRenewer(fake)
    renewer._leases["e1"] = _due_lease(_request(), Duration.from_hours(24))

    renewer._renew_due()  # must not raise

    lease = renewer._leases["e1"]  # still tracked
    expected = Timestamp.now().add(RENEW_RETRY_INTERVAL)
    assert abs(lease.next_renew.epoch_ms() - expected.epoch_ms()) < 2000


def test_untracked_lease_is_not_renewed():
    fake = _FakeRegister()
    renewer = EndpointLeaseRenewer(fake)
    renewer._leases["e1"] = _due_lease(_request(), Duration.from_hours(24))

    renewer.untrack("e1")
    renewer._renew_due()

    assert fake.requests == []


def test_close_stops_the_renewal_thread():
    fake = _FakeRegister()
    renewer = EndpointLeaseRenewer(fake)
    renewer.track(_request(), Duration.from_hours(72))  # starts the daemon thread
    thread = renewer._thread
    assert thread is not None and thread.is_alive()

    renewer.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    # 72h lease → first renewal is ~24h out, so nothing fired during the test.
    assert fake.requests == []
