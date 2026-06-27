# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EndpointClient + EndpointLeaseRenewer.

The renewer is driven through its public surface: ``track`` registers a lease,
``tick(now=...)`` advances the schedule to a chosen instant, and the fake
register records every renewal RPC, so the tests assert on observable renewals
rather than internal scheduler state. ``EndpointClient`` is exercised against a
fake stub that records register/unregister RPCs.
"""

from iris.cluster.client.endpoint_client import EndpointClient, EndpointLeaseRenewer, renew_interval
from iris.cluster.types import JobName, TaskAttempt
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, ExponentialBackoff, Timestamp


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


def test_due_lease_is_reregistered_with_same_request():
    fake = _FakeRegister()
    renewer = EndpointLeaseRenewer(fake)
    request = _request()
    start = Timestamp.now()
    renewer.track(request, Duration.from_hours(24), now=start)

    # Before the renewal is due nothing fires; once past it, the same request
    # (same endpoint_id) is re-sent, which the server treats as a renew.
    renewer.tick(now=start.add(renew_interval(Duration.from_hours(24))).add(Duration.from_ms(-1)))
    assert fake.requests == []
    renewer.tick(now=start.add(renew_interval(Duration.from_hours(24))).add(Duration.from_ms(1)))
    assert fake.requests == [request]


def test_renewal_paces_off_the_granted_lease():
    # Lease asks for a 24h cadence but the server grants 30m; later renewals must
    # follow the grant, not the original ask.
    fake = _FakeRegister(granted=Duration.from_minutes(30))
    renewer = EndpointLeaseRenewer(fake)
    start = Timestamp.now()
    renewer.track(_request(), Duration.from_hours(24), now=start)

    first = start.add(renew_interval(Duration.from_hours(24)))
    renewer.tick(now=first.add(Duration.from_ms(1)))
    assert len(fake.requests) == 1

    # One granted interval (10m) later it renews again — far sooner than the 8h
    # the original 24h lease would have implied.
    granted_interval = renew_interval(Duration.from_minutes(30))
    renewer.tick(now=first.add(granted_interval).add(Duration.from_ms(-1)))
    assert len(fake.requests) == 1  # not yet due under the granted cadence
    renewer.tick(now=first.add(granted_interval).add(Duration.from_ms(1)))
    assert len(fake.requests) == 2


def test_failed_renewal_keeps_lease_and_retries():
    fake = _FakeRegister(raises=True)
    renewer = EndpointLeaseRenewer(fake)
    start = Timestamp.now()
    renewer.track(_request(), Duration.from_hours(24), now=start)

    due = start.add(renew_interval(Duration.from_hours(24))).add(Duration.from_ms(1))
    renewer.tick(now=due)  # must not raise
    assert len(fake.requests) == 1

    # The lease is kept and retried: a later tick re-attempts the renewal rather
    # than dropping the endpoint.
    renewer.tick(now=due.add(Duration.from_minutes(10)))
    assert len(fake.requests) == 2


def test_untracked_lease_is_not_renewed():
    fake = _FakeRegister()
    renewer = EndpointLeaseRenewer(fake)
    start = Timestamp.now()
    renewer.track(_request(), Duration.from_hours(24), now=start)

    renewer.untrack("e1")
    renewer.tick(now=start.add(Duration.from_hours(24)))

    assert fake.requests == []


def test_start_then_close_stops_the_renewer():
    fake = _FakeRegister()
    renewer = EndpointLeaseRenewer(fake)
    renewer.start()
    assert renewer.is_running

    renewer.close()
    stopped = ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
        lambda: not renewer.is_running, timeout=Duration.from_seconds(5)
    )
    assert stopped


# --- EndpointClient ----------------------------------------------------------


class _FakeStub:
    """Records register/unregister RPCs and returns a configurable granted lease."""

    def __init__(self, granted: Duration | None = None, unregister_raises: bool = False):
        self.granted = granted
        self.unregister_raises = unregister_raises
        self.registered: list[controller_pb2.Controller.RegisterEndpointRequest] = []
        self.unregistered: list[str] = []
        self.closed = False

    def register_endpoint(self, request, *, timeout_ms=None):
        self.registered.append(request)
        lease = self.granted if self.granted is not None else Duration.from_hours(72)
        return controller_pb2.Controller.RegisterEndpointResponse(
            endpoint_id=request.endpoint_id,
            lease_duration=duration_to_proto(lease),
        )

    def unregister_endpoint(self, request, *, timeout_ms=None):
        if self.unregister_raises:
            raise RuntimeError("controller unavailable")
        self.unregistered.append(request.endpoint_id)
        return job_pb2.Empty()

    def list_endpoints(self, request, *, timeout_ms=None):
        return controller_pb2.Controller.ListEndpointsResponse()

    def close(self):
        self.closed = True


def _attempt() -> TaskAttempt:
    return TaskAttempt(task_id=JobName.from_wire("/u/j/0"), attempt_id=0)


def test_register_returns_endpoint_id_and_registers():
    stub = _FakeStub()
    client = EndpointClient(stub)

    endpoint_id = client.register("svc", "h:1", _attempt())

    assert [r.endpoint_id for r in stub.registered] == [endpoint_id]
    assert stub.registered[0].name == "svc"


def test_close_unregisters_each_registered_endpoint():
    stub = _FakeStub()
    client = EndpointClient(stub)
    first = client.register("a", "h:1", _attempt())
    second = client.register("b", "h:2", _attempt())

    client.close()

    assert sorted(stub.unregistered) == sorted([first, second])
    assert stub.closed


def test_unregister_then_close_does_not_redelete():
    stub = _FakeStub()
    client = EndpointClient(stub)
    endpoint_id = client.register("svc", "h:1", _attempt())

    client.unregister(endpoint_id)
    client.close()

    # Unregistered exactly once: close() must not re-issue a delete for an
    # endpoint already removed from the registry.
    assert stub.unregistered == [endpoint_id]


def test_close_is_best_effort_when_unregister_fails():
    stub = _FakeStub(unregister_raises=True)
    client = EndpointClient(stub)
    client.register("svc", "h:1", _attempt())

    # A failing delete must not abort shutdown; the stub is still closed and the
    # lease is left to expire on its own.
    client.close()

    assert stub.closed
