# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Service-endpoint leasing, end to end.

Covers the leased projection (expiry/sweep), the ``EndpointServiceImpl``
register-or-renew RPC, and the ``EndpointClient`` driving that real service in
process. The lease *renewer* is exercised separately against a deterministic
clock, since its behavior is "when does it re-register", which real wall-clock
time cannot observe.
"""

import uuid

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.client.endpoint_client import EndpointClient, EndpointLeaseRenewer, renew_interval
from iris.cluster.controller.endpoint_service import ENDPOINT_LEASE, EndpointServiceImpl
from iris.cluster.controller.projections.endpoints import EndpointRow, EndpointsProjection
from iris.cluster.controller.schema import tasks_table
from iris.cluster.types import JobName, TaskAttempt
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Duration, ExponentialBackoff, Timestamp
from sqlalchemy import update as sa_update

from .conftest import make_job_request, query_task, submit_job


def _service(state, *, lease: Duration = ENDPOINT_LEASE) -> EndpointServiceImpl:
    return EndpointServiceImpl(db=state._db, endpoints=state._endpoints, lease=lease)


def _row(endpoint_id: str, name: str, task_id: JobName, *, lease_deadline: Timestamp | None) -> EndpointRow:
    return EndpointRow(
        endpoint_id=endpoint_id,
        name=name,
        address="h:1",
        task_id=task_id,
        metadata={},
        registered_at=Timestamp.now(),
        lease_deadline=lease_deadline,
    )


def _register_request(
    name: str, task_id: JobName, *, attempt_id: int, endpoint_id: str = ""
) -> controller_pb2.Controller.RegisterEndpointRequest:
    return controller_pb2.Controller.RegisterEndpointRequest(
        name=name,
        address="h:1",
        task_id=task_id.to_wire(),
        attempt_id=attempt_id,
        endpoint_id=endpoint_id,
    )


def _live_task(state) -> tuple[JobName, int]:
    """Submit a job and return its single task with its current attempt id."""
    task = submit_job(state, "j", make_job_request("j"))[0].task_id
    return task, query_task(state, task).current_attempt_id


class _ServiceStub:
    """In-process ``EndpointStub`` over a real ``EndpointServiceImpl``.

    Drives the actual leased registry — reads go through the projection — while
    recording register/unregister RPCs and optionally failing so tests can
    exercise the retry and best-effort-close paths.
    """

    def __init__(self, service: EndpointServiceImpl, *, fail_register: bool = False, fail_unregister: bool = False):
        self._service = service
        self._fail_register = fail_register
        self._fail_unregister = fail_unregister
        self.registered: list[controller_pb2.Controller.RegisterEndpointRequest] = []
        self.unregistered: list[str] = []
        self.closed = False

    def register_endpoint(self, request, *, timeout_ms=None):
        self.registered.append(request)
        if self._fail_register:
            raise RuntimeError("controller unavailable")
        return self._service.register_endpoint(request, None)

    def unregister_endpoint(self, request, *, timeout_ms=None):
        if self._fail_unregister:
            raise RuntimeError("controller unavailable")
        self.unregistered.append(request.endpoint_id)
        return self._service.unregister_endpoint(request, None)

    def list_endpoints(self, request, *, timeout_ms=None):
        return self._service.list_endpoints(request, None)

    def close(self):
        self.closed = True


# --- Projection: expiry hides rows from reads --------------------------------


def test_expired_lease_hidden_from_reads(state):
    task = submit_job(state, "j", make_job_request("j"))[0].task_id
    past = Timestamp.now().add(Duration.from_ms(-1000))
    future = Timestamp.now().add(Duration.from_hours(1))
    with state._db.transaction() as cur:
        state._endpoints.add(cur, _row("dead", "svc", task, lease_deadline=past))
        state._endpoints.add(cur, _row("live", "svc", task, lease_deadline=future))

    assert [r.endpoint_id for r in state._endpoints.query()] == ["live"]
    assert state._endpoints.get("dead") is None
    assert state._endpoints.get("live") is not None
    # resolve() shares a name across both ids; it must skip the expired one.
    assert state._endpoints.resolve("svc").endpoint_id == "live"
    assert [r.endpoint_id for r in state._endpoints.all()] == ["live"]


def test_null_lease_never_expires(state):
    task = submit_job(state, "j", make_job_request("j"))[0].task_id
    with state._db.transaction() as cur:
        state._endpoints.add(cur, _row("forever", "svc", task, lease_deadline=None))

    assert [r.endpoint_id for r in state._endpoints.query()] == ["forever"]
    with state._db.transaction() as cur:
        assert state._endpoints.sweep_expired(cur, Timestamp.now()) == []
    assert state._endpoints.get("forever") is not None


def test_sweep_expired_deletes_only_expired(state):
    task = submit_job(state, "j", make_job_request("j"))[0].task_id
    past = Timestamp.now().add(Duration.from_ms(-1000))
    future = Timestamp.now().add(Duration.from_hours(1))
    with state._db.transaction() as cur:
        state._endpoints.add(cur, _row("dead", "a", task, lease_deadline=past))
        state._endpoints.add(cur, _row("live", "b", task, lease_deadline=future))

    with state._db.transaction() as cur:
        removed = state._endpoints.sweep_expired(cur, Timestamp.now())
    assert removed == ["dead"]

    # Gone from the index and from storage (a fresh projection re-reads the DB).
    assert state._endpoints.get("dead") is None
    reloaded = EndpointsProjection(state._db)
    assert [r.endpoint_id for r in reloaded.query()] == ["live"]


# --- EndpointService: register-or-renew --------------------------------------


def test_register_returns_lease_duration(state):
    task, attempt = _live_task(state)
    svc = _service(state)

    response = svc.register_endpoint(_register_request("svc", task, attempt_id=attempt), None)

    assert response.endpoint_id
    assert response.lease_duration.milliseconds == ENDPOINT_LEASE.to_ms()
    assert state._endpoints.resolve("svc").address == "h:1"


def test_reregister_renews_expired_endpoint(state):
    """Re-registering the same endpoint_id renews its lease, bringing an expired
    endpoint back into reads — the register-or-renew contract."""
    task, attempt = _live_task(state)
    endpoint_id = str(uuid.uuid4())
    expired = _service(state, lease=Duration.from_ms(0))
    live = _service(state, lease=Duration.from_hours(72))

    expired.register_endpoint(_register_request("svc", task, attempt_id=attempt, endpoint_id=endpoint_id), None)
    assert state._endpoints.query() == []  # lease of 0ms expires immediately

    renewed = live.register_endpoint(_register_request("svc", task, attempt_id=attempt, endpoint_id=endpoint_id), None)
    assert renewed.endpoint_id == endpoint_id
    assert [r.endpoint_id for r in state._endpoints.query()] == [endpoint_id]  # single row, renewed


def test_register_terminal_task_raises(state):
    task = submit_job(state, "j", make_job_request("j"))[0].task_id
    with state._db.transaction() as tx:
        tx.execute(
            sa_update(tasks_table).where(tasks_table.c.task_id == task).values(state=job_pb2.TASK_STATE_SUCCEEDED)
        )
    svc = _service(state)

    with pytest.raises(ConnectError) as excinfo:
        svc.register_endpoint(_register_request("svc", task, attempt_id=0), None)
    assert excinfo.value.code is Code.FAILED_PRECONDITION


def test_system_endpoints_resolve_and_list(state):
    svc = _service(state)
    svc.register_system_endpoint("/system/log-server", "logs:9000")

    assert svc.resolve_endpoint("/system/log-server") == "logs:9000"
    listed = svc.list_endpoints(
        controller_pb2.Controller.ListEndpointsRequest(prefix="/system/log-server", exact=True), None
    )
    assert [(e.name, e.address) for e in listed.endpoints] == [("/system/log-server", "logs:9000")]


# --- EndpointClient: registers and keeps leases against the real service ------


def test_register_resolves_through_service(state):
    task, attempt = _live_task(state)
    stub = _ServiceStub(_service(state))
    client = EndpointClient(stub)

    endpoint_id = client.register("svc", "h:1", TaskAttempt(task_id=task, attempt_id=attempt))

    assert [r.endpoint_id for r in stub.registered] == [endpoint_id]
    assert state._endpoints.resolve("svc").address == "h:1"
    client.close()


def test_close_unregisters_each_registered_endpoint(state):
    task, attempt = _live_task(state)
    stub = _ServiceStub(_service(state))
    client = EndpointClient(stub)
    first = client.register("a", "h:1", TaskAttempt(task_id=task, attempt_id=attempt))
    second = client.register("b", "h:2", TaskAttempt(task_id=task, attempt_id=attempt))

    client.close()

    assert sorted(stub.unregistered) == sorted([first, second])
    assert stub.closed
    assert state._endpoints.query() == []  # deletes landed in the registry


def test_unregister_then_close_does_not_redelete(state):
    task, attempt = _live_task(state)
    stub = _ServiceStub(_service(state))
    client = EndpointClient(stub)
    endpoint_id = client.register("svc", "h:1", TaskAttempt(task_id=task, attempt_id=attempt))

    client.unregister(endpoint_id)
    client.close()

    # Unregistered exactly once: close() must not re-issue a delete for an
    # endpoint already removed from the registry.
    assert stub.unregistered == [endpoint_id]


def test_close_is_best_effort_when_unregister_fails(state):
    task, attempt = _live_task(state)
    stub = _ServiceStub(_service(state), fail_unregister=True)
    client = EndpointClient(stub)
    client.register("svc", "h:1", TaskAttempt(task_id=task, attempt_id=attempt))

    # A failing delete must not abort shutdown; the stub is still closed and the
    # lease is left to expire on its own.
    client.close()

    assert stub.closed


# --- Lease renewer: re-registers on a deterministic clock --------------------


def test_due_lease_is_reregistered_with_same_request(state):
    task, attempt = _live_task(state)
    stub = _ServiceStub(_service(state))
    renewer = EndpointLeaseRenewer(stub.register_endpoint)
    request = _register_request("svc", task, attempt_id=attempt, endpoint_id="e1")
    start = Timestamp.now()
    renewer.track(request, Duration.from_hours(24), now=start)

    interval = renew_interval(Duration.from_hours(24))
    renewer.tick(now=start.add(interval).add(Duration.from_ms(-1)))
    assert stub.registered == []  # not yet due
    renewer.tick(now=start.add(interval).add(Duration.from_ms(1)))
    assert stub.registered == [request]  # re-sent, which the service treats as a renew
    assert state._endpoints.resolve("svc").endpoint_id == "e1"


def test_renewal_paces_off_the_granted_lease(state):
    # Lease asks for a 24h cadence but the service grants 30m; later renewals
    # must follow the grant, not the original ask.
    task, attempt = _live_task(state)
    stub = _ServiceStub(_service(state, lease=Duration.from_minutes(30)))
    renewer = EndpointLeaseRenewer(stub.register_endpoint)
    start = Timestamp.now()
    renewer.track(
        _register_request("svc", task, attempt_id=attempt, endpoint_id="e1"), Duration.from_hours(24), now=start
    )

    first = start.add(renew_interval(Duration.from_hours(24)))
    renewer.tick(now=first.add(Duration.from_ms(1)))
    assert len(stub.registered) == 1

    # One granted interval (10m) later it renews again — far sooner than the 8h
    # the original 24h lease would have implied.
    granted_interval = renew_interval(Duration.from_minutes(30))
    renewer.tick(now=first.add(granted_interval).add(Duration.from_ms(-1)))
    assert len(stub.registered) == 1  # not yet due under the granted cadence
    renewer.tick(now=first.add(granted_interval).add(Duration.from_ms(1)))
    assert len(stub.registered) == 2


def test_failed_renewal_keeps_lease_and_retries(state):
    stub = _ServiceStub(_service(state), fail_register=True)
    renewer = EndpointLeaseRenewer(stub.register_endpoint)
    start = Timestamp.now()
    renewer.track(
        _register_request("svc", JobName.from_wire("/u/j/0"), attempt_id=0, endpoint_id="e1"),
        Duration.from_hours(24),
        now=start,
    )

    due = start.add(renew_interval(Duration.from_hours(24))).add(Duration.from_ms(1))
    renewer.tick(now=due)  # must not raise
    assert len(stub.registered) == 1

    # The lease is kept and retried: a later tick re-attempts the renewal rather
    # than dropping the endpoint.
    renewer.tick(now=due.add(Duration.from_minutes(10)))
    assert len(stub.registered) == 2


def test_untracked_lease_is_not_renewed(state):
    stub = _ServiceStub(_service(state))
    renewer = EndpointLeaseRenewer(stub.register_endpoint)
    start = Timestamp.now()
    renewer.track(
        _register_request("svc", JobName.from_wire("/u/j/0"), attempt_id=0, endpoint_id="e1"),
        Duration.from_hours(24),
        now=start,
    )

    renewer.untrack("e1")
    renewer.tick(now=start.add(Duration.from_hours(24)))

    assert stub.registered == []


def test_start_then_close_stops_the_renewer(state):
    renewer = EndpointLeaseRenewer(_ServiceStub(_service(state)).register_endpoint)
    renewer.start()
    assert renewer.is_running

    renewer.close()
    stopped = ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
        lambda: not renewer.is_running, timeout=Duration.from_seconds(5)
    )
    assert stopped
