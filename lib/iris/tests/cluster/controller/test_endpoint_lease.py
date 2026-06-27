# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint leasing: expiry filtering, the sweep, and EndpointService register-or-renew."""

import uuid

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller.endpoint_service import ENDPOINT_LEASE, EndpointServiceImpl
from iris.cluster.controller.projections.endpoints import EndpointRow, EndpointsProjection
from iris.cluster.controller.schema import tasks_table
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Duration, Timestamp
from sqlalchemy import update as sa_update

from .conftest import make_job_request, query_task, submit_job


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


# --- Projection: sweep reclaims expired rows ---------------------------------


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
    task = submit_job(state, "j", make_job_request("j"))[0].task_id
    attempt = query_task(state, task).current_attempt_id
    svc = EndpointServiceImpl(db=state._db, endpoints=state._endpoints)

    response = svc.register_endpoint(_register_request("svc", task, attempt_id=attempt), None)

    assert response.endpoint_id
    assert response.lease_duration.milliseconds == ENDPOINT_LEASE.to_ms()
    assert state._endpoints.resolve("svc").address == "h:1"


def test_reregister_renews_expired_endpoint(state):
    """Re-registering the same endpoint_id renews its lease, bringing an expired
    endpoint back into reads — the register-or-renew contract."""
    task = submit_job(state, "j", make_job_request("j"))[0].task_id
    attempt = query_task(state, task).current_attempt_id
    endpoint_id = str(uuid.uuid4())
    expired = EndpointServiceImpl(db=state._db, endpoints=state._endpoints, lease=Duration.from_ms(0))
    live = EndpointServiceImpl(db=state._db, endpoints=state._endpoints, lease=Duration.from_hours(72))

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
    svc = EndpointServiceImpl(db=state._db, endpoints=state._endpoints)

    with pytest.raises(ConnectError) as excinfo:
        svc.register_endpoint(_register_request("svc", task, attempt_id=0), None)
    assert excinfo.value.code is Code.FAILED_PRECONDITION


# --- EndpointService: system endpoints (never leased) ------------------------


def test_system_endpoints_resolve_and_list(state):
    svc = EndpointServiceImpl(db=state._db, endpoints=state._endpoints)
    svc.register_system_endpoint("/system/log-server", "logs:9000")

    assert svc.resolve_endpoint("/system/log-server") == "logs:9000"
    listed = svc.list_endpoints(
        controller_pb2.Controller.ListEndpointsRequest(prefix="/system/log-server", exact=True), None
    )
    assert [(e.name, e.address) for e in listed.endpoints] == [("/system/log-server", "logs:9000")]
