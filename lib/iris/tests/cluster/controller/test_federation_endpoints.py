# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-cluster endpoint federation: the parent-side registry, resolve, and
/proxy authorization for endpoints mirrored from a federated child.

The wire-level sync round-trip (child ``federation_sync`` → parent
``apply_sync_batch``) and the ``FederatedEndpointProxy`` HTTP forward are covered
in ``test_federation_endpoint_sync.py`` and ``test_federated_endpoint_proxy.py``.
"""

import pytest
from iris.cluster.controller.dashboard import (
    _authorize_federation_peer_proxy,
    _authorize_proxy,
)
from iris.cluster.controller.endpoint_service import EndpointServiceImpl, ResolvedEndpoint
from iris.cluster.controller.projections.endpoints import EndpointRow, EndpointsProjection
from iris.cluster.types import EndpointAccess, JobName
from iris.rpc import controller_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE
from rigging.server_auth import VerifiedIdentity
from rigging.timing import Timestamp
from starlette.requests import Request

from .conftest import make_job_request, query_task, submit_job

PEER = "cw-rno2a"


def _service(state) -> EndpointServiceImpl:
    return EndpointServiceImpl(db=state._db)


def _live_task(state, name: str = "j") -> JobName:
    return submit_job(state, name, make_job_request(name))[0].task_id


def _remote_row(endpoint_id: str, name: str, task_id: JobName, *, access=EndpointAccess.ENDPOINT_ACCESS_PRIVATE):
    return EndpointRow(
        endpoint_id=endpoint_id,
        name=name,
        address="10.0.0.9:8000",
        task_id=task_id,
        metadata={},
        registered_at=Timestamp.now(),
        lease_deadline=Timestamp.now().add_ms(600_000),
        access=access,
        peer_id=PEER,
    )


# --- set-replace of a peer's mirrored endpoints ------------------------------


def test_replace_remote_for_peer_inserts_and_resolves(state):
    task = _live_task(state)
    proj: EndpointsProjection = state._endpoints
    with state._db.transaction() as cur:
        proj.replace_remote_for_peer(cur, PEER, [_remote_row("e1", "/serve/foo", task)])

    row = proj.resolve("/serve/foo")
    assert row is not None
    assert row.peer_id == PEER
    assert row.address == "10.0.0.9:8000"


def test_replace_remote_for_peer_set_replaces(state):
    task = _live_task(state)
    proj: EndpointsProjection = state._endpoints
    with state._db.transaction() as cur:
        proj.replace_remote_for_peer(
            cur, PEER, [_remote_row("e1", "/serve/foo", task), _remote_row("e2", "/serve/bar", task)]
        )
    # A later snapshot without e2 drops it; e1 stays.
    with state._db.transaction() as cur:
        proj.replace_remote_for_peer(cur, PEER, [_remote_row("e1", "/serve/foo", task)])

    assert proj.resolve("/serve/foo") is not None
    assert proj.resolve("/serve/bar") is None


def test_replace_remote_for_peer_skips_row_with_absent_task(state):
    """An endpoint whose mirrored task row is missing is skipped (FK safety), not fatal."""
    task = _live_task(state)
    proj: EndpointsProjection = state._endpoints
    ghost = JobName.from_wire("/nobody/ghost/0")
    with state._db.transaction() as cur:
        proj.replace_remote_for_peer(
            cur, PEER, [_remote_row("e1", "/serve/foo", task), _remote_row("e2", "/serve/ghost", ghost)]
        )

    assert proj.resolve("/serve/foo") is not None
    assert proj.resolve("/serve/ghost") is None


def test_replace_remote_for_peer_is_isolated_per_peer(state):
    task = _live_task(state)
    proj: EndpointsProjection = state._endpoints
    with state._db.transaction() as cur:
        proj.replace_remote_for_peer(cur, "peer-a", [_remote_row("a1", "/serve/a", task)])
        proj.replace_remote_for_peer(cur, "peer-b", [_remote_row("b1", "/serve/b", task)])
    # Re-syncing peer-a empty must not touch peer-b's endpoints.
    with state._db.transaction() as cur:
        proj.replace_remote_for_peer(cur, "peer-a", [])

    assert proj.resolve("/serve/a") is None
    assert proj.resolve("/serve/b") is not None


# --- resolve_proxy_target: peer_id surfacing + fail-closed ambiguity ---------


def test_resolve_proxy_target_surfaces_peer_and_task(state):
    task = _live_task(state)
    with state._db.transaction() as cur:
        state._endpoints.replace_remote_for_peer(cur, PEER, [_remote_row("e1", "/serve/foo", task)])

    resolved = _service(state).resolve_proxy_target("serve.foo")
    assert resolved is not None
    assert resolved.peer_id == PEER
    assert resolved.task_id == task


def test_resolve_proxy_target_ambiguous_local_and_remote_fails_closed(state):
    """A name that resolves to both a local and a remote row is refused (returns None)."""
    task = _live_task(state)
    svc = _service(state)
    # Local registration (attempt id must match the live task's current attempt).
    svc.register_endpoint(
        controller_pb2.Controller.RegisterEndpointRequest(
            name="/serve/dup",
            address="local:1",
            task_id=task.to_wire(),
            attempt_id=query_task(state, task).current_attempt_id,
        ),
        None,
    )
    # A remote row with the SAME name.
    with state._db.transaction() as cur:
        state._endpoints.replace_remote_for_peer(cur, PEER, [_remote_row("r1", "/serve/dup", task)])

    assert svc.resolve_proxy_target("serve.dup") is None


def test_list_endpoints_reports_peer_id(state):
    task = _live_task(state)
    with state._db.transaction() as cur:
        state._endpoints.replace_remote_for_peer(cur, PEER, [_remote_row("e1", "/serve/foo", task)])

    resp = _service(state).list_endpoints(
        controller_pb2.Controller.ListEndpointsRequest(prefix="/serve/foo", exact=True), None
    )
    assert [e.peer_id for e in resp.endpoints] == [PEER]


# --- federation-peer /proxy authorization -----------------------------------

_OWNED = ResolvedEndpoint(
    name="/serve/foo",
    address="10.0.0.9:8000",
    access=EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    peer_id=None,
    task_id=JobName.from_wire("/alice/serve-foo/0"),
)


def test_federation_peer_authorized_only_for_owned_job():
    def owns(root, peer):
        return root == JobName.from_wire("/alice/serve-foo") and peer == "marin"

    # Owned by the requesting peer -> allowed (access mode not re-checked here).
    assert _authorize_federation_peer_proxy(_OWNED, "marin", owns) is None
    # A different peer that did not hand this job here -> 403.
    assert _authorize_federation_peer_proxy(_OWNED, "other", owns).status_code == 403


def test_federation_peer_denied_without_owner_check():
    assert _authorize_federation_peer_proxy(_OWNED, "marin", None).status_code == 403


def test_federation_peer_denied_for_unknown_or_system_endpoint():
    owns = lambda root, peer: True  # noqa: E731 - would-allow, but resolution must gate first
    assert _authorize_federation_peer_proxy(None, "marin", owns).status_code == 404
    system = ResolvedEndpoint(name="/system/log-server", address="l:1", access=0, peer_id=None, task_id=None)
    assert _authorize_federation_peer_proxy(system, "marin", owns).status_code == 404


class _FakePolicy:
    """Minimal RequestAuthPolicy stand-in: resolves any token to a fixed identity."""

    def __init__(self, identity: VerifiedIdentity | None):
        self._identity = identity

    def resolve(self, token, *, client_address=None, headers=None) -> VerifiedIdentity:
        if self._identity is None:
            raise ValueError("unauthenticated")
        return self._identity


def _request() -> Request:
    return Request({"type": "http", "headers": [], "method": "GET", "client": ("127.0.0.1", 1)})


def test_authorize_proxy_routes_federation_peer_to_ownership_check():
    """A federation-peer identity is authorized by job ownership, not access mode:
    it reaches even a PRIVATE endpoint on an owned job, and is denied on a non-owned one."""
    fed = VerifiedIdentity(user_id="marin", role=FEDERATION_PEER_ROLE)
    owns = lambda root, peer: peer == "marin"  # noqa: E731

    deny = _authorize_proxy(_request(), _OWNED, _FakePolicy(fed), federation_owner_check=owns)
    assert deny is None

    not_owns = lambda root, peer: False  # noqa: E731
    deny = _authorize_proxy(_request(), _OWNED, _FakePolicy(fed), federation_owner_check=not_owns)
    assert deny is not None and deny.status_code == 403


def test_authorize_proxy_regular_identity_still_access_gated():
    """A normal full identity keeps the existing PRIVATE/LINK behavior; the
    federation ownership check does not apply to it."""
    user = VerifiedIdentity(user_id="alice", role="user")
    # PRIVATE endpoint, full identity -> allowed; owner check is irrelevant.
    assert _authorize_proxy(_request(), _OWNED, _FakePolicy(user), federation_owner_check=lambda r, p: False) is None
    # Unauthenticated -> 401.
    assert _authorize_proxy(_request(), _OWNED, _FakePolicy(None)).status_code == 401


@pytest.mark.parametrize("access", [EndpointAccess.ENDPOINT_ACCESS_PRIVATE, EndpointAccess.ENDPOINT_ACCESS_LINK])
def test_federation_peer_reaches_private_and_link(access):
    resolved = ResolvedEndpoint(
        name="/serve/foo", address="a:1", access=access, peer_id=None, task_id=JobName.from_wire("/alice/serve-foo/0")
    )
    assert _authorize_federation_peer_proxy(resolved, "marin", lambda r, p: True) is None
