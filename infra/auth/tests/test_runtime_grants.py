# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime model checks against the executable selected by MARIN_AUTH_TEST_OPENFGA_EXECUTABLE."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from rigging.grants import Grant, GrantDenied, GrantStore, Permission, PrincipalAuthority

from infra.auth.openfga import AuthorizationUnavailable, OpenFGARuntime, principal_user, scope_object
from infra.auth.proof_fga import configure_runtime, isolated_openfga

pytestmark = pytest.mark.integration
PERMISSIONS = (Permission("iris", frozenset({"job.read"}), "/jobs/"),)


@pytest.fixture
def operator(tmp_path):
    executable = Path(os.environ["MARIN_AUTH_TEST_OPENFGA_EXECUTABLE"])
    with isolated_openfga(executable, tmp_path) as operator:
        yield operator


@pytest.fixture
def runtime(operator):
    return OpenFGARuntime(**configure_runtime(operator))


@pytest.fixture
def root_grant():
    return Grant("root", "owner", "session", PERMISSIONS, None, "owner", 1000, 2000, 2, None, "revision")


def test_model_binds_actor_and_intersects_every_ancestor_bounds_and_expiry(runtime, root_grant):
    runtime.put(root_grant)
    # Deliberately bypass issuance validation to exercise the engine's ancestor bounds.
    child = replace(
        root_grant,
        id="child",
        actor="task",
        parent_id=root_grant.id,
        expires_at=3000,
        permissions=(Permission("iris", frozenset({"job.read", "job.cancel"}), "/"),),
    )
    runtime.put(child)
    assert runtime.get(child.id) == child
    assert runtime.is_active(child, 1500)
    assert not runtime.is_active(replace(child, actor="other-task"), 1500)
    assert runtime.allows(child, service="iris", action="job.read", resource="/jobs/train", now=1500)
    assert not runtime.allows(child, service="iris", action="job.cancel", resource="/jobs/train", now=1500)
    assert not runtime.allows(child, service="iris", action="job.read", resource="/other/train", now=1500)
    assert not runtime.allows(child, service="loom", action="job.read", resource="/jobs/train", now=1500)
    assert not runtime.is_active(child, 2000)
    runtime.revoke(root_grant, 1600)
    assert not runtime.is_active(child, 1700)
    assert runtime.get(child.id) == child
    assert runtime.get(root_grant.id).revoked_at == 1600


def test_atomic_singleton_preserves_first_actor_bounds_and_parent_on_conflicting_retries(runtime, root_grant):
    first = runtime.put(root_grant)
    conflicting = replace(root_grant, actor="other", parent_id="other-parent", permissions=())
    assert runtime.put(conflicting) == first
    assert not runtime.is_active(conflicting, 1500)
    assert runtime.get(root_grant.id) == first


@pytest.mark.parametrize("actor", ["loom:session-id", "iris:attempt/one", "user:alice@example.org"])
def test_runtime_actor_encoding_preserves_native_identity_and_cannot_alias_percent_text(runtime, root_grant, actor):
    grant = replace(root_grant, actor=actor)
    runtime.put(grant)
    assert runtime.get(grant.id).actor == actor
    assert runtime.is_active(grant, 1500)
    assert runtime.allows(grant, service="iris", action="job.read", resource="/jobs/train", now=1500)
    assert not runtime.is_active(replace(grant, actor=actor.replace(":", "%3A")), 1500)


def test_revocation_marker_prevents_later_activation_of_the_same_identity(runtime, root_grant):
    runtime.revoke(root_grant, 1200)
    runtime.put(root_grant)
    assert not runtime.is_active(root_grant, 1500)
    assert runtime.get(root_grant.id).revoked_at == 1200


def test_concurrent_child_write_and_parent_revocation_cannot_leave_live_authority(runtime, root_grant):
    runtime.put(root_grant)
    child = replace(root_grant, id="child", actor="task", parent_id=root_grant.id)
    barrier = threading.Barrier(2)

    def create():
        barrier.wait(timeout=5)
        return runtime.put(child)

    def revoke():
        barrier.wait(timeout=5)
        runtime.revoke(root_grant, 1600)

    with ThreadPoolExecutor(max_workers=2) as executor:
        writes = [executor.submit(create), executor.submit(revoke)]
        for write in writes:
            write.result(timeout=15)
    assert runtime.get(child.id) == child
    assert not runtime.is_active(child, 1700)
    assert not runtime.allows(child, service="iris", action="job.read", resource="/jobs/train", now=1700)


def test_uncertain_write_and_revoke_results_are_resolved_from_persisted_authority(runtime, root_grant, monkeypatch):
    request = runtime._request

    def lost_response(operation, payload):
        result = request(operation, payload)
        if operation == "write":
            raise AuthorizationUnavailable("Synthetic connection loss after the server committed")
        return result

    monkeypatch.setattr(runtime, "_request", lost_response)
    assert runtime.put(root_grant) == root_grant
    runtime.revoke(root_grant, 1600)
    assert not runtime.is_active(root_grant, 1700)


def test_failed_atomic_write_leaves_no_readable_or_active_grant(runtime, root_grant, monkeypatch):
    request = runtime._request

    def invalid_write(operation, payload):
        if operation == "write":
            payload["writes"]["tuple_keys"][-1]["relation"] = "unknown-relation"
        return request(operation, payload)

    monkeypatch.setattr(runtime, "_request", invalid_write)
    with pytest.raises(AuthorizationUnavailable):
        runtime.put(root_grant)
    assert runtime.get(root_grant.id) is None
    assert not runtime.is_active(root_grant, 1500)


class StandingPolicy:
    def __init__(self):
        self.member = True

    def allows_grant(self, principal, permissions):
        return self.member and principal == "owner" and tuple(permissions) == PERMISSIONS

    def check(self, principal, service, action, resource):
        return self.member and principal == "owner"


def test_operator_reconciliation_accepts_null_conditions_and_repeats_without_duplicate_writes(operator):
    model = Path(__file__).parents[1] / "policy/model.json"
    store_id, model_id = operator.publish("standing", json.loads(model.read_text()))
    relationship = {
        "user": principal_user("service:grafana"),
        "relation": "accessor",
        "object": scope_object("iris", "job.read", "/jobs/"),
    }
    operator.reconcile(store_id, model_id, [relationship])
    rows = operator.request(f"stores/{store_id}/read", {})["tuples"]
    assert rows[0]["key"]["condition"] is None
    operator.reconcile(store_id, model_id, [relationship])
    changed = {**relationship, "user": principal_user("service:other")}
    operator.reconcile(store_id, model_id, [{**changed, "condition": None}])
    for user, allowed in [(relationship["user"], False), (changed["user"], True)]:
        decision = operator.request(
            f"stores/{store_id}/check",
            {
                "authorization_model_id": model_id,
                "tuple_key": {"user": user, "relation": "allowed", "object": relationship["object"]},
                "consistency": "HIGHER_CONSISTENCY",
            },
        )
        assert decision["allowed"] is allowed


def test_store_idempotency_and_current_standing_membership_use_runtime_model(runtime):
    policy = StandingPolicy()
    store = GrantStore(runtime, [PrincipalAuthority("owner", "owner", 1000, 2, "revision")], policy, now=lambda: 1000)
    arguments = {"actor": "session", "permissions": PERMISSIONS, "expires_at": 1800, "idempotency_key": "request"}
    with ThreadPoolExecutor(max_workers=4) as executor:
        grants = list(executor.map(lambda _: store.create("owner", **arguments), range(4)))
    assert len({grant.id for grant in grants}) == 1
    grant = grants[0]
    with pytest.raises(GrantDenied):
        store.create("owner", **{**arguments, "actor": "other-session"})
    child = store.create(
        "session", actor="task", permissions=PERMISSIONS, expires_at=1700, parent_id=grant.id, idempotency_key="task"
    )
    store.revoke(grant.id, principal="owner")
    assert store.grant_for_request(parent_id=grant.id, idempotency_key="task").id == child.id
    with pytest.raises(GrantDenied):
        store.active(child.id)
    with pytest.raises(GrantDenied):
        store.create("owner", **arguments)
    other = store.create("owner", **{**arguments, "idempotency_key": "other-request"})
    policy.member = False
    with pytest.raises(GrantDenied):
        store.authorize(other.id, actor="session", service="iris", action="job.read", resource="/jobs/train")
