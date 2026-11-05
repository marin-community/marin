# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import replace

import pytest
from rigging.grants import GrantDenied, GrantStore, Permission, PrincipalAuthority


@pytest.fixture
def clock():
    return [time.time()]


@pytest.fixture
def authorities():
    return (
        PrincipalAuthority(
            principal="user:alice",
            owner="user:alice",
            max_ttl=600,
            delegation_depth=2,
            revision="policy-v1",
        ),
    )


@pytest.fixture
def store(tmp_path, authorities, grant_policy, clock):
    database = GrantStore(tmp_path / "grants.db", authorities, grant_policy, now=lambda: clock[0])
    yield database
    database.close()


def test_multi_service_grant_delegates_only_parent_bounds(store, clock, grant_permissions):
    parent = store.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)
    child = store.create(
        "loom:one",
        actor="iris:job",
        permissions=(Permission("iris", frozenset({"job.submit"}), "/alice/experiment/"),),
        expires_at=clock[0] + 100,
        parent_id=parent.id,
    )
    result = store.authorize(
        child.id, actor="iris:job", service="iris", action="job.submit", resource="/alice/experiment/train"
    )
    assert result.owner == "user:alice"
    assert result.parent_id == parent.id
    assert store.authority_principal(child.id) == "user:alice"
    with pytest.raises(GrantDenied):
        store.authorize(
            child.id, actor="iris:job", service="iris", action="job.read", resource="/alice/experiment/train"
        )
    with pytest.raises(GrantDenied):
        store.create(
            "iris:job",
            actor="iris:next",
            permissions=child.permissions,
            expires_at=child.expires_at,
            parent_id=child.id,
        )


@pytest.mark.parametrize(
    "permission",
    [
        Permission("iris", frozenset({"job.read"}), "/alice/experiment/"),
        Permission("iris", frozenset({"job.submit"}), "/alice/"),
        Permission("loom", frozenset({"session.create"}), "/alice/"),
    ],
)
def test_delegation_rejects_policy_allowed_actions_resources_and_services_outside_parent(store, clock, permission):
    parent = store.create(
        "user:alice",
        actor="loom:one",
        permissions=(Permission("iris", frozenset({"job.submit"}), "/alice/experiment/"),),
        expires_at=clock[0] + 300,
    )
    with pytest.raises(GrantDenied):
        store.create(
            "loom:one", actor="iris:job", permissions=(permission,), expires_at=clock[0] + 100, parent_id=parent.id
        )


def test_sessions_cannot_use_or_delegate_each_others_grants(store, clock, grant_permissions):
    grant = store.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)
    with pytest.raises(GrantDenied):
        store.authorize(grant.id, actor="loom:two", service="iris", action="job.submit", resource="/alice/train")
    with pytest.raises(GrantDenied):
        store.create(
            "loom:two",
            actor="iris:job",
            permissions=grant.permissions,
            expires_at=clock[0] + 100,
            parent_id=grant.id,
        )
    with pytest.raises(GrantDenied):
        store.revoke(grant.id, principal="loom:two")


@pytest.mark.parametrize("resource", ["/alice-other/train", "/alice/../bob/train", "/alice/%2e%2e/bob/train"])
def test_service_authorization_cannot_escape_namespace(store, clock, grant_permissions, resource):
    grant = store.create("user:alice", actor="user:alice", permissions=grant_permissions, expires_at=clock[0] + 300)
    with pytest.raises(ValueError):
        store.authorize(grant.id, actor="user:alice", service="iris", action="job.submit", resource=resource)


@pytest.mark.parametrize("expiry", [float("inf"), float("nan")])
def test_nonfinite_delegation_cannot_bypass_parent_expiry(store, clock, grant_permissions, expiry):
    parent = store.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)
    with pytest.raises(GrantDenied):
        store.create(
            "loom:one", actor="iris:job", permissions=parent.permissions, expires_at=expiry, parent_id=parent.id
        )


def test_grant_and_ancestor_expiry_apply_at_decision_time(store, clock, grant_permissions):
    parent = store.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)
    with pytest.raises(GrantDenied):
        store.create(
            "loom:one",
            actor="iris:job",
            permissions=parent.permissions,
            expires_at=parent.expires_at + 1,
            parent_id=parent.id,
        )
    child = store.create(
        "loom:one",
        actor="iris:job",
        permissions=parent.permissions,
        expires_at=clock[0] + 100,
        parent_id=parent.id,
    )
    clock[0] = child.expires_at
    with pytest.raises(GrantDenied):
        store.authorize(child.id, actor="iris:job", service="iris", action="job.submit", resource="/alice/train")


def test_revocation_survives_restart_and_invalidates_descendants(
    tmp_path, authorities, grant_permissions, grant_policy, clock
):
    path = tmp_path / "grants.db"
    database = GrantStore(path, authorities, grant_policy, now=lambda: clock[0])
    try:
        parent = database.create(
            "user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300
        )
        child = database.create(
            "loom:one",
            actor="iris:job",
            permissions=(Permission("iris", frozenset({"job.submit"}), "/alice/"),),
            expires_at=clock[0] + 100,
            parent_id=parent.id,
        )
        database.revoke(parent.id, principal="user:alice")
    finally:
        database.close()
    reopened = GrantStore(path, authorities, grant_policy, now=lambda: clock[0])
    try:
        with pytest.raises(GrantDenied):
            reopened.authorize(child.id, actor="iris:job", service="iris", action="job.submit", resource="/alice/train")
        with pytest.raises(GrantDenied):
            reopened.authority_principal(child.id)
    finally:
        reopened.close()


def test_policy_denial_prevents_root_grant_creation(store, grant_policy, grant_permissions, clock):
    grant_policy.permissions.clear()
    with pytest.raises(GrantDenied):
        store.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)


def test_policy_reduction_invalidates_existing_delegation_chain(store, grant_policy, grant_permissions, clock):
    parent = store.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)
    child = store.create(
        "loom:one",
        actor="iris:job",
        permissions=(Permission("iris", frozenset({"job.submit"}), "/alice/experiment/"),),
        expires_at=clock[0] + 100,
        parent_id=parent.id,
    )
    grant_policy.permissions["user:alice"] = child.permissions
    with pytest.raises(GrantDenied):
        store.authorize(
            child.id, actor="iris:job", service="iris", action="job.submit", resource="/alice/experiment/train"
        )
    with pytest.raises(GrantDenied):
        store.authority_principal(child.id)


def test_concrete_policy_denial_applies_within_an_active_grant(store, grant_policy, grant_permissions, clock):
    grant = store.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)
    grant_policy.denied_resources.add("/alice/blocked")
    assert store.active(grant.id).id == grant.id
    assert (
        store.authorize(grant.id, actor="loom:one", service="iris", action="job.submit", resource="/alice/train").id
        == grant.id
    )
    with pytest.raises(GrantDenied):
        store.authorize(grant.id, actor="loom:one", service="iris", action="job.submit", resource="/alice/blocked")


def test_child_policy_denial_blocks_issuance_and_existing_descendants(store, grant_policy, grant_permissions, clock):
    parent = store.create("user:alice", actor="user:alice", permissions=grant_permissions, expires_at=clock[0] + 300)
    child_permissions = (Permission("iris", frozenset({"job.submit"}), "/alice/restricted/"),)
    grant_policy.denied_grant_resources.add("/alice/restricted/")
    with pytest.raises(GrantDenied):
        store.create(
            "user:alice", actor="loom:one", permissions=child_permissions, expires_at=clock[0] + 100, parent_id=parent.id
        )
    grant_policy.denied_grant_resources.clear()
    child = store.create(
        "user:alice", actor="loom:one", permissions=child_permissions, expires_at=clock[0] + 100, parent_id=parent.id
    )
    grandchild = store.create(
        "loom:one",
        actor="iris:job",
        permissions=(Permission("iris", frozenset({"job.submit"}), "/alice/restricted/narrower/"),),
        expires_at=clock[0] + 100,
        parent_id=child.id,
    )
    grant_policy.denied_grant_resources.add("/alice/restricted/")
    assert store.active(parent.id).id == parent.id
    with pytest.raises(GrantDenied):
        store.active(child.id)
    with pytest.raises(GrantDenied):
        store.active(grandchild.id)


@pytest.mark.parametrize("change", [{"revision": "policy-v2"}, {"delegation_depth": 0}])
def test_reduced_standing_authority_invalidates_existing_grants(
    tmp_path, authorities, grant_permissions, grant_policy, clock, change
):
    path = tmp_path / "grants.db"
    database = GrantStore(path, authorities, grant_policy, now=lambda: clock[0])
    try:
        grant = database.create("user:alice", actor="loom:one", permissions=grant_permissions, expires_at=clock[0] + 300)
    finally:
        database.close()
    reopened = GrantStore(path, [replace(authorities[0], **change)], grant_policy, now=lambda: clock[0])
    try:
        with pytest.raises(GrantDenied):
            reopened.active(grant.id)
    finally:
        reopened.close()
