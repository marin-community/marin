# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller import writes
from iris.cluster.controller.auth import JwtTokenManager, create_api_key, revoke_api_key
from iris.cluster.controller.db import ControllerDB
from iris.rpc.auth import DASHBOARD_ROLE, AuthzAction, authorize, authorize_method, authorize_resource_owner
from rigging.server_auth import VerifiedIdentity, _verified_identity
from rigging.timing import Timestamp

# --- read-only dashboard role: per-method authorization ----------------------


@pytest.mark.parametrize(
    "method", ["ListJobs", "GetJobStatus", "ListWorkers", "GetRpcStats", "ListPeers", "GetClusterCapabilities"]
)
def test_authorize_method_allows_dashboard_reads(method):
    # Does not raise: read methods are the dashboard role's contract.
    authorize_method(VerifiedIdentity("alice@example.com", DASHBOARD_ROLE), method)


@pytest.mark.parametrize(
    "method",
    ["LaunchJob", "TerminateJob", "CreateApiKey", "ExecInContainer", "SetUserBudget", "ExecuteRawQuery"],
)
def test_authorize_method_denies_dashboard_mutations(method):
    with pytest.raises(ConnectError) as exc:
        authorize_method(VerifiedIdentity("alice@example.com", DASHBOARD_ROLE), method)
    assert exc.value.code == Code.PERMISSION_DENIED


@pytest.mark.parametrize("role", ["admin", "user", "worker"])
def test_authorize_method_unrestricted_for_other_roles(role):
    # Non-dashboard roles are not gated by method name here; their mutating
    # actions are still checked inside the handlers by authorize/owner checks.
    authorize_method(VerifiedIdentity("alice", role), "LaunchJob")


# ---------------------------------------------------------------------------
# JwtTokenManager (replaces DbTokenVerifier)
# ---------------------------------------------------------------------------


@pytest.fixture
def jwt_manager():
    return JwtTokenManager(signing_key="test-signing-key-abcdef1234567890")


def test_jwt_token_manager_roundtrip(jwt_manager):
    token = jwt_manager.create_token(user_id="alice", role="user", key_id="k1")
    identity = jwt_manager.verify(token)
    assert identity.user_id == "alice"
    assert identity.role == "user"


def test_jwt_token_manager_rejects_wrong_key():
    manager_a = JwtTokenManager(signing_key="key-a-abcdef1234567890abcdef")
    manager_b = JwtTokenManager(signing_key="key-b-abcdef1234567890abcdef")
    token = manager_a.create_token(user_id="alice", role="user", key_id="k1")
    with pytest.raises(ValueError, match="Invalid token"):
        manager_b.verify(token)


def test_jwt_token_manager_revocation(jwt_manager):
    token = jwt_manager.create_token(user_id="alice", role="user", key_id="revoke-me")
    jwt_manager.revoke("revoke-me")
    with pytest.raises(ValueError, match="revoked"):
        jwt_manager.verify(token)


def test_jwt_token_manager_expired(jwt_manager):
    token = jwt_manager.create_token(user_id="alice", role="user", key_id="k-exp", ttl_seconds=-1)
    with pytest.raises(ValueError, match="expired"):
        jwt_manager.verify(token)


def test_jwt_token_manager_load_revocations(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    now = Timestamp.now()
    with db.transaction() as _tx:
        writes.ensure_user(_tx, "alice", now)

    manager = JwtTokenManager(signing_key="test-key-load-revocations-abc123")

    # Insert a key and revoke it in the DB
    create_api_key(
        db,
        key_id="k-revoked",
        key_prefix="jwt",
        user_id="alice",
        name="test-key",
        now=now,
    )
    revoke_api_key(db, "k-revoked", now)

    # Also insert an active key
    create_api_key(
        db,
        key_id="k-active",
        key_prefix="jwt",
        user_id="alice",
        name="active-key",
        now=now,
    )

    manager.load_revocations(db)

    revoked_token = manager.create_token(user_id="alice", role="user", key_id="k-revoked")
    active_token = manager.create_token(user_id="alice", role="user", key_id="k-active")

    with pytest.raises(ValueError, match="revoked"):
        manager.verify(revoked_token)

    identity = manager.verify(active_token)
    assert identity.user_id == "alice"

    db.close()


def test_jwt_token_manager_worker_role(jwt_manager):
    token = jwt_manager.create_token(user_id="system:worker", role="worker", key_id="w1")
    identity = jwt_manager.verify(token)
    assert identity.user_id == "system:worker"
    assert identity.role == "worker"


# ---------------------------------------------------------------------------
# Centralized authorization (authorize / authorize_resource_owner)
# ---------------------------------------------------------------------------


def test_authorize_admin_always_passes():
    reset = _verified_identity.set(VerifiedIdentity(user_id="admin-user", role="admin"))
    try:
        # Admin should pass any action, even ACT_AS_WORKER
        identity = authorize(AuthzAction.ACT_AS_WORKER)
        assert identity.user_id == "admin-user"
    finally:
        _verified_identity.reset(reset)


def test_authorize_worker_can_act_as_worker():
    reset = _verified_identity.set(VerifiedIdentity(user_id="system:worker", role="worker"))
    try:
        identity = authorize(AuthzAction.ACT_AS_WORKER)
        assert identity.role == "worker"
    finally:
        _verified_identity.reset(reset)


def test_authorize_user_cannot_act_as_worker():
    reset = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            authorize(AuthzAction.ACT_AS_WORKER)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(reset)


def test_authorize_raises_unauthenticated_when_no_identity():
    # No identity set — should raise UNAUTHENTICATED
    with pytest.raises(ConnectError) as exc_info:
        authorize(AuthzAction.ACT_AS_WORKER)
    assert exc_info.value.code == Code.UNAUTHENTICATED


def test_authorize_manage_other_keys_admin_only():
    reset = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(reset)


def test_authorize_resource_owner_same_user():
    reset = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        identity = authorize_resource_owner("alice")
        assert identity.user_id == "alice"
    finally:
        _verified_identity.reset(reset)


def test_authorize_resource_owner_different_user_denied():
    reset = _verified_identity.set(VerifiedIdentity(user_id="bob", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            authorize_resource_owner("alice")
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(reset)


def test_authorize_resource_owner_admin_can_access_any():
    reset = _verified_identity.set(VerifiedIdentity(user_id="admin-user", role="admin"))
    try:
        identity = authorize_resource_owner("alice")
        assert identity.user_id == "admin-user"
    finally:
        _verified_identity.reset(reset)
