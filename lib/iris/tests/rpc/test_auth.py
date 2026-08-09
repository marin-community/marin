# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time

import jwt
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.authorization import authorize_resource_owner
from iris.cluster.controller.auth import (
    CONTROL_PLANE_AUDIENCE,
    CONTROL_PLANE_AUDIENCES,
    JwtTokenManager,
)
from iris.resources.errors import ResourcePermissionDenied
from iris.rpc.auth import (
    DASHBOARD_ROLE,
    FEDERATION_PEER_ROLE,
    AuthzAction,
    authorize,
    authorize_method,
)
from rigging.server_auth import VerifiedIdentity, identity_scope
from rigging.token_authority import (
    Ed25519Keypair,
    JwksVerifier,
    JwtSigner,
    generate_ed25519_keypair,
    signing_key_from_private_pem,
)

_ISSUER = "test-cluster"


def _manager(*, keypair: Ed25519Keypair | None = None) -> JwtTokenManager:
    """Build a JwtTokenManager over a real EdDSA keypair (no mocking the signer)."""
    keypair = keypair or generate_ed25519_keypair()
    key = signing_key_from_private_pem(keypair.private_pem)
    signer = JwtSigner(key, issuer=_ISSUER)
    verifier = JwksVerifier(issuers={_ISSUER: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES)
    return JwtTokenManager(signer, verifier)


# --- read-only dashboard role: per-method authorization ----------------------


@pytest.mark.parametrize(
    "method",
    ["ListJobs", "GetJobStatus", "BatchDescribeTasks", "BatchDescribeEndpoints", "ListWorkers", "ListPeers"],
)
def test_authorize_method_allows_dashboard_reads(method):
    # Does not raise: read methods are the dashboard role's contract.
    authorize_method(VerifiedIdentity("alice@example.com", DASHBOARD_ROLE), method)


@pytest.mark.parametrize(
    "method",
    ["LaunchJob", "TerminateJob", "ExecInContainer", "SetUserBudget", "ExecuteRawQuery"],
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


# --- federation-peer role: method-scoped to the federation RPC subset ---------


def test_authorize_method_allows_federation_sync_for_peer():
    # Does not raise: sync is the representative federation control-plane RPC.
    authorize_method(VerifiedIdentity("peer-cluster", FEDERATION_PEER_ROLE), "FederationSync")


@pytest.mark.parametrize(
    "method",
    ["CancelJob", "RetryTask", "TerminateAttempt", "ExecAttempt", "ProfileAttempt", "GetProcessStatus"],
)
def test_authorize_method_allows_scoped_resource_operations_for_peer(method):
    # The resource handler then scopes each operation to a Job the peer federated here.
    authorize_method(VerifiedIdentity("peer-cluster", FEDERATION_PEER_ROLE), method)


@pytest.mark.parametrize("method", ["SetUserBudget", "ListJobs", "GetJobStatus", "ExecuteRawQuery"])
def test_authorize_method_denies_non_federation_rpcs_for_a_peer(method):
    # A federation bearer accepted by the composite verifier cannot reach any RPC
    # outside the federation subset and the scoped debug proxies — including every
    # read the dashboard role would be allowed.
    with pytest.raises(ConnectError) as exc:
        authorize_method(VerifiedIdentity("peer-cluster", FEDERATION_PEER_ROLE), method)
    assert exc.value.code == Code.PERMISSION_DENIED


# ---------------------------------------------------------------------------
# JwtTokenManager (replaces DbTokenVerifier)
# ---------------------------------------------------------------------------


@pytest.fixture
def jwt_manager():
    return _manager()


def test_jwt_token_manager_roundtrip(jwt_manager):
    token = jwt_manager.create_token(user_id="alice", role="user", key_id="k1", ttl_seconds=60)
    identity = jwt_manager.verify(token)
    assert identity.user_id == "alice"
    assert identity.role == "user"


def test_jwt_token_manager_rejects_wrong_key():
    # Same issuer, different keypairs: manager_b resolves a's iss to its own key
    # and the EdDSA signature fails to verify.
    manager_a = _manager()
    manager_b = _manager()
    token = manager_a.create_token(user_id="alice", role="user", key_id="k1", ttl_seconds=60)
    with pytest.raises(ValueError, match="signature"):
        manager_b.verify(token)


def test_jwt_token_manager_expired():
    # mint() forbids a non-positive ttl, so hand-sign an already-expired token
    # with the same key to exercise the verifier's exp check.
    keypair = generate_ed25519_keypair()
    manager = _manager(keypair=keypair)
    now = int(time.time())
    expired = jwt.encode(
        {
            "sub": "alice",
            "role": "user",
            "jti": "k-exp",
            "iss": _ISSUER,
            "aud": CONTROL_PLANE_AUDIENCE,
            "iat": now - 3600,
            "exp": now - 1800,
        },
        keypair.private_pem,
        algorithm="EdDSA",
        headers={"kid": keypair.kid},
    )
    with pytest.raises(ValueError, match="expired"):
        manager.verify(expired)


def test_jwt_token_manager_worker_role(jwt_manager):
    token = jwt_manager.create_token(user_id="system:worker", role="worker", key_id="w1", ttl_seconds=60)
    identity = jwt_manager.verify(token)
    assert identity.user_id == "system:worker"
    assert identity.role == "worker"
    # Workers resolve the controller-owned log endpoint during startup; the
    # authenticated resource reads are therefore part of the worker-token contract.
    authorize_method(identity, "ListEndpoints")
    authorize_method(identity, "BatchDescribeEndpoints")


# ---------------------------------------------------------------------------
# Centralized authorization (authorize / authorize_resource_owner)
# ---------------------------------------------------------------------------


def test_authorize_admin_always_passes():
    with identity_scope(VerifiedIdentity(user_id="admin-user", role="admin")):
        identity = authorize(AuthzAction.ACT_AS_WORKER)
        assert identity.user_id == "admin-user"


def test_authorize_worker_can_act_as_worker():
    with identity_scope(VerifiedIdentity(user_id="system:worker", role="worker")):
        identity = authorize(AuthzAction.ACT_AS_WORKER)
        assert identity.role == "worker"


def test_authorize_user_cannot_act_as_worker():
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        with pytest.raises(ConnectError) as exc_info:
            authorize(AuthzAction.ACT_AS_WORKER)
        assert exc_info.value.code == Code.PERMISSION_DENIED


def test_authorize_raises_unauthenticated_when_no_identity():
    # No identity set — should raise UNAUTHENTICATED
    with pytest.raises(ConnectError) as exc_info:
        authorize(AuthzAction.ACT_AS_WORKER)
    assert exc_info.value.code == Code.UNAUTHENTICATED


def test_authorize_resource_owner_same_user():
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        identity = authorize_resource_owner("alice")
        assert identity.user_id == "alice"


def test_authorize_resource_owner_different_user_denied():
    with identity_scope(VerifiedIdentity(user_id="bob", role="user")):
        with pytest.raises(ResourcePermissionDenied, match="cannot access resources owned by 'alice'"):
            authorize_resource_owner("alice")


def test_authorize_resource_owner_admin_can_access_any():
    with identity_scope(VerifiedIdentity(user_id="admin-user", role="admin")):
        identity = authorize_resource_owner("alice")
        assert identity.user_id == "admin-user"
