# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time

import jwt
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller import writes
from iris.cluster.controller.auth import (
    CONTROL_PLANE_AUDIENCE,
    CONTROL_PLANE_AUDIENCES,
    JwtTokenManager,
    create_api_key,
    revoke_api_key,
)
from iris.cluster.controller.db import ControllerDB
from iris.rpc.auth import DASHBOARD_ROLE, AuthzAction, authorize, authorize_method, authorize_resource_owner
from rigging.server_auth import VerifiedIdentity, _verified_identity
from rigging.timing import Timestamp
from rigging.token_authority import (
    Ed25519Keypair,
    JwksVerifier,
    JwtSigner,
    generate_ed25519_keypair,
    signing_key_from_private_pem,
)

_ISSUER = "test-cluster"


def _manager(*, keypair: Ed25519Keypair | None = None, db: ControllerDB | None = None) -> JwtTokenManager:
    """Build a JwtTokenManager over a real EdDSA keypair (no mocking the signer)."""
    keypair = keypair or generate_ed25519_keypair()
    key = signing_key_from_private_pem(keypair.private_pem)
    signer = JwtSigner(key, issuer=_ISSUER)
    verifier = JwksVerifier(issuers={_ISSUER: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES)
    return JwtTokenManager(signer, verifier, db=db)


# --- read-only dashboard role: per-method authorization ----------------------


@pytest.mark.parametrize("method", ["ListJobs", "GetJobStatus", "ListWorkers", "GetRpcStats", "ListPeers"])
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
    return _manager()


def test_jwt_token_manager_roundtrip(jwt_manager):
    token = jwt_manager.create_token(user_id="alice", role="user", key_id="k1")
    identity = jwt_manager.verify(token)
    assert identity.user_id == "alice"
    assert identity.role == "user"


def test_jwt_token_manager_rejects_wrong_key():
    # Same issuer, different keypairs: manager_b resolves a's iss to its own key
    # and the EdDSA signature fails to verify.
    manager_a = _manager()
    manager_b = _manager()
    token = manager_a.create_token(user_id="alice", role="user", key_id="k1")
    with pytest.raises(ValueError, match="signature"):
        manager_b.verify(token)


def test_jwt_token_manager_revocation(jwt_manager):
    token = jwt_manager.create_token(user_id="alice", role="user", key_id="revoke-me")
    jwt_manager.revoke("revoke-me")
    with pytest.raises(ValueError, match="revoked"):
        jwt_manager.verify(token)


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


def test_jwt_token_manager_load_revocations(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    now = Timestamp.now()
    with db.transaction() as _tx:
        writes.ensure_user(_tx, "alice", now)

    manager = _manager(db=db)

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
