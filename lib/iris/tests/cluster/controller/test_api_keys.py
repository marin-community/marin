# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for API key management: DB CRUD, auth_setup preloading, and service RPCs."""

import secrets
from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth import (
    WORKER_USER,
    ControllerAuth,
    JwtTokenManager,
    create_api_key,
    create_controller_auth,
    list_api_keys,
)
from iris.rpc.auth import VerifiedIdentity, hash_token
from rigging.timing import Timestamp
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.store import ControllerStores
from iris.cluster.controller.transitions import ControllerTransitions
from iris.log_server.server import LogServiceImpl
from iris.rpc import config_pb2
from iris.rpc import job_pb2
from iris.rpc.auth import _verified_identity


@pytest.fixture
def db(tmp_path):
    d = ControllerDB(db_dir=tmp_path)
    yield d
    d.close()


def _make_service(db, auth=None):
    """Create a ControllerServiceImpl with minimal dependencies for API key tests."""
    stores = ControllerStores.from_db(db)
    state = ControllerTransitions(stores=stores)

    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.create_scheduling_context = Mock(return_value=Mock())
    controller_mock.get_job_scheduling_diagnostics = Mock(return_value="")
    controller_mock.autoscaler = None
    controller_mock.provider = Mock()
    controller_mock.has_direct_provider = False

    return ControllerServiceImpl(
        state,
        stores,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(db.db_path.parent / "bundles")),
        log_service=LogServiceImpl(),
        auth=auth or ControllerAuth(),
    )


# ---------------------------------------------------------------------------
# auth_setup: static preload
# ---------------------------------------------------------------------------


def test_static_preload_inserts_keys(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice", "tok-b": "bob"}})
    auth = create_controller_auth(config, db=db)
    assert auth.verifier is not None
    assert auth.provider == "static"

    # Static tokens are now only usable via Login RPC (exchanged for JWTs).
    # The login_verifier (StaticTokenVerifier) can authenticate raw static tokens.
    alice_identity = auth.login_verifier.verify("tok-a")
    assert alice_identity.user_id == "alice"

    bob_identity = auth.login_verifier.verify("tok-b")
    assert bob_identity.user_id == "bob"

    # The JWT verifier (JwtTokenManager) verifies worker token correctly.
    assert auth.worker_token is not None
    identity = auth.verifier.verify(auth.worker_token)
    assert identity.user_id == WORKER_USER


def test_static_preload_is_idempotent(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    create_controller_auth(config, db=db)
    create_controller_auth(config, db=db)  # Should not raise

    keys = list_api_keys(db, user_id="alice")
    assert len(keys) == 1


def test_worker_token_in_api_keys(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)

    # Worker token is a JWT — verify it resolves to WORKER_USER
    assert auth.worker_token is not None
    identity = auth.verifier.verify(auth.worker_token)
    assert identity.user_id == WORKER_USER


def test_worker_token_differs_after_restart(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth1 = create_controller_auth(config, db=db)
    token1 = auth1.worker_token

    auth2 = create_controller_auth(config, db=db)
    token2 = auth2.worker_token
    assert token1 != token2

    # Both tokens should still work (old not revoked) — using auth2's verifier
    # (same signing key, loaded revocations from DB)
    identity1 = auth2.verifier.verify(token1)
    identity2 = auth2.verifier.verify(token2)
    assert identity1.user_id == WORKER_USER
    assert identity2.user_id == WORKER_USER


def test_admin_users_bootstrapped(db):
    config = config_pb2.AuthConfig(
        static={"tokens": {"tok-a": "alice"}},
        admin_users=["alice"],
    )
    create_controller_auth(config, db=db)
    assert db.get_user_role("alice") == "admin"


def test_login_verifier_set_for_gcp(db):
    config = config_pb2.AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db)
    assert auth.login_verifier is not None
    assert auth.provider == "gcp"


def test_login_verifier_set_for_static(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)
    # Static auth now also sets login_verifier (StaticTokenVerifier)
    assert auth.login_verifier is not None
    assert auth.provider == "static"


# ---------------------------------------------------------------------------
# Service RPC: Login
# ---------------------------------------------------------------------------


def test_login_rejects_system_prefix(db):
    """Login RPC rejects usernames starting with system:."""

    class SystemVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="system:hacker", role="user")

    jwt_mgr = JwtTokenManager("test-signing-key")
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=SystemVerifier(),
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, auth=auth)

    with pytest.raises(ConnectError) as exc_info:
        service.login(job_pb2.LoginRequest(identity_token="fake"), None)
    assert exc_info.value.code == Code.PERMISSION_DENIED


def test_login_creates_user_and_key(db):
    """Login RPC creates a user and returns a working JWT."""

    class FakeVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="alice@example.com", role="user")

    jwt_mgr = JwtTokenManager("test-signing-key")
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=FakeVerifier(),
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, auth=auth)

    response = service.login(job_pb2.LoginRequest(identity_token="gcp-id-token"), None)
    assert response.user_id == "alice@example.com"
    assert response.token
    assert response.key_id.startswith("iris_k_")

    # The returned JWT should verify correctly
    identity = auth.verifier.verify(response.token)
    assert identity.user_id == "alice@example.com"


def test_login_is_idempotent(db):
    """Logging in twice revokes the first login key, leaving only one active."""

    class FakeVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="alice@example.com", role="user")

    jwt_mgr = JwtTokenManager("test-signing-key")
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=FakeVerifier(),
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, auth=auth)

    resp1 = service.login(job_pb2.LoginRequest(identity_token="tok1"), None)
    resp2 = service.login(job_pb2.LoginRequest(identity_token="tok2"), None)

    # First token should be revoked (JTI added to revocation set)
    with pytest.raises(ValueError, match="revoked"):
        auth.verifier.verify(resp1.token)
    identity = auth.verifier.verify(resp2.token)
    assert identity.user_id == "alice@example.com"

    # Only 1 active login key
    all_keys = list_api_keys(db, user_id="alice@example.com")
    active = [k for k in all_keys if k.revoked_at is None]
    assert len(active) == 1
    assert active[0].key_id == resp2.key_id


def test_login_not_available_without_login_verifier(db):
    """Login RPC returns UNIMPLEMENTED when no login_verifier is configured."""
    jwt_mgr = JwtTokenManager("test-signing-key")
    auth = ControllerAuth(verifier=jwt_mgr, provider="static", jwt_manager=jwt_mgr)
    service = _make_service(db, auth=auth)

    with pytest.raises(ConnectError) as exc_info:
        service.login(job_pb2.LoginRequest(identity_token="token"), None)
    assert exc_info.value.code == Code.UNIMPLEMENTED


# ---------------------------------------------------------------------------
# Service RPC: CreateApiKey, ListApiKeys, RevokeApiKey
# ---------------------------------------------------------------------------


def test_create_api_key_returns_raw_token(db):
    """CreateApiKey returns a working JWT token."""
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    token = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        response = service.create_api_key(
            job_pb2.CreateApiKeyRequest(name="my-key"),
            None,
        )
    finally:
        _verified_identity.reset(token)

    assert response.token
    assert response.key_id.startswith("iris_k_")
    assert response.key_prefix == response.key_id[:8]

    # JWT token should verify correctly
    identity = auth.verifier.verify(response.token)
    assert identity.user_id == "alice"


def test_list_api_keys_never_exposes_hash(db):
    """ListApiKeys returns key info without exposing hashes."""
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    token = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        response = service.list_api_keys(
            job_pb2.ListApiKeysRequest(user_id="alice"),
            None,
        )
    finally:
        _verified_identity.reset(token)

    assert len(response.keys) > 0
    for key_info in response.keys:
        assert key_info.key_prefix
        assert key_info.user_id == "alice"


def test_revoke_key_owner_only(db):
    """Non-admin user cannot revoke another user's key."""
    config = config_pb2.AuthConfig(
        static={"tokens": {"tok-a": "alice", "tok-b": "bob"}},
    )
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    # Get alice's key_id
    alice_keys = list_api_keys(db, user_id="alice")
    assert alice_keys

    # Bob tries to revoke alice's key
    token = _verified_identity.set(VerifiedIdentity(user_id="bob", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            service.revoke_api_key(
                job_pb2.RevokeApiKeyRequest(key_id=alice_keys[0].key_id),
                None,
            )
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(token)


def test_admin_can_revoke_any_key(db):
    """Admin user can revoke any user's key."""
    config = config_pb2.AuthConfig(
        static={"tokens": {"tok-a": "alice", "tok-b": "bob"}},
        admin_users=["bob"],
    )
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    alice_keys = list_api_keys(db, user_id="alice")
    assert alice_keys
    alice_key_id = alice_keys[0].key_id

    token = _verified_identity.set(VerifiedIdentity(user_id="bob", role="admin"))
    try:
        service.revoke_api_key(
            job_pb2.RevokeApiKeyRequest(key_id=alice_key_id),
            None,
        )
    finally:
        _verified_identity.reset(token)

    # Alice's key should be in the revocation set — verify via a fresh JWT for alice
    alice_jwt = auth.jwt_manager.create_token("alice", "user", alice_key_id)
    with pytest.raises(ValueError, match="revoked"):
        auth.verifier.verify(alice_jwt)


# ---------------------------------------------------------------------------
# Service RPC: GetAuthInfo
# ---------------------------------------------------------------------------


def test_get_auth_info_returns_gcp_provider(db):
    """GetAuthInfo returns provider and project_id for GCP auth."""
    config = config_pb2.AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    response = service.get_auth_info(job_pb2.GetAuthInfoRequest(), None)
    assert response.provider == "gcp"
    assert response.gcp_project_id == "test-project"


def test_get_auth_info_returns_static_provider(db):
    """GetAuthInfo returns provider=static with no project_id."""
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    response = service.get_auth_info(job_pb2.GetAuthInfoRequest(), None)
    assert response.provider == "static"
    assert response.gcp_project_id == ""


def test_get_auth_info_returns_empty_when_no_auth(db):
    """GetAuthInfo returns empty provider when auth is disabled."""
    config = config_pb2.AuthConfig()
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    response = service.get_auth_info(job_pb2.GetAuthInfoRequest(), None)
    assert response.provider == ""
    assert response.gcp_project_id == ""


def test_discovery_login_flow(db):
    """Full discovery login: GetAuthInfo → Login → returned token works.

    Simulates a client that has no config file and discovers the auth
    method from the controller before logging in.
    """

    class FakeVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="alice@example.com", role="user")

    jwt_mgr = JwtTokenManager("test-signing-key")
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=FakeVerifier(),
        gcp_project_id="test-project",
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, auth=auth)

    # Step 1: Client discovers auth method (unauthenticated)
    auth_info = service.get_auth_info(job_pb2.GetAuthInfoRequest(), None)
    assert auth_info.provider == "gcp"
    assert auth_info.gcp_project_id == "test-project"

    # Step 2: Client obtains an access token (mocked) and calls Login
    response = service.login(job_pb2.LoginRequest(identity_token="fake-access-token"), None)
    assert response.user_id == "alice@example.com"
    assert response.token
    assert response.key_id.startswith("iris_k_")

    # Step 3: Returned JWT works for subsequent authenticated RPCs
    identity = auth.verifier.verify(response.token)
    assert identity.user_id == "alice@example.com"


# ---------------------------------------------------------------------------
# Null-auth mode
# ---------------------------------------------------------------------------


def test_null_auth_creates_anonymous_admin_and_worker_token(db):
    """No auth config + DB bootstraps anonymous admin and generates worker token."""
    config = config_pb2.AuthConfig()
    auth = create_controller_auth(config, db=db)
    assert db.get_user_role("anonymous") == "admin"
    assert auth.verifier is not None
    assert auth.worker_token is not None
    assert auth.provider is None
    assert auth.login_verifier is None


def test_null_auth_rpcs_work_with_anonymous_token(db):
    """Auth RPCs work in null-auth mode via JwtTokenManager."""
    config = config_pb2.AuthConfig()
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    # Create an API key for "anonymous" to simulate authenticated access
    anonymous_token = secrets.token_urlsafe(32)
    create_api_key(
        db,
        key_id=f"iris_k_test_{secrets.token_hex(4)}",
        key_hash=hash_token(anonymous_token),
        key_prefix=anonymous_token[:8],
        user_id="anonymous",
        name="test-null-auth",
        now=Timestamp.now(),
    )

    # In null-auth mode the verifier is JwtTokenManager, so create a proper JWT
    key_id = f"iris_k_test_{secrets.token_hex(4)}"
    jwt_token = auth.jwt_manager.create_token("anonymous", "admin", key_id)
    verified = auth.verifier.verify(jwt_token)
    reset = _verified_identity.set(verified)
    try:
        keys_resp = service.list_api_keys(job_pb2.ListApiKeysRequest(), None)
        assert keys_resp is not None

        create_resp = service.create_api_key(
            job_pb2.CreateApiKeyRequest(name="test-key"),
            None,
        )
        assert create_resp.token
        assert create_resp.key_id.startswith("iris_k_")
    finally:
        _verified_identity.reset(reset)


def test_null_auth_get_current_user(db):
    """GetCurrentUser returns anonymous/admin in null-auth mode."""
    config = config_pb2.AuthConfig()
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    # Create a JWT for anonymous/admin
    key_id = f"iris_k_test_{secrets.token_hex(4)}"
    jwt_token = auth.jwt_manager.create_token("anonymous", "admin", key_id)
    verified = auth.verifier.verify(jwt_token)
    reset = _verified_identity.set(verified)
    try:
        resp = service.get_current_user(job_pb2.GetCurrentUserRequest(), None)
        assert resp.user_id == "anonymous"
        assert resp.role == "admin"
    finally:
        _verified_identity.reset(reset)
