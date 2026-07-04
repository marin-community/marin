# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for API key management: DB CRUD, auth_setup preloading, and service RPCs."""

import secrets
from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import AuthConfig, GcpAuthConfig, IapAuthConfig
from iris.cluster.controller import reads
from iris.cluster.controller.auth import (
    CONTROL_PLANE_AUDIENCES,
    WORKER_USER,
    ControllerAuth,
    JwtTokenManager,
    create_api_key,
    create_controller_auth,
    list_api_keys,
    require_persistent_signing_key,
)
from iris.cluster.controller.backend import BackendCapability
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.service import ControllerServiceImpl
from iris.rpc import job_pb2
from rigging.server_auth import IapIdTokenVerifier, VerifiedIdentity, _verified_identity
from rigging.timing import Timestamp
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem

_CLUSTER = "test-cluster"


def _jwt_manager(db: ControllerDB | None = None) -> JwtTokenManager:
    """A JwtTokenManager over a real EdDSA keypair (issuer ``_CLUSTER``)."""
    key = signing_key_from_private_pem(generate_ed25519_keypair().private_pem)
    signer = JwtSigner(key, issuer=_CLUSTER)
    verifier = JwksVerifier(issuers={_CLUSTER: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES)
    return JwtTokenManager(signer, verifier, db=db)


# A persistent signing key shared by create_controller_auth calls in this module —
# realistic for an authed cluster (what a real deploy sources from a SecretSpec).
# create_controller_auth itself also accepts None (ephemeral, in-process dev); the
# persistent-key requirement is enforced at the serve boundary by
# require_persistent_signing_key, tested in test_require_persistent_signing_key.
_SIGNING_KEY = generate_ed25519_keypair().private_pem


def _seed_user_key(db: ControllerDB, user_id: str) -> str:
    """Insert a login api_key row for ``user_id`` and return its key_id.

    An authed controller no longer preloads per-user keys (the static-token
    preload was removed); tests that need an existing user key create one here.
    """
    key_id = f"iris_k_{user_id}_{secrets.token_hex(4)}"
    create_api_key(
        db, key_id=key_id, key_prefix=key_id[:8], user_id=user_id, name=f"login-{user_id}", now=Timestamp.now()
    )
    return key_id


def test_require_persistent_signing_key():
    # A gcp/iap login provider issues user JWTs; an empty signing key is a silent
    # trust-anchor break waiting to happen, so a deployed one must fail fast.
    for provider in (AuthConfig(gcp=GcpAuthConfig(project_id="p")), AuthConfig(iap=IapAuthConfig())):
        with pytest.raises(ValueError, match="requires a persistent"):
            require_persistent_signing_key(provider, None)
        require_persistent_signing_key(provider, _SIGNING_KEY)  # a key present: fine

    # CIDR-only trust and null-auth have no login provider — they mint only internal
    # worker tokens, so an ephemeral key is acceptable and must NOT be rejected.
    require_persistent_signing_key(AuthConfig(trusted_cidrs=["10.0.0.0/8"]), None)
    require_persistent_signing_key(AuthConfig(), None)
    require_persistent_signing_key(None, None)


@pytest.fixture
def db(tmp_path):
    d = ControllerDB(db_dir=tmp_path)
    yield d
    d.close()


def _make_service(db, log_client, auth=None):
    """Create a ControllerServiceImpl with minimal dependencies for API key tests."""
    endpoints = EndpointsProjection(db)

    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.get_job_scheduling_diagnostics = Mock(return_value="")
    controller_mock.last_scheduling_context = None
    controller_mock.autoscaler = None
    controller_mock.provider = Mock()
    controller_mock.capabilities = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})

    return ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(db.db_path.parent / "bundles")),
        log_client=log_client,
        db=db,
        endpoints=endpoints,
        auth=auth or ControllerAuth(),
        endpoint_service=EndpointServiceImpl(db=db, endpoints=endpoints),
    )


# ---------------------------------------------------------------------------
# auth_setup: authed providers
# ---------------------------------------------------------------------------


def test_worker_token_in_api_keys(db):
    config = AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)

    # Worker token is a JWT — verify it resolves to WORKER_USER
    assert auth.worker_token is not None
    identity = auth.verifier.verify(auth.worker_token)
    assert identity.user_id == WORKER_USER


def test_iap_provider_uses_id_token_login_verifier(db):
    config = AuthConfig(iap={"audiences": ["desktop-client-id"]})
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)

    # IAP login proof is an OIDC ID token verified by IapIdTokenVerifier.
    assert isinstance(auth.login_verifier, IapIdTokenVerifier)
    # Per-RPC requests still ride on Iris JWTs (the worker token verifies).
    assert auth.verifier.verify(auth.worker_token).user_id == WORKER_USER


def test_iap_provider_requires_audiences_or_assertion(db):
    config = AuthConfig(iap={"url": "https://iris-marin.example.com"})
    with pytest.raises(ValueError, match=r"audiences .* signed_header_audience"):
        create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)


def test_worker_token_differs_after_restart(db):
    # A persistent signing key (as a real deploy sources from a SecretSpec) is
    # shared across restarts, so a token minted before the restart still verifies
    # after it — but each start mints a fresh worker jti, so the tokens differ.
    signing_key_pem = generate_ed25519_keypair().private_pem
    config = AuthConfig(gcp={"project_id": "test-project"})
    auth1 = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=signing_key_pem)
    token1 = auth1.worker_token

    auth2 = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=signing_key_pem)
    token2 = auth2.worker_token
    assert token1 != token2

    # Both tokens should still work (old not revoked) — using auth2's verifier
    # (same signing key, loaded revocations from DB)
    identity1 = auth2.verifier.verify(token1)
    identity2 = auth2.verifier.verify(token2)
    assert identity1.user_id == WORKER_USER
    assert identity2.user_id == WORKER_USER


def test_admin_users_bootstrapped(db):
    config = AuthConfig(
        gcp={"project_id": "test-project"},
        admin_users=["alice"],
    )
    create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    with db.read_snapshot() as snap:
        assert reads.get_user_role(snap, "alice") == "admin"


def test_login_verifier_set_for_gcp(db):
    config = AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    assert auth.login_verifier is not None
    assert auth.provider == "gcp"


# ---------------------------------------------------------------------------
# Service RPC: Login
# ---------------------------------------------------------------------------


def test_login_rejects_system_prefix(db, log_client):
    """Login RPC rejects usernames starting with system:."""

    class SystemVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="system:hacker", role="user")

    jwt_mgr = _jwt_manager()
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=SystemVerifier(),
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, log_client, auth=auth)

    with pytest.raises(ConnectError) as exc_info:
        service.login(job_pb2.LoginRequest(identity_token="fake"), None)
    assert exc_info.value.code == Code.PERMISSION_DENIED


def test_login_creates_user_and_key(db, log_client):
    """Login RPC creates a user and returns a working JWT."""

    class FakeVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="alice@example.com", role="user")

    jwt_mgr = _jwt_manager()
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=FakeVerifier(),
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, log_client, auth=auth)

    response = service.login(job_pb2.LoginRequest(identity_token="gcp-id-token"), None)
    assert response.user_id == "alice@example.com"
    assert response.token
    assert response.key_id.startswith("iris_k_")

    # The returned JWT should verify correctly
    identity = auth.verifier.verify(response.token)
    assert identity.user_id == "alice@example.com"


def test_login_is_idempotent(db, log_client):
    """Logging in twice revokes the first login key, leaving only one active."""

    class FakeVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="alice@example.com", role="user")

    jwt_mgr = _jwt_manager()
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=FakeVerifier(),
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, log_client, auth=auth)

    resp1 = service.login(job_pb2.LoginRequest(identity_token="tok1"), None)
    resp2 = service.login(job_pb2.LoginRequest(identity_token="tok2"), None)

    # First token should be revoked (JTI added to revocation set)
    with pytest.raises(ValueError, match="revoked"):
        auth.verifier.verify(resp1.token)
    identity = auth.verifier.verify(resp2.token)
    assert identity.user_id == "alice@example.com"

    # Only 1 active login key
    all_keys = list_api_keys(db, user_id="alice@example.com")
    active = [k for k in all_keys if k.revoked_at_ms is None]
    assert len(active) == 1
    assert active[0].key_id == resp2.key_id


def test_login_not_available_without_login_verifier(db, log_client):
    """Login RPC returns UNIMPLEMENTED when no login_verifier is configured."""
    jwt_mgr = _jwt_manager()
    auth = ControllerAuth(verifier=jwt_mgr, provider="gcp", jwt_manager=jwt_mgr)
    service = _make_service(db, log_client, auth=auth)

    with pytest.raises(ConnectError) as exc_info:
        service.login(job_pb2.LoginRequest(identity_token="token"), None)
    assert exc_info.value.code == Code.UNIMPLEMENTED


# ---------------------------------------------------------------------------
# Service RPC: CreateApiKey, ListApiKeys, RevokeApiKey
# ---------------------------------------------------------------------------


def test_create_api_key_returns_raw_token(db, log_client):
    """CreateApiKey returns a working JWT token."""
    config = AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    service = _make_service(db, log_client, auth=auth)

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


def test_list_api_keys_never_exposes_hash(db, log_client):
    """ListApiKeys returns key info without exposing hashes."""
    config = AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    _seed_user_key(db, "alice")
    service = _make_service(db, log_client, auth=auth)

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


def test_revoke_key_owner_only(db, log_client):
    """Non-admin user cannot revoke another user's key."""
    config = AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    _seed_user_key(db, "alice")
    _seed_user_key(db, "bob")
    service = _make_service(db, log_client, auth=auth)

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


def test_admin_can_revoke_any_key(db, log_client):
    """Admin user can revoke any user's key."""
    config = AuthConfig(
        gcp={"project_id": "test-project"},
        admin_users=["bob"],
    )
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    _seed_user_key(db, "alice")
    service = _make_service(db, log_client, auth=auth)

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


def test_get_auth_info_returns_gcp_provider(db, log_client):
    """GetAuthInfo returns provider and project_id for GCP auth."""
    config = AuthConfig(gcp={"project_id": "test-project"})
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    service = _make_service(db, log_client, auth=auth)

    response = service.get_auth_info(job_pb2.GetAuthInfoRequest(), None)
    assert response.provider == "gcp"
    assert response.gcp_project_id == "test-project"


def test_get_auth_info_returns_empty_when_no_auth(db, log_client):
    """GetAuthInfo returns empty provider when auth is disabled."""
    config = AuthConfig()
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER)
    service = _make_service(db, log_client, auth=auth)

    response = service.get_auth_info(job_pb2.GetAuthInfoRequest(), None)
    assert response.provider == ""
    assert response.gcp_project_id == ""


def test_discovery_login_flow(db, log_client):
    """Full discovery login: GetAuthInfo → Login → returned token works.

    Simulates a client that has no config file and discovers the auth
    method from the controller before logging in.
    """

    class FakeVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(user_id="alice@example.com", role="user")

    jwt_mgr = _jwt_manager()
    auth = ControllerAuth(
        verifier=jwt_mgr,
        provider="gcp",
        login_verifier=FakeVerifier(),
        gcp_project_id="test-project",
        jwt_manager=jwt_mgr,
    )
    service = _make_service(db, log_client, auth=auth)

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
    config = AuthConfig()
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER)
    with db.read_snapshot() as snap:
        assert reads.get_user_role(snap, "anonymous") == "admin"
    assert auth.verifier is not None
    assert auth.worker_token is not None
    assert auth.provider is None
    assert auth.login_verifier is None


def test_null_auth_rpcs_work_with_anonymous_token(db, log_client):
    """Auth RPCs work in null-auth mode via JwtTokenManager."""
    config = AuthConfig()
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER)
    service = _make_service(db, log_client, auth=auth)

    # Create an API key for "anonymous" to simulate authenticated access
    anonymous_token = secrets.token_urlsafe(32)
    create_api_key(
        db,
        key_id=f"iris_k_test_{secrets.token_hex(4)}",
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


def test_null_auth_get_current_user(db, log_client):
    """GetCurrentUser returns anonymous/admin in null-auth mode."""
    config = AuthConfig()
    auth = create_controller_auth(config, db=db, cluster_name=_CLUSTER)
    service = _make_service(db, log_client, auth=auth)

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
