# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth: session cookies, CSRF, default-deny middleware, auth DB isolation,
stateless JWT, controller auth setup, and null-auth mode."""

from unittest.mock import Mock

import jwt
import pytest
import sqlalchemy.exc
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import AuthConfig, IapAuthConfig, PeerConfig
from iris.cluster.controller.auth import (
    _LEGACY_ISSUER,
    CONTROL_PLANE_AUDIENCES,
    FEDERATION_AUDIENCE,
    FEDERATION_PEER_ROLE,
    SESSION_TOKEN_TTL_SECONDS,
    WORKER_TOKEN_TTL_SECONDS,
    WORKER_USER,
    ControllerAuth,
    FederationTokenVerifier,
    JwtTokenManager,
    RolePolicy,
    _build_jwt_token_manager,
    _ControlPlaneOrFederationVerifier,
    create_controller_auth,
    request_auth_policy,
    require_persistent_signing_key,
)
from iris.cluster.controller.backend import BackendCapability
from iris.cluster.controller.dashboard import (
    _UNAUTHENTICATED_RPCS,
    ControllerDashboard,
    _SubdomainProxyMiddleware,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import DEFAULT_BACKEND_ID
from iris.rpc import job_pb2
from iris.rpc.auth import DASHBOARD_ROLE, SESSION_COOKIE, authorize_method
from rigging.server_auth import (
    PolicyAuthInterceptor,
    RequestAuthPolicy,
    RouteAuthMiddleware,
    VerifiedIdentity,
    _verified_identity,
    get_verified_identity,
    requires_auth,
)
from rigging.testing import MockVerifier
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem
from sqlalchemy import text
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from tests.cluster.controller._test_support import ControllerTestState

_TEST_TOKEN = "valid-test-token"
_TEST_USER = "test-user"
_CLUSTER = "test-cluster"
CSRF_HEADERS = {"Origin": "http://testserver"}

# A persistent signing key for authed create_controller_auth calls: an authed
# cluster requires one (the ephemeral fallback is null-auth only).
_SIGNING_KEYPAIR = generate_ed25519_keypair()
_SIGNING_KEY = _SIGNING_KEYPAIR.private_pem
_PUBLIC_KEY = _SIGNING_KEYPAIR.public_pem


def _jwt_manager() -> JwtTokenManager:
    """A JwtTokenManager over a real EdDSA keypair (issuer ``_CLUSTER``)."""
    key = signing_key_from_private_pem(generate_ed25519_keypair().private_pem)
    signer = JwtSigner(key, issuer=_CLUSTER)
    verifier = JwksVerifier(issuers={_CLUSTER: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES)
    return JwtTokenManager(signer, verifier)


def _make_service(db, log_client, auth=None):
    """A ControllerServiceImpl with minimal deps for login / auth-setup tests."""
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
        auth=auth or ControllerAuth(),
        endpoint_service=EndpointServiceImpl(db=db),
    )


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    yield db
    db.close()


@pytest.fixture
def state(db, tmp_path):
    s = ControllerTestState(db)
    yield s


@pytest.fixture
def service(state, tmp_path, log_client):
    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.autoscaler = None
    worker_caps = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
    controller_mock.provider = Mock(capabilities=worker_caps)
    controller_mock.provider.name = "worker"
    controller_mock.capabilities = worker_caps
    controller_mock.backends = {DEFAULT_BACKEND_ID: controller_mock.provider}
    return ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )


@pytest.fixture
def verifier():
    return MockVerifier({_TEST_TOKEN: _TEST_USER})


@pytest.fixture
def authed_client(service, verifier):
    dashboard = ControllerDashboard(
        service,
        auth_provider="iap",
        auth_policy=RequestAuthPolicy.enforcing(verifier=verifier),
    )
    return TestClient(dashboard.app)


@pytest.fixture
def noauth_client(service):
    dashboard = ControllerDashboard(service)
    return TestClient(dashboard.app)


# -- Token verification -------------------------------------------------------


def test_auth_session_rejects_invalid_token(authed_client):
    resp = authed_client.post("/auth/session", json={"token": "bad-token"}, headers=CSRF_HEADERS)
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid token"


def test_auth_session_accepts_valid_token(authed_client):
    resp = authed_client.post("/auth/session", json={"token": _TEST_TOKEN}, headers=CSRF_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "iris_session" in resp.cookies


def test_auth_session_returns_400_for_empty_token(authed_client):
    resp = authed_client.post("/auth/session", json={"token": "  "}, headers=CSRF_HEADERS)
    assert resp.status_code == 400


def test_auth_session_skips_verification_when_auth_disabled(noauth_client):
    resp = noauth_client.post("/auth/session", json={"token": "any-token-works"}, headers=CSRF_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# -- CSRF protection ----------------------------------------------------------


@pytest.mark.parametrize(
    "headers, expected_status",
    [
        ({"Origin": "http://evil.example.com"}, 403),
        ({}, 403),  # no Origin or Referer
        ({"Origin": "http://testserver"}, 200),
        ({"Referer": "http://testserver/auth/login"}, 200),
    ],
    ids=["mismatched-origin", "missing-origin-and-referer", "matching-origin", "matching-referer"],
)
def test_csrf_on_session_endpoint(authed_client, headers, expected_status):
    resp = authed_client.post("/auth/session", json={"token": _TEST_TOKEN}, headers=headers)
    assert resp.status_code == expected_status


def test_csrf_on_logout_rejects_missing_origin(authed_client):
    assert authed_client.post("/auth/logout").status_code == 403


def test_csrf_on_logout_accepts_matching_origin(authed_client):
    assert authed_client.post("/auth/logout", headers=CSRF_HEADERS).status_code == 200


def test_csrf_accepts_x_forwarded_host(authed_client):
    """CSRF check should use X-Forwarded-Host when behind a reverse proxy."""
    resp = authed_client.post(
        "/auth/session",
        json={"token": _TEST_TOKEN},
        headers={
            "Origin": "https://proxy.example.com",
            "X-Forwarded-Host": "proxy.example.com",
            "X-Forwarded-Proto": "https",
        },
    )
    assert resp.status_code == 200


def test_csrf_rejects_wrong_x_forwarded_host(authed_client):
    """CSRF check should reject when Origin doesn't match X-Forwarded-Host."""
    resp = authed_client.post(
        "/auth/session",
        json={"token": _TEST_TOKEN},
        headers={
            "Origin": "https://evil.example.com",
            "X-Forwarded-Host": "proxy.example.com",
            "X-Forwarded-Proto": "https",
        },
    )
    assert resp.status_code == 403


# -- Per-route auth policy -----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/", "/job/123", "/worker/456", "/bundles/" + "a" * 64 + ".zip", "/health", "/auth/config"],
    ids=["dashboard-root", "job-page", "worker-page", "bundle-download", "health", "auth-config"],
)
def test_public_route_accessible_without_auth(authed_client, path):
    """All @public routes serve content without a session cookie."""
    resp = authed_client.get(path)
    assert resp.status_code != 401


def test_auth_config_reports_enabled(authed_client):
    assert authed_client.get("/auth/config").json()["auth_enabled"] is True


def test_static_accessible_without_auth(authed_client):
    # Static mount may 404 (no actual files), but should NOT 401
    assert authed_client.get("/static/nonexistent.js").status_code != 401


def test_rpc_routes_skip_middleware(authed_client):
    """RPC routes are mounts the HTTP middleware SKIPs, so a valid-token RPC reaches
    the service through its own interceptor chain rather than being blocked as an
    unannotated (default-deny) route."""
    resp = authed_client.post(
        "/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_TEST_TOKEN}"},
    )
    assert resp.status_code == 200


def test_all_routes_accessible_when_auth_disabled(noauth_client):
    """The permissive chain admits every route when auth is not configured."""
    for path in ["/job/123", "/worker/456", "/health", "/auth/config"]:
        assert noauth_client.get(path).status_code == 200


# -- Auth DB isolation ---------------------------------------------------------


def _write_secret(db: ControllerDB, key: str, value: str) -> None:
    """Write a row into the attached auth DB (``controller_secrets``)."""
    with db.transaction() as tx:
        tx.execute(
            text("INSERT INTO auth.controller_secrets (key, value, created_at_ms) VALUES (:k, :v, 1000)"),
            {"k": key, "v": value},
        )


def test_read_snapshot_cannot_access_auth_tables(db: ControllerDB):
    """Read pool connections must not see the attached auth DB's tables."""
    _write_secret(db, "signing_key", "pem")

    with db.read_snapshot() as q:
        for table in ["controller_secrets", "auth.controller_secrets"]:
            with pytest.raises(sqlalchemy.exc.OperationalError, match="no such table"):
                q.execute(text(f"SELECT * FROM {table}"))


def test_write_connection_can_access_auth_tables(db: ControllerDB):
    _write_secret(db, "signing_key", "pem")

    with db.transaction() as q:
        rows = q.execute(text("SELECT value FROM auth.controller_secrets WHERE key = 'signing_key'")).all()
        assert len(rows) == 1
        assert rows[0].value == "pem"


# -- Stateless JWT -------------------------------------------------------------


def test_jwt_create_and_verify():
    """A minted control-plane token round-trips through the stateless verifier."""
    mgr = _jwt_manager()
    token = mgr.create_token("bob", "user", "k-bob", ttl_seconds=SESSION_TOKEN_TTL_SECONDS)
    identity = mgr.verify(token)
    assert identity.user_id == "bob"
    assert identity.role == "user"


def _federation_setup(requester: str = "parent-cluster"):
    """A parent JwtTokenManager plus a peer's verifier trusting the parent's key."""
    key = signing_key_from_private_pem(generate_ed25519_keypair().private_pem)
    signer = JwtSigner(key, issuer=requester)
    cp_verifier = JwksVerifier(issuers={requester: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES)
    parent = JwtTokenManager(signer, cp_verifier)
    peer_verifier = FederationTokenVerifier({requester: key.public_pem})
    return parent, peer_verifier


def test_federation_token_round_trips_to_a_scoped_requester_identity():
    """A peer verifies a parent's federation token against the parent's key and gets a
    method-scoped federation-peer identity whose user_id is the verified requester."""
    parent, peer_verifier = _federation_setup("parent-cluster")
    token = parent.create_federation_token("parent-cluster", "k-fed")
    identity = peer_verifier.verify(token)
    assert identity.user_id == "parent-cluster"
    assert identity.role == FEDERATION_PEER_ROLE
    assert identity.audience is None


def test_control_plane_verify_rejects_a_federation_token():
    """The cross-plane guard: a parent's own aud="federation" token is rejected at its
    control-plane verify, so it can never be replayed at the general RPC surface."""
    parent, _ = _federation_setup("parent-cluster")
    token = parent.create_federation_token("parent-cluster", "k-fed")
    with pytest.raises(ValueError):
        parent.verify(token)
    assert jwt.decode(token, options={"verify_signature": False})["aud"] == FEDERATION_AUDIENCE


def test_federation_verifier_rejects_a_control_plane_token():
    """The federation verifier accepts only aud="federation"; a control-plane token
    signed by the same key is rejected (the plane is bound, not just the key)."""
    parent, peer_verifier = _federation_setup("parent-cluster")
    control = parent.create_token("bob", "admin", "k-ctl", ttl_seconds=60)
    with pytest.raises(ValueError):
        peer_verifier.verify(control)


def test_federation_verifier_rejects_an_untrusted_issuer():
    """A federation token minted by a cluster the peer does not trust is rejected —
    only configured peer keys verify, so the requester can't be forged."""
    stranger, _ = _federation_setup("evil-cluster")
    _, peer_verifier = _federation_setup("parent-cluster")  # trusts only parent-cluster
    forged = stranger.create_federation_token("evil-cluster", "k-fed")
    with pytest.raises(ValueError):
        peer_verifier.verify(forged)


# ---------------------------------------------------------------------------
# Token issuer: the cluster name, with a transitional legacy issuer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("audience", "mint"),
    [
        ("control-plane", lambda mgr: mgr.create_token("alice", "admin", "k", ttl_seconds=60)),
        ("proxy", lambda mgr: mgr.create_endpoint_token("ep", "k")),
        ("federation", lambda mgr: mgr.create_federation_token("named", "k")),
    ],
)
def test_every_token_is_issued_under_the_cluster_name(audience, mint):
    """``iss`` is the cluster's identity on every plane — it is what a federation peer
    keys its trust on, so no audience may mint under a different issuer."""
    mgr = _build_jwt_token_manager(cluster_name="named", signing_key_pem=_SIGNING_KEY, previous_public_keys=())
    assert jwt.decode(mint(mgr), options={"verify_signature": False})["iss"] == "named"


def test_worker_token_minted_before_the_cluster_was_named_still_verifies():
    """Naming a cluster must not invalidate the worker tokens its fleet already holds.

    A worker token is minted once per controller start and injected into every worker with
    no refresh path, so it outlives the restart that first sets ``name``. Under the same
    signing key, the control-plane verifier still accepts the legacy issuer."""
    unnamed = _build_jwt_token_manager(cluster_name="", signing_key_pem=_SIGNING_KEY, previous_public_keys=())
    legacy = unnamed.create_token(WORKER_USER, "worker", "w1", ttl_seconds=WORKER_TOKEN_TTL_SECONDS)
    assert jwt.decode(legacy, options={"verify_signature": False})["iss"] == _LEGACY_ISSUER

    named = _build_jwt_token_manager(cluster_name="named", signing_key_pem=_SIGNING_KEY, previous_public_keys=())
    identity = named.verify(legacy)
    assert identity.user_id == WORKER_USER
    assert identity.role == "worker"


def test_legacy_issuer_is_not_accepted_on_the_federation_plane():
    """The legacy issuer is a control-plane migration aid only. A peer trusts each
    cluster under its real name, so an unnamed cluster cannot federate to it."""
    unnamed = _build_jwt_token_manager(cluster_name="", signing_key_pem=_SIGNING_KEY, previous_public_keys=())
    peer_verifier = FederationTokenVerifier({"named": _PUBLIC_KEY})
    with pytest.raises(ValueError, match="issuer"):
        peer_verifier.verify(unnamed.create_federation_token("", "k-fed"))


def test_legacy_issuer_still_requires_this_controllers_key():
    """Accepting the legacy issuer widens nothing: it resolves to the same key set, so a
    token another cluster signed under ``iris`` is still rejected on the signature."""
    stranger = _build_jwt_token_manager(
        cluster_name="", signing_key_pem=generate_ed25519_keypair().private_pem, previous_public_keys=()
    )
    forged = stranger.create_token("mallory", "admin", "k", ttl_seconds=60)
    named = _build_jwt_token_manager(cluster_name="named", signing_key_pem=_SIGNING_KEY, previous_public_keys=())
    with pytest.raises(ValueError, match="signature"):
        named.verify(forged)


def test_composite_verifier_routes_each_plane_to_its_verifier():
    """The composite accepts a control-plane token as a full identity and a federation
    token as a method-scoped federation-peer identity, never crossing planes."""
    parent, peer_verifier = _federation_setup("parent-cluster")
    composite = _ControlPlaneOrFederationVerifier(parent, peer_verifier)

    control = parent.create_token("bob", "admin", "k-ctl", ttl_seconds=60)
    control_identity = composite.verify(control)
    assert control_identity.user_id == "bob"
    assert control_identity.role == "admin"

    federation = parent.create_federation_token("parent-cluster", "k-fed")
    fed_identity = composite.verify(federation)
    assert fed_identity.role == FEDERATION_PEER_ROLE
    assert fed_identity.user_id == "parent-cluster"


def test_endpoint_token_surfaces_endpoint_as_audience():
    """An endpoint token verifies (aud is the fixed proxy plane) and surfaces its
    bound endpoint name as ``identity.audience`` (what the /proxy gate matches)."""
    mgr = _jwt_manager()
    token = mgr.create_endpoint_token("/serve/foo", "k-ep", ttl_seconds=60)
    identity = mgr.verify(token)
    assert identity.audience == "/serve/foo"


# -- CIDR network-location auth -------------------------------------------------


def test_cidr_only_auth_config_enables_request_auth():
    """An auth block with only trusted_cidrs turns auth on.

    Direct in-network peers resolve to an admin identity; external and
    forwarded peers are rejected; the cluster's worker JWT still verifies
    through the same policy.
    """
    auth = create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"]), cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY
    )
    policy = request_auth_policy(auth)
    assert not policy.allows_anonymous

    inside = policy.resolve(None, client_address="10.1.2.3:5555", headers={})
    assert inside is not None
    assert inside.role == "admin"

    with pytest.raises(ValueError, match="Missing authentication"):
        policy.resolve(None, client_address="203.0.113.9:5555", headers={})

    # A forwarded request whose socket peer is an in-CIDR ingress hop must not
    # inherit the hop's network location.
    with pytest.raises(ValueError, match="Missing authentication"):
        policy.resolve(None, client_address="10.1.2.3:5555", headers={"x-forwarded-for": "203.0.113.9"})

    worker = policy.resolve(auth.worker_token, client_address="203.0.113.9:5555", headers={})
    assert worker is not None
    assert worker.user_id == WORKER_USER


# -- Optional auth (gradual adoption) -----------------------------------------


@pytest.fixture
def optional_auth_client(service, verifier):
    """Dashboard with auth configured but optional — tokens verified if present, anonymous fallback."""
    dashboard = ControllerDashboard(
        service,
        auth_provider="iap",
        auth_policy=RequestAuthPolicy.enforcing(verifier=verifier, optional=True),
    )
    return TestClient(dashboard.app)


def test_optional_auth_allows_unauthenticated_rpc(optional_auth_client):
    """RPCs succeed without a token, falling back to anonymous/admin identity."""
    resp = optional_auth_client.post(
        "/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200


def test_optional_auth_uses_token_when_present(optional_auth_client):
    """When a valid token is supplied, the authenticated identity is used."""
    resp = optional_auth_client.post(
        "/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_TEST_TOKEN}"},
    )
    assert resp.status_code == 200


def test_optional_auth_rejects_invalid_token(optional_auth_client):
    """An invalid token is rejected — optional mode still enforces token validity."""
    resp = optional_auth_client.post(
        "/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={"Content-Type": "application/json", "Authorization": "Bearer bad-token"},
    )
    assert resp.status_code == 401


def test_optional_auth_dashboard_accessible(optional_auth_client):
    """Dashboard pages are accessible without auth in optional mode."""
    for path in ["/", "/job/123", "/worker/456", "/health"]:
        assert optional_auth_client.get(path).status_code == 200


def test_optional_auth_config_reports_optional(optional_auth_client):
    """The /auth/config endpoint reports optional=true."""
    data = optional_auth_client.get("/auth/config").json()
    assert data["auth_enabled"] is True
    assert data["optional"] is True
    assert data["provider"] == "iap"


def test_auth_config_reports_not_optional(authed_client):
    """Non-optional auth reports optional=false."""
    data = authed_client.get("/auth/config").json()
    assert data["optional"] is False


# -- Route middleware parity: HTTP agrees with the auth chain ----------------


@pytest.mark.parametrize(
    "token, optional, should_allow",
    [
        (None, False, False),
        (None, True, True),
        (_TEST_TOKEN, False, True),
        (_TEST_TOKEN, True, True),
        ("bad-token", False, False),
        ("bad-token", True, False),
    ],
    ids=[
        "no-token-required",
        "no-token-optional",
        "valid-required",
        "valid-optional",
        "invalid-required",
        "invalid-optional",
    ],
)
def test_route_auth_middleware_matches_rpc_policy(service, verifier, token, optional, should_allow):
    """RouteAuthMiddleware applies the same auth chain as the RPC interceptor.

    We build a dashboard with a @requires_auth route injected and verify it
    agrees with the chain for every (token, optional) combination.
    """
    policy = RequestAuthPolicy.enforcing(verifier=verifier, optional=optional)
    dashboard = _dashboard_with_protected_route(service, policy)

    client = TestClient(dashboard.app)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = client.get("/test-protected", headers=headers)
    if should_allow:
        assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}"
    else:
        assert resp.status_code == 401, f"Expected 401 but got {resp.status_code}"


def test_route_auth_middleware_rejects_endpoint_scoped_token(service):
    """A valid endpoint-scoped token gets 403 from @requires_auth routes.

    Such a token authorizes only its endpoint's /proxy path; the middleware must
    refuse it everywhere else even though the token itself verifies.
    """
    mgr = _jwt_manager()
    token = mgr.create_endpoint_token("/u/job/ep", "iris_ket_route", ttl_seconds=60)
    dashboard = _dashboard_with_protected_route(service, RequestAuthPolicy.enforcing(verifier=mgr))

    resp = TestClient(dashboard.app).get("/test-protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def _dashboard_with_protected_route(service, policy: RequestAuthPolicy) -> ControllerDashboard:
    """A dashboard with a @requires_auth route injected for middleware tests."""

    @requires_auth
    def _protected(_request):
        return JSONResponse({"ok": True})

    dashboard = ControllerDashboard(service, auth_provider="iap", auth_policy=policy)
    # Walk down to the Starlette router so the new route participates in route
    # matching.
    app = dashboard.app
    while isinstance(app, _SubdomainProxyMiddleware | RouteAuthMiddleware):
        app = app._app
    app.router.routes.insert(0, Route("/test-protected", _protected))
    return dashboard


# -- IAP implicit dashboard role through the live auth interceptor ------------


def _dashboard_interceptor(**verifiers):
    """The interceptor exactly as the dashboard wires it (RPC exemptions + RBAC)."""
    policy = RequestAuthPolicy.enforcing(verifier=MockVerifier({}), **verifiers)
    return PolicyAuthInterceptor(
        policy,
        cookie_name=SESSION_COOKIE,
        unauthenticated_methods=_UNAUTHENTICATED_RPCS,
        authorize=authorize_method,
    )


class _StubAssertionVerifier:
    """IapAssertionVerifier stand-in: a present signed-header => dashboard identity."""

    def identity_from_headers(self, headers):
        if headers.get("x-goog-iap-jwt-assertion"):
            return VerifiedIdentity(user_id="alice@example.com", role=DASHBOARD_ROLE)
        return None


def _assertion_ctx(method_name: str):
    """Fake RPC ctx for an IAP-fronted, tokenless request (no Iris JWT)."""

    class _Ctx:
        def method(self):
            info = Mock()
            info.name = method_name  # Mock(name=...) sets repr, not the attribute
            return info

        def request_headers(self):
            return {"x-goog-iap-jwt-assertion": "signed.assertion.jwt"}

        def client_address(self):
            return "10.0.0.7:443"  # arrived via the load balancer, not loopback

    return _Ctx()


def test_dashboard_interceptor_allows_read_for_iap_browser():
    interceptor = _dashboard_interceptor(iap_assertion_verifier=_StubAssertionVerifier())
    seen = []

    def handler(_req, _ctx):
        seen.append(get_verified_identity())
        return "ok"

    result = interceptor.intercept_unary_sync(handler, "req", _assertion_ctx("ListJobs"))
    assert result == "ok"
    assert seen == [VerifiedIdentity(user_id="alice@example.com", role=DASHBOARD_ROLE)]


def test_dashboard_interceptor_denies_mutation_for_iap_browser():
    interceptor = _dashboard_interceptor(iap_assertion_verifier=_StubAssertionVerifier())
    ran = []

    def handler(_req, _ctx):
        ran.append(True)
        return "ok"

    with pytest.raises(ConnectError) as exc:
        interceptor.intercept_unary_sync(handler, "req", _assertion_ctx("LaunchJob"))
    assert exc.value.code == Code.PERMISSION_DENIED
    assert ran == []  # the handler never runs for a denied mutation


class _RoleAssertionVerifier:
    """IapAssertionVerifier stand-in returning a fixed role for the asserted email.

    Mirrors the controller's email->role resolution: a provisioned admin/user
    resolves to their real role, an unprovisioned email to read-only dashboard.
    """

    def __init__(self, role):
        self._role = role

    def identity_from_headers(self, headers):
        if headers.get("x-goog-iap-jwt-assertion"):
            return VerifiedIdentity(user_id="admin@example.com", role=self._role)
        return None


def test_dashboard_interceptor_allows_mutation_for_provisioned_iap_admin():
    # The point of resolving the IAP identity to its real role: a provisioned
    # admin behind IAP (no Iris JWT) resolves to the admin role and so reaches a
    # gated mutation that the read-only dashboard role would be denied.
    interceptor = _dashboard_interceptor(iap_assertion_verifier=_RoleAssertionVerifier("admin"))
    seen = []

    def handler(_req, _ctx):
        seen.append(get_verified_identity())
        return "ok"

    result = interceptor.intercept_unary_sync(handler, "req", _assertion_ctx("LaunchJob"))
    assert result == "ok"
    assert seen == [VerifiedIdentity(user_id="admin@example.com", role="admin")]


def test_role_policy_resolves_admin_worker_and_default():
    # The in-memory, config-derived role map: config-listed admins -> admin, the
    # internal worker identity -> worker, everyone else -> the default role.
    policy = RolePolicy(admins=frozenset({"alice@example.com"}), default_role="user")
    assert policy.role_for("alice@example.com") == "admin"
    assert policy.role_for(WORKER_USER) == "worker"
    assert policy.role_for("stranger@example.com") == "user"

    # unprovisioned_role=admin (IAP's own allowlist as the sole gate): a non-listed
    # email acts as admin via the default; the worker identity is still worker.
    open_policy = RolePolicy(admins=frozenset(), default_role="admin")
    assert open_policy.role_for("stranger@example.com") == "admin"
    assert open_policy.role_for(WORKER_USER) == "worker"


def test_role_policy_default_role_from_provider():
    # default_role is wired from the provider: IAP uses its unprovisioned_role;
    # cidr-only uses "user". These are the roles an authenticated non-admin resolves to.
    iap = create_controller_auth(
        AuthConfig(
            iap=IapAuthConfig(
                signed_header_audience="/projects/1/global/backendServices/2", unprovisioned_role="dashboard"
            )
        ),
        cluster_name=_CLUSTER,
        signing_key_pem=_SIGNING_KEY,
    )
    assert iap.role_policy is not None
    assert iap.role_policy.default_role == DASHBOARD_ROLE

    cidr = create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"]),
        cluster_name=_CLUSTER,
        signing_key_pem=_SIGNING_KEY,
    )
    assert cidr.role_policy is not None
    assert cidr.role_policy.default_role == "user"


def test_iap_assertion_resolver_is_the_role_policy():
    # The controller injects role_policy.role_for as the IAP assertion resolver, so a
    # config-listed admin email resolves to admin and everyone else to the default —
    # all in memory, no DB.
    auth = create_controller_auth(
        AuthConfig(
            iap=IapAuthConfig(
                signed_header_audience="/projects/1/global/backendServices/2", unprovisioned_role="dashboard"
            ),
            admin_users=["admin@example.com"],
        ),
        cluster_name=_CLUSTER,
        signing_key_pem=_SIGNING_KEY,
    )
    assert auth.role_policy is not None
    assert auth.role_policy.role_for("admin@example.com") == "admin"
    assert auth.role_policy.role_for("stranger@example.com") == DASHBOARD_ROLE


# -- require_persistent_signing_key --------------------------------------------


def test_require_persistent_signing_key():
    # A controller with peers signs federation tokens each peer pins to its public key,
    # so an empty signing key is a silent trust-anchor break: fail fast.
    peers = {"cw-rno2a": PeerConfig(controller_address="https://peer:8080", cluster="cw-rno2a")}
    with pytest.raises(ValueError, match="requires a persistent"):
        require_persistent_signing_key(peers, None)
    require_persistent_signing_key(peers, _SIGNING_KEY)  # a key present: fine

    # No peers: nothing external pins this controller's key, so an ephemeral key is fine
    # and must NOT be rejected. A cluster that only *receives* federated calls verifies
    # with its peers' published keys and signs nothing anyone else pins.
    require_persistent_signing_key({}, None)


# -- Controller auth setup -----------------------------------------------------


def test_worker_token_verifies():
    auth = create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"]), cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY
    )
    assert auth.worker_token is not None
    assert auth.verifier.verify(auth.worker_token).user_id == WORKER_USER


def test_iap_provider_requires_signed_header_audience():
    # Pure-IAP: an iap arm with no signed_header_audience has no way to authenticate
    # a request (the controller mints no login token), so construction fails fast.
    with pytest.raises(ValueError, match="signed_header_audience"):
        create_controller_auth(
            AuthConfig(iap={"url": "https://iris-marin.example.com"}),
            cluster_name=_CLUSTER,
            signing_key_pem=_SIGNING_KEY,
        )


def test_worker_token_differs_after_restart():
    # A persistent signing key is shared across restarts, so a token minted before
    # the restart still verifies after it — but each start mints a fresh worker jti,
    # so the tokens differ. Old worker tokens simply age out at their TTL.
    signing_key_pem = generate_ed25519_keypair().private_pem
    config = AuthConfig(trusted_cidrs=["10.0.0.0/8"])
    auth1 = create_controller_auth(config, cluster_name=_CLUSTER, signing_key_pem=signing_key_pem)
    auth2 = create_controller_auth(config, cluster_name=_CLUSTER, signing_key_pem=signing_key_pem)
    assert auth1.worker_token != auth2.worker_token
    # Both still verify (old not revoked) under auth2's verifier (same signing key).
    assert auth2.verifier.verify(auth1.worker_token).user_id == WORKER_USER
    assert auth2.verifier.verify(auth2.worker_token).user_id == WORKER_USER


def test_admin_users_resolve_to_admin_in_role_policy():
    # Config's admin_users are the authoritative admin set, carried on the in-memory
    # RolePolicy — no DB projection, no reconciliation.
    auth = create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"], admin_users=["alice"]),
        cluster_name=_CLUSTER,
        signing_key_pem=_SIGNING_KEY,
    )
    assert auth.role_policy is not None
    assert auth.role_policy.role_for("alice") == "admin"
    assert auth.role_policy.role_for("bob") == "user"


def test_admin_deprovisioned_by_rebuilding_policy_from_new_config():
    # Deprovisioning is purely in-memory: rebuild the auth (as a controller restart
    # does) from config with a user removed from admin_users, and the new RolePolicy
    # resolves them to the non-admin default — no DB, no reconciliation step.
    def _boot(admin_users):
        return create_controller_auth(
            AuthConfig(trusted_cidrs=["10.0.0.0/8"], admin_users=admin_users),
            cluster_name=_CLUSTER,
            signing_key_pem=_SIGNING_KEY,
        )

    both = _boot(["alice", "bob"])
    assert both.role_policy.role_for("alice") == "admin"
    assert both.role_policy.role_for("bob") == "admin"

    after = _boot(["alice"])  # bob de-listed, then controller reloaded (policy rebuilt)
    assert after.role_policy.role_for("alice") == "admin"
    assert after.role_policy.role_for("bob") == "user"


# -- Null-auth mode ------------------------------------------------------------


def test_null_auth_yields_worker_token_and_default_policy():
    # Null-auth mints a worker token and a verifier but no login provider. The
    # anonymous/loopback admin identity is assigned by the permissive auth chain,
    # not resolved through the RolePolicy (whose default is irrelevant here) — there
    # is no users table to consult.
    auth = create_controller_auth(AuthConfig(), cluster_name=_CLUSTER)
    assert auth.verifier is not None
    assert auth.worker_token is not None
    assert auth.provider is None
    assert auth.role_policy is not None
    assert auth.role_policy.role_for(WORKER_USER) == "worker"


def test_null_auth_get_current_user(db, log_client):
    auth = create_controller_auth(AuthConfig(), cluster_name=_CLUSTER)
    service = _make_service(db, log_client, auth=auth)
    jwt_token = auth.jwt_manager.create_token("anonymous", "admin", "iris_s_test", ttl_seconds=SESSION_TOKEN_TTL_SECONDS)
    reset = _verified_identity.set(auth.verifier.verify(jwt_token))
    try:
        resp = service.get_current_user(job_pb2.GetCurrentUserRequest(), None)
        assert resp.user_id == "anonymous"
        assert resp.role == "admin"
    finally:
        _verified_identity.reset(reset)
