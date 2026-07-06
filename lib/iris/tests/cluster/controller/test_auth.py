# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth: session cookies, CSRF, default-deny middleware, auth DB isolation, API keys, and JWT."""

from unittest.mock import Mock

import pytest
import sqlalchemy.exc
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import AuthConfig
from iris.cluster.controller import reads, writes
from iris.cluster.controller.auth import (
    WORKER_USER,
    JwtTokenManager,
    _get_or_create_signing_key,
    _make_iap_role_resolver,
    create_api_key,
    create_controller_auth,
    list_api_keys,
    lookup_api_key_by_id,
    request_auth_policy,
    revoke_api_key,
    revoke_login_keys_for_user,
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
from iris.rpc.auth import DASHBOARD_ROLE, SESSION_COOKIE, authorize_method
from rigging.server_auth import (
    PolicyAuthInterceptor,
    RequestAuthPolicy,
    RouteAuthMiddleware,
    StaticTokenVerifier,
    VerifiedIdentity,
    get_verified_identity,
    requires_auth,
)
from rigging.timing import Timestamp
from sqlalchemy import text
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from tests.cluster.controller._test_support import ControllerTestState

_TEST_TOKEN = "valid-test-token"
_TEST_USER = "test-user"
CSRF_HEADERS = {"Origin": "http://testserver"}


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
    return StaticTokenVerifier({_TEST_TOKEN: _TEST_USER})


@pytest.fixture
def authed_client(service, verifier):
    dashboard = ControllerDashboard(
        service,
        auth_provider="gcp",
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
    """RPC routes use their own interceptor chain, not the HTTP middleware."""
    resp = authed_client.post(
        "/iris.cluster.ControllerService/GetAuthInfo",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code != 401


def test_all_routes_accessible_when_auth_disabled(noauth_client):
    """The permissive chain admits every route when auth is not configured."""
    for path in ["/job/123", "/worker/456", "/health", "/auth/config"]:
        assert noauth_client.get(path).status_code == 200


# -- Session bootstrap ---------------------------------------------------------


def test_session_bootstrap_valid_token(authed_client):
    resp = authed_client.get(f"/auth/session_bootstrap?token={_TEST_TOKEN}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"].endswith("/")
    assert SESSION_COOKIE in resp.cookies


def test_session_bootstrap_invalid_token(authed_client):
    resp = authed_client.get("/auth/session_bootstrap?token=bad-token", follow_redirects=False)
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid token"


def test_session_bootstrap_no_token(authed_client):
    resp = authed_client.get("/auth/session_bootstrap", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"].endswith("/")
    assert SESSION_COOKIE not in resp.cookies


def test_session_bootstrap_no_auth_configured(noauth_client):
    resp = noauth_client.get(f"/auth/session_bootstrap?token={_TEST_TOKEN}", follow_redirects=False)
    assert resp.status_code == 302
    assert SESSION_COOKIE not in resp.cookies


# -- Auth DB isolation ---------------------------------------------------------


def test_read_snapshot_cannot_access_auth_tables(db: ControllerDB):
    """Read pool connections must not see auth tables."""
    now = Timestamp.now()
    with db.transaction() as _tx:
        writes.ensure_user(_tx, "test-user", now)
    _get_or_create_signing_key(db)
    create_api_key(db, key_id="k1", key_prefix="pfx", user_id="test-user", name="test", now=now)

    with db.read_snapshot() as q:
        for table in ["api_keys", "controller_secrets", "auth.api_keys"]:
            with pytest.raises(sqlalchemy.exc.OperationalError, match="no such table"):
                q.execute(text(f"SELECT * FROM {table}"))


def test_write_connection_can_access_auth_tables(db: ControllerDB):
    now = Timestamp.now()
    with db.transaction() as _tx:
        writes.ensure_user(_tx, "test-user", now)
    _get_or_create_signing_key(db)
    create_api_key(db, key_id="k1", key_prefix="pfx", user_id="test-user", name="test", now=now)

    with db.transaction() as q:
        rows = q.execute(text("SELECT key_id FROM auth.api_keys")).all()
        assert len(rows) == 1
        assert rows[0].key_id == "k1"


# -- API keys and JWT ----------------------------------------------------------


def test_api_key_create_lookup_revoke(db: ControllerDB):
    now = Timestamp.now()
    with db.transaction() as _tx:
        writes.ensure_user(_tx, "alice", now, role="admin")
        writes.set_user_role(_tx, "alice", "admin")
    with db.read_snapshot() as _snap:
        assert reads.get_user_role(_snap, "alice") == "admin"

    create_api_key(db, key_id="k1", key_prefix="sec", user_id="alice", name="my-key", now=now)

    found = lookup_api_key_by_id(db, "k1")
    assert found is not None
    assert found.key_id == "k1"
    assert found.key_prefix == "sec"

    keys = list_api_keys(db, user_id="alice")
    assert len(keys) == 1

    assert revoke_api_key(db, "k1", now)


def test_jwt_create_and_verify(db: ControllerDB):
    now = Timestamp.now()
    with db.transaction() as _tx:
        writes.ensure_user(_tx, "bob", now, role="user")

    signing_key = _get_or_create_signing_key(db)
    mgr = JwtTokenManager(signing_key, db=db)

    create_api_key(db, key_id="k-bob", key_prefix="jwt", user_id="bob", name="test", now=now)

    token = mgr.create_token("bob", "user", "k-bob")
    identity = mgr.verify(token)
    assert identity.user_id == "bob"
    assert identity.role == "user"


def test_revoke_login_keys(db: ControllerDB):
    now = Timestamp.now()
    with db.transaction() as _tx:
        writes.ensure_user(_tx, "carol", now)

    for i in (1, 2):
        create_api_key(
            db,
            key_id=f"k-login-{i}",
            key_prefix="jwt",
            user_id="carol",
            name=f"login-{i}",
            now=now,
        )

    revoked_ids = revoke_login_keys_for_user(db, "carol", now)
    assert set(revoked_ids) == {"k-login-1", "k-login-2"}


# -- CIDR network-location auth -------------------------------------------------


def test_cidr_only_auth_config_enables_request_auth(db: ControllerDB):
    """An auth block with only trusted_cidrs turns auth on.

    Direct in-network peers resolve to an admin identity; external and
    forwarded peers are rejected; the cluster's worker JWT still verifies
    through the same policy.
    """
    auth = create_controller_auth(AuthConfig(trusted_cidrs=["10.0.0.0/8"]), db=db)
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
        auth_provider="static",
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
        "/iris.cluster.ControllerService/GetAuthInfo",
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
    assert data["provider"] == "static"


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
    mgr = JwtTokenManager("route-auth-test-signing-key")
    token = mgr.create_endpoint_token("/u/job/ep", "iris_ket_route", ttl_seconds=60)
    dashboard = _dashboard_with_protected_route(service, RequestAuthPolicy.enforcing(verifier=mgr))

    resp = TestClient(dashboard.app).get("/test-protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def _dashboard_with_protected_route(service, policy: RequestAuthPolicy) -> ControllerDashboard:
    """A dashboard with a @requires_auth route injected for middleware tests."""

    @requires_auth
    def _protected(_request):
        return JSONResponse({"ok": True})

    dashboard = ControllerDashboard(service, auth_provider="static", auth_policy=policy)
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
    policy = RequestAuthPolicy.enforcing(verifier=StaticTokenVerifier({}), **verifiers)
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


def test_dashboard_interceptor_login_reachable_for_unprovisioned_iap_browser():
    # `Login`/`GetAuthInfo` are exempt from auth (in _UNAUTHENTICATED_RPCS), so
    # even an unprovisioned IAP caller (read-only dashboard role) reaches the
    # Login handler — `iris login` is never blocked by the dashboard gate. Guards
    # against accidentally moving the role check ahead of that exemption.
    interceptor = _dashboard_interceptor(iap_assertion_verifier=_RoleAssertionVerifier(DASHBOARD_ROLE))
    result = interceptor.intercept_unary_sync(lambda _req, _ctx: "ok", "req", _assertion_ctx("Login"))
    assert result == "ok"


def test_iap_role_resolver_maps_provisioned_and_unknown_emails(db: ControllerDB):
    # The resolver the controller injects into IapAssertionVerifier: a provisioned
    # email gets its stored role; an unprovisioned email gets the configured
    # fallback role.
    now = Timestamp.now()
    with db.transaction() as tx:
        writes.ensure_user(tx, "admin@example.com", now, role="admin")
        writes.set_user_role(tx, "admin@example.com", "admin")
        writes.ensure_user(tx, "user@example.com", now, role="user")

    resolve = _make_iap_role_resolver(db, DASHBOARD_ROLE)
    assert resolve("admin@example.com") == "admin"
    assert resolve("user@example.com") == "user"
    assert resolve("stranger@example.com") == DASHBOARD_ROLE

    # unprovisioned_role=admin (IAP's own allowlist as the sole gate): strangers
    # act as admin, a provisioned user still gets their stored role.
    resolve_admin = _make_iap_role_resolver(db, "admin")
    assert resolve_admin("user@example.com") == "user"
    assert resolve_admin("stranger@example.com") == "admin"
