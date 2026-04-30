# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth: session cookies, CSRF, default-deny middleware, auth DB isolation, API keys, and JWT."""

import sqlite3
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from iris.cluster.bundle import BundleStore
from finelog.server import LogServiceImpl
from iris.cluster.controller.auth import (
    JwtTokenManager,
    _get_or_create_signing_key,
    create_api_key,
    list_api_keys,
    lookup_api_key_by_hash,
    revoke_api_key,
    revoke_login_keys_for_user,
)
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import ControllerTransitions
from iris.rpc.auth import SESSION_COOKIE, StaticTokenVerifier, hash_token, resolve_auth
from rigging.timing import Timestamp

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
    s = ControllerTransitions(store=ControllerStore(db))
    yield s


@pytest.fixture
def service(state, tmp_path):
    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.autoscaler = None
    controller_mock.provider = Mock()
    controller_mock.has_direct_provider = False
    return ControllerServiceImpl(
        state,
        state._store,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=LogServiceImpl(),
    )


@pytest.fixture
def verifier():
    return StaticTokenVerifier({_TEST_TOKEN: _TEST_USER})


@pytest.fixture
def authed_client(service, verifier):
    dashboard = ControllerDashboard(
        service, log_service=service._log_service, auth_verifier=verifier, auth_provider="gcp"
    )
    return TestClient(dashboard.app)


@pytest.fixture
def noauth_client(service):
    dashboard = ControllerDashboard(service, log_service=service._log_service)
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


def test_no_middleware_when_auth_disabled(noauth_client):
    """All routes accessible when auth is not configured."""
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
    db.ensure_user("test-user", now)
    _get_or_create_signing_key(db)
    create_api_key(db, key_id="k1", key_hash="hash1", key_prefix="pfx", user_id="test-user", name="test", now=now)

    with db.read_snapshot() as q:
        for table in ["api_keys", "controller_secrets", "auth.api_keys"]:
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                q.raw(f"SELECT * FROM {table}")


def test_write_connection_can_access_auth_tables(db: ControllerDB):
    now = Timestamp.now()
    db.ensure_user("test-user", now)
    _get_or_create_signing_key(db)
    create_api_key(db, key_id="k1", key_hash="hash1", key_prefix="pfx", user_id="test-user", name="test", now=now)

    with db.snapshot() as q:
        rows = q.raw(f"SELECT key_id FROM {db.api_keys_table}", decoders={"key_id": str})
        assert len(rows) == 1
        assert rows[0].key_id == "k1"


def test_auth_db_file_created(tmp_path):
    auth_path = tmp_path / "auth.sqlite3"
    assert not auth_path.exists()
    db = ControllerDB(db_dir=tmp_path)
    assert auth_path.exists()
    db.close()


# -- API keys and JWT ----------------------------------------------------------


def test_api_key_create_lookup_revoke(db: ControllerDB):
    now = Timestamp.now()
    db.ensure_user("alice", now, role="admin")
    db.set_user_role("alice", "admin")
    assert db.get_user_role("alice") == "admin"

    create_api_key(
        db, key_id="k1", key_hash=hash_token("secret1"), key_prefix="sec", user_id="alice", name="my-key", now=now
    )

    found = lookup_api_key_by_hash(db, hash_token("secret1"))
    assert found is not None
    assert found.key_id == "k1"

    keys = list_api_keys(db, user_id="alice")
    assert len(keys) == 1

    assert revoke_api_key(db, "k1", now)


def test_jwt_create_and_verify(db: ControllerDB):
    now = Timestamp.now()
    db.ensure_user("bob", now, role="user")

    signing_key = _get_or_create_signing_key(db)
    mgr = JwtTokenManager(signing_key, db=db)

    create_api_key(db, key_id="k-bob", key_hash="jwt:k-bob", key_prefix="jwt", user_id="bob", name="test", now=now)

    token = mgr.create_token("bob", "user", "k-bob")
    identity = mgr.verify(token)
    assert identity.user_id == "bob"
    assert identity.role == "user"


def test_revoke_login_keys(db: ControllerDB):
    now = Timestamp.now()
    db.ensure_user("carol", now)

    for i in (1, 2):
        create_api_key(
            db,
            key_id=f"k-login-{i}",
            key_hash=f"jwt:k-login-{i}",
            key_prefix="jwt",
            user_id="carol",
            name=f"login-{i}",
            now=now,
        )

    revoked_ids = revoke_login_keys_for_user(db, "carol", now)
    assert set(revoked_ids) == {"k-login-1", "k-login-2"}


# -- Optional auth (gradual adoption) -----------------------------------------


@pytest.fixture
def optional_auth_client(service, verifier):
    """Dashboard with auth configured but optional — tokens verified if present, anonymous fallback."""
    dashboard = ControllerDashboard(
        service,
        log_service=service._log_service,
        auth_verifier=verifier,
        auth_provider="static",
        auth_optional=True,
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


# -- resolve_auth parity: gRPC and HTTP agree on allow/deny ----------------


@pytest.mark.parametrize(
    "token, optional, should_succeed",
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
def test_resolve_auth_policy(verifier, token, optional, should_succeed):
    """resolve_auth encodes the single auth policy used by both gRPC and HTTP."""
    if should_succeed:
        identity = resolve_auth(token, verifier, optional)
        if token == _TEST_TOKEN:
            assert identity is not None
            assert identity.user_id == _TEST_USER
        else:
            assert identity is None
    else:
        with pytest.raises(ValueError):
            resolve_auth(token, verifier, optional)


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
def test_route_auth_middleware_uses_resolve_auth(service, verifier, token, optional, should_allow):
    """_RouteAuthMiddleware applies the same resolve_auth policy as the gRPC interceptor.

    We build a dashboard with a @requires_auth route injected and verify it
    agrees with resolve_auth for every (token, optional) combination.
    """
    from iris.cluster.controller.dashboard import ControllerDashboard, requires_auth

    from iris.cluster.controller.dashboard import _RouteAuthMiddleware
    from starlette.routing import Route
    from starlette.responses import JSONResponse as _J

    @requires_auth
    def _protected(_request):
        return _J({"ok": True})

    dashboard = ControllerDashboard(
        service,
        log_service=service._log_service,
        auth_verifier=verifier,
        auth_provider="static",
        auth_optional=optional,
    )
    # Inject a @requires_auth route. The app may be wrapped in _RouteAuthMiddleware,
    # so reach through to the inner Starlette app to add the route.
    inner_app = dashboard.app
    if isinstance(inner_app, _RouteAuthMiddleware):
        inner_app._router.routes.insert(0, Route("/test-protected", _protected))
    else:
        inner_app.router.routes.insert(0, Route("/test-protected", _protected))

    client = TestClient(dashboard.app)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = client.get("/test-protected", headers=headers)
    if should_allow:
        assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}"
    else:
        assert resp.status_code == 401, f"Expected 401 but got {resp.status_code}"
