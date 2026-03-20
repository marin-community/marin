# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth: session cookies, CSRF, default-deny middleware, auth DB isolation, API keys, and JWT."""

import sqlite3
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from iris.cluster.bundle import BundleStore
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
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.log_store import LogStore
from iris.rpc.auth import SESSION_COOKIE, StaticTokenVerifier, hash_token
from iris.time_utils import Timestamp

_TEST_TOKEN = "valid-test-token"
_TEST_USER = "test-user"
CSRF_HEADERS = {"Origin": "http://testserver"}


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    db = ControllerDB(db_path=tmp_path / "controller.sqlite3", auth_db_path=tmp_path / "auth.sqlite3")
    yield db
    db.close()


@pytest.fixture
def state(db, tmp_path):
    log_store = LogStore(log_dir=tmp_path / "logs")
    s = ControllerTransitions(db=db, log_store=log_store)
    yield s
    log_store.close()


@pytest.fixture
def service(state, tmp_path):
    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.autoscaler = None
    controller_mock.provider = Mock()
    controller_mock.has_direct_provider = False
    return ControllerServiceImpl(
        state,
        state._db,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_store=state._log_store,
    )


@pytest.fixture
def verifier():
    return StaticTokenVerifier({_TEST_TOKEN: _TEST_USER})


@pytest.fixture
def authed_client(service, verifier):
    dashboard = ControllerDashboard(service, auth_verifier=verifier, auth_provider="gcp")
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
    db = ControllerDB(db_path=tmp_path / "controller.sqlite3", auth_db_path=auth_path)
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
