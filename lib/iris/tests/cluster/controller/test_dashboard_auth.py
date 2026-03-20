# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for default-deny auth middleware on Starlette HTTP routes."""

from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from iris.cluster.bundle import BundleStore
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.log_store import LogStore
from iris.rpc.auth import SESSION_COOKIE, StaticTokenVerifier

_TEST_TOKEN = "valid-test-token"
_TEST_USER = "test-user"


@pytest.fixture
def state(tmp_path):
    db = ControllerDB(db_path=tmp_path / "controller.sqlite3")
    log_store = LogStore(log_dir=tmp_path / "logs")
    s = ControllerTransitions(db=db, log_store=log_store)
    yield s
    log_store.close()
    db.close()


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
    """Client with auth middleware enabled."""
    dashboard = ControllerDashboard(service, auth_verifier=verifier, auth_provider="gcp")
    return TestClient(dashboard.app)


@pytest.fixture
def unauthed_client(service):
    """Client with auth disabled (no middleware)."""
    dashboard = ControllerDashboard(service)
    return TestClient(dashboard.app)


# -- Protected routes return 401 without auth --


def test_job_page_requires_auth(authed_client):
    resp = authed_client.get("/job/123")
    assert resp.status_code == 401
    assert resp.json()["error"] == "authentication required"


def test_worker_page_requires_auth(authed_client):
    resp = authed_client.get("/worker/456")
    assert resp.status_code == 401
    assert resp.json()["error"] == "authentication required"


def test_bundle_download_requires_auth(authed_client):
    resp = authed_client.get("/bundles/" + "a" * 64 + ".zip")
    assert resp.status_code == 401


def test_dashboard_root_requires_auth(authed_client):
    resp = authed_client.get("/")
    assert resp.status_code == 401


# -- Invalid token returns 401 --


def test_invalid_token_returns_401(authed_client):
    resp = authed_client.get("/job/123", cookies={SESSION_COOKIE: "bad-token"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid session"


# -- Valid session cookie grants access --


def test_authenticated_job_page(authed_client):
    resp = authed_client.get("/job/123", cookies={SESSION_COOKIE: _TEST_TOKEN})
    assert resp.status_code == 200


def test_authenticated_worker_page(authed_client):
    resp = authed_client.get("/worker/456", cookies={SESSION_COOKIE: _TEST_TOKEN})
    assert resp.status_code == 200


def test_authenticated_via_bearer_header(authed_client):
    resp = authed_client.get("/job/123", headers={"Authorization": f"Bearer {_TEST_TOKEN}"})
    assert resp.status_code == 200


# -- Public paths skip auth --


def test_health_accessible_without_auth(authed_client):
    resp = authed_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_auth_config_accessible_without_auth(authed_client):
    resp = authed_client.get("/auth/config")
    assert resp.status_code == 200
    assert resp.json()["auth_enabled"] is True


def test_static_accessible_without_auth(authed_client):
    # Static mount may 404 (no actual files), but should NOT 401
    resp = authed_client.get("/static/nonexistent.js")
    assert resp.status_code != 401


def test_rpc_routes_skip_middleware(authed_client):
    """RPC routes use their own interceptor chain, not the HTTP middleware."""
    resp = authed_client.post(
        "/iris.cluster.ControllerService/GetAuthInfo",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code != 401


# -- No middleware when auth disabled --


def test_no_middleware_when_auth_disabled(unauthed_client):
    """All routes accessible when auth is not configured."""
    assert unauthed_client.get("/job/123").status_code == 200
    assert unauthed_client.get("/worker/456").status_code == 200
    assert unauthed_client.get("/health").status_code == 200
    assert unauthed_client.get("/auth/config").status_code == 200
