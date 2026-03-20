# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for dashboard auth session and CSRF protection."""

import pytest
from starlette.testclient import TestClient

from iris.cluster.bundle import BundleStore
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.scheduler import Scheduler
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.log_store import LogStore
from iris.rpc.auth import StaticTokenVerifier

from .test_dashboard import _make_controller_mock


@pytest.fixture
def state(tmp_path):
    db_path = tmp_path / "controller.sqlite3"
    db = ControllerDB(db_path=db_path)
    log_store = LogStore(log_dir=tmp_path / "logs")
    s = ControllerTransitions(db=db, log_store=log_store)
    yield s
    log_store.close()
    db.close()


@pytest.fixture
def service(state, tmp_path):
    controller_mock = _make_controller_mock(state, Scheduler())
    return ControllerServiceImpl(
        state,
        state._db,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_store=state._log_store,
    )


@pytest.fixture
def verifier():
    return StaticTokenVerifier({"valid-token": "test-user"})


@pytest.fixture
def authed_client(service, verifier):
    """Client with auth enabled."""
    dashboard = ControllerDashboard(service, auth_verifier=verifier, auth_provider="test")
    return TestClient(dashboard.app)


@pytest.fixture
def noauth_client(service):
    """Client with auth disabled (no verifier)."""
    dashboard = ControllerDashboard(service)
    return TestClient(dashboard.app)


CSRF_HEADERS = {"Origin": "http://testserver"}


# =============================================================================
# Token verification (F-1)
# =============================================================================


def test_auth_session_rejects_invalid_token(authed_client):
    resp = authed_client.post(
        "/auth/session",
        json={"token": "bad-token"},
        headers=CSRF_HEADERS,
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid token"


def test_auth_session_accepts_valid_token(authed_client):
    resp = authed_client.post(
        "/auth/session",
        json={"token": "valid-token"},
        headers=CSRF_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "iris_session" in resp.cookies


def test_auth_session_returns_400_for_empty_token(authed_client):
    resp = authed_client.post(
        "/auth/session",
        json={"token": "  "},
        headers=CSRF_HEADERS,
    )
    assert resp.status_code == 400


def test_auth_session_skips_verification_when_auth_disabled(noauth_client):
    resp = noauth_client.post(
        "/auth/session",
        json={"token": "any-token-works"},
        headers=CSRF_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# =============================================================================
# CSRF protection (F-8)
# =============================================================================


def test_csrf_rejects_mismatched_origin(authed_client):
    resp = authed_client.post(
        "/auth/session",
        json={"token": "valid-token"},
        headers={"Origin": "http://evil.example.com"},
    )
    assert resp.status_code == 403
    assert "CSRF" in resp.json()["error"]


def test_csrf_rejects_missing_origin_and_referer(authed_client):
    resp = authed_client.post(
        "/auth/session",
        json={"token": "valid-token"},
        # No Origin or Referer header
    )
    assert resp.status_code == 403


def test_csrf_accepts_matching_origin(authed_client):
    resp = authed_client.post(
        "/auth/session",
        json={"token": "valid-token"},
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code == 200


def test_csrf_accepts_matching_referer(authed_client):
    resp = authed_client.post(
        "/auth/session",
        json={"token": "valid-token"},
        headers={"Referer": "http://testserver/auth/login"},
    )
    assert resp.status_code == 200


def test_csrf_on_logout_rejects_missing_origin(authed_client):
    resp = authed_client.post("/auth/logout")
    assert resp.status_code == 403


def test_csrf_on_logout_accepts_matching_origin(authed_client):
    resp = authed_client.post(
        "/auth/logout",
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code == 200
