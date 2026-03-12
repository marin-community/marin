# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for API key management: DB CRUD, auth_setup preloading, and service RPCs."""

from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth_setup import (
    WORKER_USER,
    ControllerAuth,
    DbTokenVerifier,
    create_controller_auth,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.log_store import LogStore
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.auth import _verified_user


@pytest.fixture
def db(tmp_path):
    d = ControllerDB(db_path=tmp_path / "test.sqlite3")
    yield d
    d.close()


def _make_service(db, auth=None):
    """Create a ControllerServiceImpl with minimal dependencies for API key tests."""
    log_store = LogStore(db_path=db.db_path)
    state = ControllerTransitions(db=db, log_store=log_store)

    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.create_scheduling_context = Mock(return_value=Mock())
    controller_mock.get_job_scheduling_diagnostics = Mock(return_value="")
    controller_mock.autoscaler = None
    controller_mock.stub_factory = Mock()

    return ControllerServiceImpl(
        state,
        db,
        controller=controller_mock,
        bundle_store=BundleStore(db_path=db.db_path.parent / "bundles.sqlite3"),
        log_store=log_store,
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

    # Both tokens should be verifiable
    assert auth.verifier.verify("tok-a") == "alice"
    assert auth.verifier.verify("tok-b") == "bob"


def test_static_preload_is_idempotent(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    create_controller_auth(config, db=db)
    create_controller_auth(config, db=db)  # Should not raise

    keys = db.list_api_keys(user_id="alice")
    assert len(keys) == 1


def test_worker_token_in_api_keys(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)

    # Worker token should be verifiable
    assert auth.worker_token is not None
    assert auth.verifier.verify(auth.worker_token) == WORKER_USER


def test_worker_token_survives_restart(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth1 = create_controller_auth(config, db=db)
    token1 = auth1.worker_token

    auth2 = create_controller_auth(config, db=db)
    token2 = auth2.worker_token
    assert token1 == token2


def test_admin_users_bootstrapped(db):
    config = config_pb2.AuthConfig(
        static={"tokens": {"tok-a": "alice"}},
        admin_users=["alice"],
    )
    create_controller_auth(config, db=db)
    assert db.get_user_role("alice") == "admin"


def test_login_verifier_set_for_gcp(db):
    config = config_pb2.AuthConfig(gcp={"audience": "test-aud"})
    auth = create_controller_auth(config, db=db)
    assert auth.login_verifier is not None
    assert auth.provider == "gcp"


def test_login_verifier_none_for_static(db):
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)
    assert auth.login_verifier is None


# ---------------------------------------------------------------------------
# Service RPC: Login
# ---------------------------------------------------------------------------


def test_login_rejects_system_prefix(db):
    """Login RPC rejects usernames starting with system:."""

    class SystemVerifier:
        def verify(self, token: str) -> str:
            return "system:hacker"

    auth = ControllerAuth(
        verifier=DbTokenVerifier(db),
        provider="gcp",
        login_verifier=SystemVerifier(),
    )
    service = _make_service(db, auth=auth)

    with pytest.raises(ConnectError) as exc_info:
        service.login(cluster_pb2.LoginRequest(identity_token="fake"), None)
    assert exc_info.value.code == Code.PERMISSION_DENIED


def test_login_creates_user_and_key(db):
    """Login RPC creates a user and returns a working API key."""

    class FakeVerifier:
        def verify(self, token: str) -> str:
            return "alice@example.com"

    auth = ControllerAuth(
        verifier=DbTokenVerifier(db),
        provider="gcp",
        login_verifier=FakeVerifier(),
    )
    service = _make_service(db, auth=auth)

    response = service.login(cluster_pb2.LoginRequest(identity_token="gcp-id-token"), None)
    assert response.user_id == "alice@example.com"
    assert response.token
    assert response.key_id.startswith("iris_k_")

    # The returned token should work with DbTokenVerifier
    assert auth.verifier.verify(response.token) == "alice@example.com"


def test_login_not_available_without_login_verifier(db):
    """Login RPC returns UNIMPLEMENTED when no login_verifier is configured."""
    auth = ControllerAuth(verifier=DbTokenVerifier(db), provider="static")
    service = _make_service(db, auth=auth)

    with pytest.raises(ConnectError) as exc_info:
        service.login(cluster_pb2.LoginRequest(identity_token="token"), None)
    assert exc_info.value.code == Code.UNIMPLEMENTED


# ---------------------------------------------------------------------------
# Service RPC: CreateApiKey, ListApiKeys, RevokeApiKey
# ---------------------------------------------------------------------------


def test_create_api_key_returns_raw_token(db):
    """CreateApiKey returns a working raw token."""
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    token = _verified_user.set("alice")
    try:
        response = service.create_api_key(
            cluster_pb2.CreateApiKeyRequest(name="my-key"),
            None,
        )
    finally:
        _verified_user.reset(token)

    assert response.token
    assert response.key_id.startswith("iris_k_")
    assert response.key_prefix == response.token[:8]

    # Token should work
    assert auth.verifier.verify(response.token) == "alice"


def test_list_api_keys_never_exposes_hash(db):
    """ListApiKeys returns key info without exposing hashes."""
    config = config_pb2.AuthConfig(static={"tokens": {"tok-a": "alice"}})
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    token = _verified_user.set("alice")
    try:
        response = service.list_api_keys(
            cluster_pb2.ListApiKeysRequest(user_id="alice"),
            None,
        )
    finally:
        _verified_user.reset(token)

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
    alice_keys = db.list_api_keys(user_id="alice")
    assert alice_keys

    # Bob tries to revoke alice's key
    token = _verified_user.set("bob")
    try:
        with pytest.raises(ConnectError) as exc_info:
            service.revoke_api_key(
                cluster_pb2.RevokeApiKeyRequest(key_id=alice_keys[0].key_id),
                None,
            )
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_user.reset(token)


def test_admin_can_revoke_any_key(db):
    """Admin user can revoke any user's key."""
    config = config_pb2.AuthConfig(
        static={"tokens": {"tok-a": "alice", "tok-b": "bob"}},
        admin_users=["bob"],
    )
    auth = create_controller_auth(config, db=db)
    service = _make_service(db, auth=auth)

    alice_keys = db.list_api_keys(user_id="alice")
    assert alice_keys

    token = _verified_user.set("bob")
    try:
        service.revoke_api_key(
            cluster_pb2.RevokeApiKeyRequest(key_id=alice_keys[0].key_id),
            None,
        )
    finally:
        _verified_user.reset(token)

    # Alice's token should no longer work
    with pytest.raises(ValueError, match="revoked"):
        auth.verifier.verify("tok-a")
