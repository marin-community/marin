# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for separate auth DB via SQLite ATTACH.

Verifies the core security property: read pool connections cannot access
api_keys or controller_secrets, while the main write connection can.
"""

import sqlite3

import pytest

from iris.cluster.controller.auth import (
    JwtTokenManager,
    _get_or_create_signing_key,
    create_api_key,
    list_api_keys,
    lookup_api_key_by_hash,
    revoke_api_key,
    revoke_login_keys_for_user,
)
from iris.cluster.controller.db import ControllerDB
from iris.rpc.auth import hash_token
from iris.time_utils import Timestamp


@pytest.fixture
def db(tmp_path):
    db = ControllerDB(
        db_path=tmp_path / "controller.sqlite3",
        auth_db_path=tmp_path / "auth.sqlite3",
    )
    yield db
    db.close()


def test_read_snapshot_cannot_access_auth_tables(db: ControllerDB):
    """Core security property: read pool connections must not see auth tables."""
    now = Timestamp.now()
    db.ensure_user("test-user", now)

    _get_or_create_signing_key(db)

    create_api_key(
        db,
        key_id="k1",
        key_hash="hash1",
        key_prefix="pfx",
        user_id="test-user",
        name="test",
        now=now,
    )

    with db.read_snapshot() as q:
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM api_keys")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM controller_secrets")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM auth.api_keys")


def test_write_connection_can_access_auth_tables(db: ControllerDB):
    """The write connection (with ATTACH) can access auth tables."""
    now = Timestamp.now()
    db.ensure_user("test-user", now)

    key = _get_or_create_signing_key(db)
    assert key

    create_api_key(
        db,
        key_id="k1",
        key_hash="hash1",
        key_prefix="pfx",
        user_id="test-user",
        name="test",
        now=now,
    )

    with db.snapshot() as q:
        rows = q.raw(
            f"SELECT key_id FROM {db.api_keys_table}",
            decoders={"key_id": str},
        )
        assert len(rows) == 1
        assert rows[0].key_id == "k1"


def test_auth_operations(db: ControllerDB):
    """Full auth workflow: create user, create/list/revoke keys."""
    now = Timestamp.now()
    db.ensure_user("alice", now, role="admin")
    db.set_user_role("alice", "admin")
    assert db.get_user_role("alice") == "admin"

    create_api_key(
        db,
        key_id="k1",
        key_hash=hash_token("secret1"),
        key_prefix="sec",
        user_id="alice",
        name="my-key",
        now=now,
    )

    found = lookup_api_key_by_hash(db, hash_token("secret1"))
    assert found is not None
    assert found.key_id == "k1"

    keys = list_api_keys(db, user_id="alice")
    assert len(keys) == 1

    revoked = revoke_api_key(db, "k1", now)
    assert revoked


def test_jwt_manager(db: ControllerDB):
    """JWT token creation and verification work with auth DB."""
    now = Timestamp.now()
    db.ensure_user("bob", now, role="user")

    signing_key = _get_or_create_signing_key(db)
    mgr = JwtTokenManager(signing_key, db=db)

    create_api_key(
        db,
        key_id="k-bob",
        key_hash="jwt:k-bob",
        key_prefix="jwt",
        user_id="bob",
        name="test",
        now=now,
    )

    token = mgr.create_token("bob", "user", "k-bob")
    identity = mgr.verify(token)
    assert identity.user_id == "bob"
    assert identity.role == "user"


def test_revoke_login_keys(db: ControllerDB):
    """Login key revocation works with auth DB."""
    now = Timestamp.now()
    db.ensure_user("carol", now)

    create_api_key(
        db,
        key_id="k-login-1",
        key_hash="jwt:k-login-1",
        key_prefix="jwt",
        user_id="carol",
        name="login-111",
        now=now,
    )
    create_api_key(
        db,
        key_id="k-login-2",
        key_hash="jwt:k-login-2",
        key_prefix="jwt",
        user_id="carol",
        name="login-222",
        now=now,
    )

    revoked_ids = revoke_login_keys_for_user(db, "carol", now)
    assert set(revoked_ids) == {"k-login-1", "k-login-2"}


def test_auth_db_file_created(tmp_path):
    """The auth.sqlite3 file is created when auth_db_path is provided."""
    auth_path = tmp_path / "auth.sqlite3"
    assert not auth_path.exists()

    db = ControllerDB(
        db_path=tmp_path / "controller.sqlite3",
        auth_db_path=auth_path,
    )
    assert auth_path.exists()
    db.close()
