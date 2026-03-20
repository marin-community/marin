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
def db_with_auth(tmp_path):
    """ControllerDB with a separate auth database."""
    db = ControllerDB(
        db_path=tmp_path / "controller.sqlite3",
        auth_db_path=tmp_path / "auth.sqlite3",
    )
    yield db
    db.close()


@pytest.fixture
def db_without_auth(tmp_path):
    """ControllerDB in legacy single-DB mode (no auth separation)."""
    db = ControllerDB(db_path=tmp_path / "controller.sqlite3")
    yield db
    db.close()


def test_read_snapshot_cannot_access_auth_tables(db_with_auth: ControllerDB):
    """Core security property: read pool connections must not see auth tables."""
    now = Timestamp.now()
    db_with_auth.ensure_user("test-user", now)

    # Write a signing key so controller_secrets has data
    _get_or_create_signing_key(db_with_auth)

    # Create an API key so api_keys has data
    create_api_key(
        db_with_auth,
        key_id="k1",
        key_hash="hash1",
        key_prefix="pfx",
        user_id="test-user",
        name="test",
        now=now,
    )

    # read_snapshot uses the read pool (no auth DB attached)
    with db_with_auth.read_snapshot() as q:
        # api_keys should not be accessible
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM api_keys")

        # controller_secrets should not be accessible
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM controller_secrets")

        # auth-qualified names should also fail
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM auth.api_keys")


def test_write_connection_can_access_auth_tables(db_with_auth: ControllerDB):
    """The write connection (with ATTACH) can access auth tables."""
    now = Timestamp.now()
    db_with_auth.ensure_user("test-user", now)

    key = _get_or_create_signing_key(db_with_auth)
    assert key  # non-empty signing key

    create_api_key(
        db_with_auth,
        key_id="k1",
        key_hash="hash1",
        key_prefix="pfx",
        user_id="test-user",
        name="test",
        now=now,
    )

    # snapshot() uses the write connection
    with db_with_auth.snapshot() as q:
        rows = q.raw(
            f"SELECT key_id FROM {db_with_auth.api_keys_table}",
            decoders={"key_id": str},
        )
        assert len(rows) == 1
        assert rows[0].key_id == "k1"


def test_auth_operations_with_separate_db(db_with_auth: ControllerDB):
    """Full auth workflow: create user, create/list/revoke keys."""
    now = Timestamp.now()
    db_with_auth.ensure_user("alice", now, role="admin")
    db_with_auth.set_user_role("alice", "admin")
    assert db_with_auth.get_user_role("alice") == "admin"

    create_api_key(
        db_with_auth,
        key_id="k1",
        key_hash=hash_token("secret1"),
        key_prefix="sec",
        user_id="alice",
        name="my-key",
        now=now,
    )

    found = lookup_api_key_by_hash(db_with_auth, hash_token("secret1"))
    assert found is not None
    assert found.key_id == "k1"

    keys = list_api_keys(db_with_auth, user_id="alice")
    assert len(keys) == 1

    revoked = revoke_api_key(db_with_auth, "k1", now)
    assert revoked


def test_jwt_manager_with_separate_db(db_with_auth: ControllerDB):
    """JWT token creation and verification work with auth DB."""
    now = Timestamp.now()
    db_with_auth.ensure_user("bob", now, role="user")

    signing_key = _get_or_create_signing_key(db_with_auth)
    mgr = JwtTokenManager(signing_key, db=db_with_auth)

    create_api_key(
        db_with_auth,
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


def test_revoke_login_keys_with_separate_db(db_with_auth: ControllerDB):
    """Login key revocation works with auth DB."""
    now = Timestamp.now()
    db_with_auth.ensure_user("carol", now)

    create_api_key(
        db_with_auth,
        key_id="k-login-1",
        key_hash="jwt:k-login-1",
        key_prefix="jwt",
        user_id="carol",
        name="login-111",
        now=now,
    )
    create_api_key(
        db_with_auth,
        key_id="k-login-2",
        key_hash="jwt:k-login-2",
        key_prefix="jwt",
        user_id="carol",
        name="login-222",
        now=now,
    )

    revoked_ids = revoke_login_keys_for_user(db_with_auth, "carol", now)
    assert set(revoked_ids) == {"k-login-1", "k-login-2"}


def test_migration_from_single_to_split_db(tmp_path):
    """Simulate upgrading from single DB to split DB with auth separation."""
    db_path = tmp_path / "controller.sqlite3"

    # Phase 1: Create a single-DB instance (no auth separation)
    db_single = ControllerDB(db_path=db_path)
    now = Timestamp.now()
    db_single.ensure_user("migrated-user", now, role="admin")
    create_api_key(
        db_single,
        key_id="old-key",
        key_hash="old-hash",
        key_prefix="old",
        user_id="migrated-user",
        name="old-key",
        now=now,
    )
    # Write a signing key
    _get_or_create_signing_key(db_single)
    db_single.close()

    # Phase 2: Re-open with auth DB — migration should move data
    auth_path = tmp_path / "auth.sqlite3"
    db_split = ControllerDB(db_path=db_path, auth_db_path=auth_path)

    # Auth data should be accessible via write connection
    signing_key = _get_or_create_signing_key(db_split)
    assert signing_key  # should find the previously created key

    keys = list_api_keys(db_split, user_id="migrated-user")
    assert len(keys) == 1
    assert keys[0].key_id == "old-key"

    # Read pool should NOT see auth tables
    with db_split.read_snapshot() as q:
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM api_keys")
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            q.raw("SELECT * FROM controller_secrets")

    # Users table should still be in main (accessible from read pool)
    with db_split.read_snapshot() as q:
        rows = q.raw("SELECT user_id FROM users", decoders={"user_id": str})
        assert any(r.user_id == "migrated-user" for r in rows)

    db_split.close()


def test_legacy_mode_still_works(db_without_auth: ControllerDB):
    """Single-DB mode (no auth_db_path) continues to work."""
    now = Timestamp.now()
    db_without_auth.ensure_user("legacy-user", now)

    create_api_key(
        db_without_auth,
        key_id="legacy-key",
        key_hash="legacy-hash",
        key_prefix="lgc",
        user_id="legacy-user",
        name="legacy",
        now=now,
    )

    key = lookup_api_key_by_hash(db_without_auth, "legacy-hash")
    assert key is not None
    assert key.key_id == "legacy-key"

    # In legacy mode, read_snapshot CAN access api_keys (no separation)
    with db_without_auth.read_snapshot() as q:
        rows = q.raw("SELECT key_id FROM api_keys", decoders={"key_id": str})
        assert len(rows) == 1


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
