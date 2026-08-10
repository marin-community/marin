# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opaque per-cluster credential store."""

import stat

from rigging import credential_store
from rigging.credential_store import (
    CredentialRecord,
    delete_credentials,
    load_credentials,
    save_credentials,
)


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path)
    record = CredentialRecord(
        cluster="marin",
        endpoint="https://iris-marin.oa.dev",
        edge_refresh_token="refresh",
        metadata={"user_id": "alice@x"},
    )
    save_credentials(record)
    loaded = load_credentials("marin")
    assert loaded == record


def test_load_missing_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path)
    assert load_credentials("nope") is None


def test_file_and_dir_are_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path / "creds")
    path = save_credentials(CredentialRecord(cluster="marin", endpoint="https://e", edge_refresh_token="t"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_save_overwrites_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path)
    save_credentials(CredentialRecord(cluster="marin", endpoint="https://e", edge_refresh_token="old"))
    save_credentials(CredentialRecord(cluster="marin", endpoint="https://e", edge_refresh_token="new"))
    loaded = load_credentials("marin")
    assert loaded is not None and loaded.edge_refresh_token == "new"


def test_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path)
    save_credentials(CredentialRecord(cluster="marin", endpoint="https://e"))
    assert delete_credentials("marin") is True
    assert delete_credentials("marin") is False
    assert load_credentials("marin") is None
