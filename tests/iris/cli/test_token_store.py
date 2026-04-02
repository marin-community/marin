# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from iris.cli.token_store import (
    ClusterCredential,
    cluster_name_from_url,
    load_any_token,
    load_token,
    store_token,
)


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "tokens.json"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://localhost:10000", "localhost-10000"),
        ("http://controller.example.com:8080", "controller.example.com-8080"),
        ("https://my-cluster.internal", "my-cluster.internal"),
        ("http://127.0.0.1:54321", "127.0.0.1-54321"),
    ],
)
def test_cluster_name_from_url(url: str, expected: str):
    assert cluster_name_from_url(url) == expected


def test_store_and_load_token(store_path: Path):
    store_token("local", "http://127.0.0.1:9999", "secret", store_path=store_path)

    cred = load_token("local", store_path=store_path)
    assert cred == ClusterCredential(url="http://127.0.0.1:9999", token="secret")


def test_store_token_creates_parent_dirs(tmp_path: Path):
    store_path = tmp_path / "deep" / "nested" / "tokens.json"
    store_token("c1", "http://host:1", "tok", store_path=store_path)
    assert store_path.exists()


def test_store_token_sets_file_permissions(store_path: Path):
    store_token("c1", "http://host:1", "tok", store_path=store_path)
    mode = os.stat(store_path).st_mode & 0o777
    assert mode == 0o600


def test_store_token_upserts(store_path: Path):
    store_token("c1", "http://host:1", "old", store_path=store_path)
    store_token("c1", "http://host:1", "new", store_path=store_path)

    cred = load_token("c1", store_path=store_path)
    assert cred is not None
    assert cred.token == "new"


def test_store_preserves_other_clusters(store_path: Path):
    store_token("a", "http://a:1", "ta", store_path=store_path)
    store_token("b", "http://b:2", "tb", store_path=store_path)

    assert load_token("a", store_path=store_path) is not None
    assert load_token("b", store_path=store_path) is not None


def test_load_token_missing_cluster(store_path: Path):
    store_token("a", "http://a:1", "ta", store_path=store_path)
    assert load_token("nonexistent", store_path=store_path) is None


def test_load_token_no_file(store_path: Path):
    assert load_token("anything", store_path=store_path) is None


def test_load_any_token_prefers_default(store_path: Path):
    store_token("other", "http://other:1", "t1", store_path=store_path)
    store_token("default", "http://default:2", "t2", store_path=store_path)

    cred = load_any_token(store_path=store_path)
    assert cred is not None
    assert cred.token == "t2"


def test_load_any_token_falls_back_to_first(store_path: Path):
    store_token("alpha", "http://alpha:1", "ta", store_path=store_path)

    cred = load_any_token(store_path=store_path)
    assert cred is not None
    assert cred.token == "ta"


def test_load_any_token_empty_store(store_path: Path):
    assert load_any_token(store_path=store_path) is None


def test_legacy_migration(tmp_path: Path):
    """Old ~/.iris/token file is migrated into tokens.json under 'default'."""
    legacy = tmp_path / "token"
    legacy.write_text("legacy-secret\n")
    store_path = tmp_path / "tokens.json"

    cred = load_token("default", store_path=store_path)
    assert cred is not None
    assert cred.token == "legacy-secret"
    assert not legacy.exists(), "legacy file should be deleted after migration"


def test_legacy_migration_empty_token(tmp_path: Path):
    """Empty legacy token file is deleted without creating a store entry."""
    legacy = tmp_path / "token"
    legacy.write_text("  \n")
    store_path = tmp_path / "tokens.json"

    cred = load_any_token(store_path=store_path)
    assert cred is None
    assert not legacy.exists()


def test_store_format(store_path: Path):
    """Verify the on-disk JSON matches the expected schema."""
    store_token("local", "http://127.0.0.1:54321", "abc", store_path=store_path)
    data = json.loads(store_path.read_text())
    assert data == {
        "clusters": {
            "local": {"url": "http://127.0.0.1:54321", "token": "abc"},
        },
    }
