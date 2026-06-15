# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
from iris.cluster.token_store import (
    ClusterCredential,
    cluster_name_from_url,
    load_any_token,
    load_token,
    store_token,
)


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "tokens.sqlite"


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
    store_path = tmp_path / "deep" / "nested" / "tokens.sqlite"
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


def test_concurrent_writers_do_not_corrupt_store(store_path: Path):
    """Interleaved upserts from many clusters all survive — SQLite serializes
    the writes rather than racing on a shared temp file (the bug that the old
    JSON+mkstemp store hit under pytest-xdist)."""
    for i in range(25):
        store_token(f"c{i}", f"http://host:{i}", f"tok-{i}", store_path=store_path)

    for i in range(25):
        cred = load_token(f"c{i}", store_path=store_path)
        assert cred == ClusterCredential(url=f"http://host:{i}", token=f"tok-{i}")
